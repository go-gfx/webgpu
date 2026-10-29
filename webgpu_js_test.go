//go:build js

package webgpu

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"sync"
	"testing"
	"time"
)

// These tests run in a browser worker (internal/cmd/wasmbrowser). With
// WEBGPU_REQUIRE_GPU=1 a fallback (software) adapter fails them: that is
// the macOS CI lane, which must prove a real GPU. WEBGPU_ALLOW_FALLBACK=1
// runs them on SwiftShader and says so.

var adapterOnce sync.Once

func ctx(t *testing.T) context.Context {
	c, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	t.Cleanup(cancel)
	return c
}

// getAdapter requests a fresh adapter: an adapter is "consumed" by the
// device it creates.
func getAdapter(t *testing.T) *Adapter {
	t.Helper()
	g, err := Navigator()
	if err != nil {
		t.Fatal(err)
	}
	allow := os.Getenv("WEBGPU_ALLOW_FALLBACK") == "1"
	adapter, err := g.RequestAdapter(ctx(t), &RequestAdapterOptions{AllowFallback: allow})
	if err != nil {
		t.Fatalf("adapter: %v", err)
	}
	adapterOnce.Do(func() {
		info := adapter.Info()
		t.Logf("adapter: vendor %q, architecture %q, description %q, fallback %v", info.Vendor, info.Architecture, info.Description, info.Fallback)
	})
	if os.Getenv("WEBGPU_REQUIRE_GPU") == "1" && adapter.Info().Fallback {
		t.Fatal("a fallback adapter where a GPU is required")
	}
	return adapter
}

func newDev(t *testing.T, q *Quota) *Device {
	t.Helper()
	d, err := getAdapter(t).RequestDevice(ctx(t), &DeviceDescriptor{Label: t.Name(), Quota: q})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Destroy)
	return d
}

func must[T any](v T, err error) func(t *testing.T) T {
	return func(t *testing.T) T {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
}

func TestNavigatorAndAdapter(t *testing.T) {
	a := getAdapter(t)
	if v, ok := a.Limit("maxBufferSize"); !ok || v < 1<<20 {
		t.Fatalf("maxBufferSize %d %v", v, ok)
	}
	g, _ := Navigator()
	switch f := g.PreferredCanvasFormat(); f {
	case TextureFormatBGRA8Unorm, TextureFormatRGBA8Unorm:
	default:
		t.Fatalf("preferred canvas format %q", f)
	}
	_ = a.HasFeature(FeatureTimestampQuery)
}

func TestQuotaRefusesBeforeTheBrowser(t *testing.T) {
	q := &Quota{MemoryBytes: 1 << 16, Pipelines: 1, ShaderBytes: 64}
	// A feature the quota does not list is refused even if the adapter has it.
	_, err := getAdapter(t).RequestDevice(ctx(t), &DeviceDescriptor{Quota: q, RequiredFeatures: []FeatureName{FeatureTimestampQuery}})
	var qe *QuotaError
	if !errors.As(err, &qe) {
		t.Fatalf("timestamp-query: %v", err)
	}
	d := newDev(t, q)
	b := must(d.CreateBuffer(BufferDescriptor{Size: 1 << 15, Usage: BufferUsageCopyDst}))(t)
	if _, err := d.CreateBuffer(BufferDescriptor{Size: 1<<15 + 4, Usage: BufferUsageCopyDst}); !errors.As(err, &qe) {
		t.Fatalf("over memory: %v", err)
	}
	if _, err := d.CreateTexture(TextureDescriptor{Size: Extent3D{Width: 256, Height: 256}, Format: "r8uint", Usage: TextureUsageCopyDst}); !errors.As(err, &qe) {
		t.Fatalf("texture over memory: %v", err)
	}
	b.Destroy()
	b.Destroy() // refunds once
	if m, _ := q.Usage(); m != 0 {
		t.Fatalf("usage %d after destroy", m)
	}
	if _, err := d.CreateShaderModule(ShaderModuleDescriptor{Code: string(make([]byte, 65))}); !errors.As(err, &qe) {
		t.Fatalf("shader: %v", err)
	}
}

func TestBufferRoundTripAndBounds(t *testing.T) {
	d := newDev(t, nil)
	src := must(d.CreateBuffer(BufferDescriptor{Label: "src", Size: 256, Usage: BufferUsageStorage | BufferUsageCopyDst | BufferUsageCopySrc}))(t)
	dst := must(d.CreateBuffer(BufferDescriptor{Label: "dst", Size: 256, Usage: BufferUsageMapRead | BufferUsageCopyDst}))(t)
	data := make([]byte, 256)
	for i := range data {
		data[i] = byte(i * 7)
	}
	if err := d.Queue().WriteBuffer(src, 0, data); err != nil {
		t.Fatal(err)
	}
	var be *BoundsError
	if err := d.Queue().WriteBuffer(src, 4, data); !errors.As(err, &be) {
		t.Fatalf("out-of-range write: %v", err)
	}
	noDst := must(d.CreateBuffer(BufferDescriptor{Size: 16, Usage: BufferUsageStorage}))(t)
	if err := d.Queue().WriteBuffer(noDst, 0, data[:4]); !errors.As(err, &be) {
		t.Fatalf("write without COPY_DST: %v", err)
	}
	enc := must(d.CreateCommandEncoder("copy"))(t)
	if err := enc.CopyBufferToBuffer(src, 0, dst, 0, 256); err != nil {
		t.Fatal(err)
	}
	if err := enc.CopyBufferToBuffer(src, 4, dst, 0, 256); !errors.As(err, &be) {
		t.Fatalf("out-of-range copy: %v", err)
	}
	cb := must(enc.Finish())(t)
	if err := d.Queue().Submit(cb); err != nil {
		t.Fatal(err)
	}
	if err := dst.MapAsync(ctx(t), MapModeRead, 0, 0); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, 256)
	if err := dst.ReadMapped(0, got); err != nil {
		t.Fatal(err)
	}
	if err := dst.Unmap(); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if got[i] != data[i] {
			t.Fatalf("byte %d: %d, want %d", i, got[i], data[i])
		}
	}
	if err := dst.ReadMapped(0, make([]byte, 300)); !errors.As(err, &be) {
		t.Fatalf("read past the buffer: %v", err)
	}
	// Unmapped: the browser throws, and that is an error, not a panic.
	var je *JSError
	if err := dst.ReadMapped(0, got); !errors.As(err, &je) {
		t.Fatalf("read while unmapped: %v", err)
	}
}

func TestMappedAtCreation(t *testing.T) {
	d := newDev(t, nil)
	b := must(d.CreateBuffer(BufferDescriptor{Size: 16, Usage: BufferUsageMapWrite | BufferUsageCopySrc, MappedAtCreation: true}))(t)
	if err := b.WriteMapped(0, []byte{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if err := b.Unmap(); err != nil {
		t.Fatal(err)
	}
}

const quakeWGSL = `
struct Inst { off: vec4f };
@group(0) @binding(0) var<storage, read> insts: array<Inst>;
@group(0) @binding(1) var idxTex: texture_2d<u32>;
@group(0) @binding(2) var palTex: texture_2d<f32>;
@group(0) @binding(3) var palSamp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) inst: u32,
};

@vertex fn vs(@location(0) p: vec2f, @location(1) uv: vec2f, @builtin(instance_index) i: u32) -> VOut {
  var o: VOut;
  let d = insts[i].off;
  o.pos = vec4f(p.x + d.x, p.y, d.y, 1.0);
  o.uv = uv;
  o.inst = i;
  return o;
}

@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let t = vec2u(clamp(v.uv * 4.0, vec2f(0.0), vec2f(3.0)));
  let idx = textureLoad(idxTex, t, 0).r;
  let c = textureSample(palTex, palSamp, vec2f((f32(idx) + 0.5) / 256.0, 0.5));
  if (v.inst == 2u) { return vec4f(1.0, 0.0, 1.0, 1.0); }
  return c;
}
`

func f32s(v ...float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

func palette(k int) [4]byte { return [4]byte{byte(k * 10), byte(255 - k*10), byte(k), 255} }

// scene is a small GLQuake-shaped frame: an 8-bit index texture looked up
// through a palette texture with a sampler, vertex and index buffers,
// instancing with per-instance data in a storage buffer, a depth buffer
// (instance 2 is behind instance 0 and must not show), an explicit bind
// group layout and pipeline layout.
type scene struct {
	d                  *Device
	idx, target, depth *Texture
	targetView, depthV *TextureView
	pipe               *RenderPipeline
	group              *BindGroup
	vb, ib, readback   *Buffer
}

func newScene(t *testing.T, d *Device) *scene {
	s := &scene{d: d}
	q := d.Queue()
	s.idx = must(d.CreateTexture(TextureDescriptor{Label: "index", Size: Extent3D{Width: 4, Height: 4}, Format: TextureFormatR8Uint,
		Usage: TextureUsageTextureBinding | TextureUsageCopyDst}))(t)
	idx := make([]byte, 16)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			idx[y*4+x] = byte(1 + x + 4*y)
		}
	}
	if err := q.WriteTexture(TexelCopyTextureInfo{Texture: s.idx}, idx, TexelCopyBufferLayout{BytesPerRow: 4}, Extent3D{Width: 4, Height: 4}); err != nil {
		t.Fatal(err)
	}
	pal := must(d.CreateTexture(TextureDescriptor{Label: "palette", Size: Extent3D{Width: 256, Height: 1}, Format: TextureFormatRGBA8Unorm,
		Usage: TextureUsageTextureBinding | TextureUsageCopyDst}))(t)
	pb := make([]byte, 256*4)
	for k := 0; k < 256; k++ {
		c := palette(k)
		copy(pb[4*k:], c[:])
	}
	if err := q.WriteTexture(TexelCopyTextureInfo{Texture: pal}, pb, TexelCopyBufferLayout{}, Extent3D{Width: 256}); err != nil {
		t.Fatal(err)
	}
	samp := must(d.CreateSampler(&SamplerDescriptor{MagFilter: FilterModeNearest, MinFilter: FilterModeNearest}))(t)

	s.vb = must(d.CreateBuffer(BufferDescriptor{Label: "vertices", Size: 64, Usage: BufferUsageVertex | BufferUsageCopyDst}))(t)
	if err := q.WriteBuffer(s.vb, 0, f32s(-1, 1, 0, 0, 0, 1, 1, 0, -1, -1, 0, 1, 0, -1, 1, 1)); err != nil {
		t.Fatal(err)
	}
	s.ib = must(d.CreateBuffer(BufferDescriptor{Label: "indices", Size: 12, Usage: BufferUsageIndex | BufferUsageCopyDst}))(t)
	ib := make([]byte, 12)
	for i, v := range []uint16{0, 1, 2, 2, 1, 3} {
		binary.LittleEndian.PutUint16(ib[2*i:], v)
	}
	if err := q.WriteBuffer(s.ib, 0, ib); err != nil {
		t.Fatal(err)
	}
	inst := must(d.CreateBuffer(BufferDescriptor{Label: "instances", Size: 48, Usage: BufferUsageStorage | BufferUsageCopyDst}))(t)
	if err := q.WriteBuffer(inst, 0, f32s(0, 0.5, 0, 0, 1, 0.5, 0, 0, 0, 0.9, 0, 0)); err != nil {
		t.Fatal(err)
	}

	bgl := must(d.CreateBindGroupLayout(BindGroupLayoutDescriptor{Entries: []BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageVertex, Buffer: &BufferBindingLayout{Type: BufferBindingTypeReadOnlyStorage}},
		{Binding: 1, Visibility: ShaderStageFragment, Texture: &TextureBindingLayout{SampleType: TextureSampleTypeUint}},
		{Binding: 2, Visibility: ShaderStageFragment, Texture: &TextureBindingLayout{SampleType: TextureSampleTypeFloat}},
		{Binding: 3, Visibility: ShaderStageFragment, Sampler: &SamplerBindingLayout{Type: SamplerBindingTypeFiltering}},
	}}))(t)
	pl := must(d.CreatePipelineLayout(PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}}))(t)
	idxView := must(s.idx.CreateView(nil))(t)
	palView := must(pal.CreateView(nil))(t)
	s.group = must(d.CreateBindGroup(BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{
		{Binding: 0, Buffer: inst},
		{Binding: 1, TextureView: idxView},
		{Binding: 2, TextureView: palView},
		{Binding: 3, Sampler: samp},
	}}))(t)
	mod := must(d.CreateShaderModule(ShaderModuleDescriptor{Label: "quake", Code: quakeWGSL}))(t)
	s.pipe = must(d.CreateRenderPipelineAsync(ctx(t), RenderPipelineDescriptor{
		Layout: pl,
		Vertex: VertexState{Module: mod, EntryPoint: "vs", Buffers: []VertexBufferLayout{{ArrayStride: 16, Attributes: []VertexAttribute{
			{Format: VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
			{Format: VertexFormatFloat32x2, Offset: 8, ShaderLocation: 1},
		}}}},
		Primitive:    PrimitiveState{Topology: PrimitiveTopologyTriangleList, CullMode: CullModeNone},
		DepthStencil: &DepthStencilState{Format: TextureFormatDepth24Plus, DepthWriteEnabled: true, DepthCompare: CompareFunctionLess},
		Fragment:     &FragmentState{Module: mod, EntryPoint: "fs", Targets: []ColorTargetState{{Format: TextureFormatRGBA8Unorm}}},
	}))(t)

	s.target = must(d.CreateTexture(TextureDescriptor{Size: Extent3D{Width: 16, Height: 16}, Format: TextureFormatRGBA8Unorm,
		Usage: TextureUsageRenderAttachment | TextureUsageCopySrc}))(t)
	s.targetView = must(s.target.CreateView(nil))(t)
	s.depth = must(d.CreateTexture(TextureDescriptor{Size: Extent3D{Width: 16, Height: 16}, Format: TextureFormatDepth24Plus, Usage: TextureUsageRenderAttachment}))(t)
	s.depthV = must(s.depth.CreateView(nil))(t)
	s.readback = must(d.CreateBuffer(BufferDescriptor{Size: 256 * 16, Usage: BufferUsageMapRead | BufferUsageCopyDst}))(t)
	return s
}

func (s *scene) pass() RenderPassDescriptor {
	return RenderPassDescriptor{
		ColorAttachments:       []*RenderPassColorAttachment{{View: s.targetView, LoadOp: LoadOpClear, StoreOp: StoreOpStore}},
		DepthStencilAttachment: &RenderPassDepthStencilAttachment{View: s.depthV, DepthClearValue: 1, DepthLoadOp: LoadOpClear, DepthStoreOp: StoreOpStore},
	}
}

// frame renders with draw (direct commands, or a bundle) and reads the
// target back.
func (s *scene) frame(t *testing.T, draw func(*RenderPassEncoder)) []byte {
	t.Helper()
	enc := must(s.d.CreateCommandEncoder("frame"))(t)
	p := must(enc.BeginRenderPass(s.pass()))(t)
	draw(p)
	if err := p.End(); err != nil {
		t.Fatal(err)
	}
	if err := enc.CopyTextureToBuffer(TexelCopyTextureInfo{Texture: s.target}, TexelCopyBufferInfo{Buffer: s.readback, Layout: TexelCopyBufferLayout{BytesPerRow: 256}},
		Extent3D{Width: 16, Height: 16}); err != nil {
		t.Fatal(err)
	}
	s.d.PushErrorScope(ErrorFilterValidation)
	if err := s.d.Queue().Submit(must(enc.Finish())(t)); err != nil {
		t.Fatal(err)
	}
	if err := s.d.PopErrorScope(ctx(t)); err != nil {
		t.Fatal(err)
	}
	if err := s.readback.MapAsync(ctx(t), MapModeRead, 0, 0); err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 256*16)
	if err := s.readback.ReadMapped(0, out); err != nil {
		t.Fatal(err)
	}
	if err := s.readback.Unmap(); err != nil {
		t.Fatal(err)
	}
	return out
}

func (s *scene) record(r interface {
	SetPipeline(*RenderPipeline)
	SetBindGroup(uint32, *BindGroup, ...uint32)
	SetVertexBuffer(uint32, *Buffer, uint64, uint64)
	SetIndexBuffer(*Buffer, IndexFormat, uint64, uint64)
	DrawIndexed(uint32, uint32, uint32, int32, uint32)
}) {
	r.SetPipeline(s.pipe)
	r.SetBindGroup(0, s.group)
	r.SetVertexBuffer(0, s.vb, 0, 0)
	r.SetIndexBuffer(s.ib, IndexFormatUint16, 0, 0)
	r.DrawIndexed(6, 3, 0, 0, 0)
}

// check compares every pixel with the expected palette color; idx says
// which index each 4x4 texel holds.
func check(t *testing.T, px []byte, idx func(x, y int) int) {
	t.Helper()
	for py := 0; py < 16; py++ {
		for x := 0; x < 16; x++ {
			tx, ty := (x%8)/2, py/4
			want := palette(idx(tx, ty))
			got := px[py*256+4*x : py*256+4*x+4]
			if got[0] != want[0] || got[1] != want[1] || got[2] != want[2] || got[3] != want[3] {
				t.Fatalf("pixel (%d,%d): %v, want %v (texel %d,%d)", x, py, got, want, tx, ty)
			}
		}
	}
}

func TestQuakeShapedFrame(t *testing.T) {
	d := newDev(t, &Quota{MemoryBytes: 1 << 20, Pipelines: 4})
	s := newScene(t, d)
	base := func(x, y int) int { return 1 + x + 4*y }
	check(t, s.frame(t, func(p *RenderPassEncoder) { s.record(p) }), base)

	// A dirty rectangle, as a lightmap update would be: texels (1,1)-(2,2).
	if err := d.Queue().WriteTexture(TexelCopyTextureInfo{Texture: s.idx, Origin: Origin3D{X: 1, Y: 1}}, []byte{200, 200, 200, 200},
		TexelCopyBufferLayout{BytesPerRow: 2}, Extent3D{Width: 2, Height: 2}); err != nil {
		t.Fatal(err)
	}
	dirty := func(x, y int) int {
		if x >= 1 && x <= 2 && y >= 1 && y <= 2 {
			return 200
		}
		return base(x, y)
	}
	check(t, s.frame(t, func(p *RenderPassEncoder) { s.record(p) }), dirty)

	// An out-of-range dirty rectangle is refused at the call.
	var be *BoundsError
	if err := d.Queue().WriteTexture(TexelCopyTextureInfo{Texture: s.idx, Origin: Origin3D{X: 3, Y: 3}}, []byte{1, 2, 3, 4},
		TexelCopyBufferLayout{BytesPerRow: 2}, Extent3D{Width: 2, Height: 2}); !errors.As(err, &be) {
		t.Fatalf("out-of-range writeTexture: %v", err)
	}
	if err := d.Queue().WriteTexture(TexelCopyTextureInfo{Texture: s.idx}, []byte{1, 2, 3},
		TexelCopyBufferLayout{BytesPerRow: 2}, Extent3D{Width: 2, Height: 2}); !errors.As(err, &be) {
		t.Fatalf("short writeTexture data: %v", err)
	}

	// The same frame from a render bundle.
	be2 := must(d.CreateRenderBundleEncoder(RenderBundleEncoderDescriptor{ColorFormats: []TextureFormat{TextureFormatRGBA8Unorm}, DepthStencilFormat: TextureFormatDepth24Plus}))(t)
	s.record(be2)
	bundle := must(be2.Finish("bundle"))(t)
	check(t, s.frame(t, func(p *RenderPassEncoder) { p.ExecuteBundles(bundle) }), dirty)

	if m, p := d.Quota().Usage(); m == 0 || p != 1 {
		t.Fatalf("quota usage %d bytes, %d pipelines", m, p)
	}
	s.pipe.Release()
	if _, p := d.Quota().Usage(); p != 0 {
		t.Fatalf("%d pipelines after release", p)
	}
}

func TestPassRefusesOutOfRange(t *testing.T) {
	d := newDev(t, nil)
	s := newScene(t, d)
	enc := must(d.CreateCommandEncoder(""))(t)
	p := must(enc.BeginRenderPass(s.pass()))(t)
	p.SetVertexBuffer(0, s.vb, 32, 64) // 32+64 > 64
	p.SetViewport(0, 0, 16, 16, 0, 1)
	p.SetScissorRect(0, 0, 16, 16)
	p.SetBlendConstant(Color{1, 1, 1, 1})
	p.SetStencilReference(0)
	var be *BoundsError
	if err := p.End(); !errors.As(err, &be) {
		t.Fatalf("End: %v", err)
	}
	if p.Err() == nil {
		t.Fatal("Err")
	}
}

func TestComputeDispatch(t *testing.T) {
	d := newDev(t, nil)
	const n = 1000
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i], b[i] = float32(i), float32(2*i)
	}
	mk := func(v []float32) *Buffer {
		buf := must(d.CreateBuffer(BufferDescriptor{Size: 4 * n, Usage: BufferUsageStorage | BufferUsageCopyDst | BufferUsageCopySrc}))(t)
		if v != nil {
			if err := d.Queue().WriteBuffer(buf, 0, f32s(v...)); err != nil {
				t.Fatal(err)
			}
		}
		return buf
	}
	ba, bb, bc := mk(a), mk(b), mk(nil)
	mod := must(d.CreateShaderModule(ShaderModuleDescriptor{Code: `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < arrayLength(&c)) { c[id.x] = a[id.x] + b[id.x]; }
}`}))(t)
	pipe := must(d.CreateComputePipeline(ComputePipelineDescriptor{Compute: ProgrammableStage{Module: mod, EntryPoint: "main"}}))(t)
	defer pipe.Release()
	bgl := must(pipe.GetBindGroupLayout(0))(t)
	g := must(d.CreateBindGroup(BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, Buffer: ba}, {Binding: 1, Buffer: bb}, {Binding: 2, Buffer: bc}}}))(t)
	rb := must(d.CreateBuffer(BufferDescriptor{Size: 4 * n, Usage: BufferUsageMapRead | BufferUsageCopyDst}))(t)
	enc := must(d.CreateCommandEncoder(""))(t)
	p := must(enc.BeginComputePass("add"))(t)
	p.SetPipeline(pipe)
	p.SetBindGroup(0, g)
	p.DispatchWorkgroups((n+63)/64, 1, 1)
	if err := p.End(); err != nil {
		t.Fatal(err)
	}
	if err := enc.CopyBufferToBuffer(bc, 0, rb, 0, 4*n); err != nil {
		t.Fatal(err)
	}
	if err := d.Queue().Submit(must(enc.Finish())(t)); err != nil {
		t.Fatal(err)
	}
	if err := d.Queue().OnSubmittedWorkDone(ctx(t)); err != nil {
		t.Fatal(err)
	}
	if err := rb.MapAsync(ctx(t), MapModeRead, 0, 0); err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 4*n)
	if err := rb.ReadMapped(0, out); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if got := math.Float32frombits(binary.LittleEndian.Uint32(out[4*i:])); got != a[i]+b[i] {
			t.Fatalf("c[%d] = %v, want %v", i, got, a[i]+b[i])
		}
	}
}

func TestErrorScopeAndUncaptured(t *testing.T) {
	d := newDev(t, nil)
	d.PushErrorScope(ErrorFilterValidation)
	// MAP_READ may only be combined with COPY_DST: a validation error.
	if _, err := d.CreateBuffer(BufferDescriptor{Size: 16, Usage: BufferUsageMapRead | BufferUsageStorage}); err != nil {
		t.Fatal(err)
	}
	var ge *GPUError
	if err := d.PopErrorScope(ctx(t)); !errors.As(err, &ge) || ge.Kind != "validation" {
		t.Fatalf("PopErrorScope: %v", err)
	}
	got := make(chan error, 1)
	remove := d.OnUncapturedError(func(err error) { got <- err })
	defer remove()
	if _, err := d.CreateBuffer(BufferDescriptor{Size: 16, Usage: BufferUsageMapRead | BufferUsageStorage}); err != nil {
		t.Fatal(err)
	}
	// Errors travel from the GPU process with the next round trip: wait
	// for one.
	if err := d.Queue().OnSubmittedWorkDone(ctx(t)); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-got:
		if !errors.As(err, &ge) {
			t.Fatalf("uncaptured: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("no uncaptured error")
	}
}

func TestDestroyAndCancel(t *testing.T) {
	q := &Quota{}
	d := newDev(t, q)
	if _, err := d.CreateBuffer(BufferDescriptor{Size: 1024, Usage: BufferUsageCopyDst}); err != nil {
		t.Fatal(err)
	}
	c, cancel := context.WithCancel(context.Background())
	cancel()
	if err := d.Queue().OnSubmittedWorkDone(c); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled wait: %v", err)
	}
	d.Destroy()
	select {
	case <-d.Lost():
	case <-time.After(10 * time.Second):
		t.Fatal("not lost after Destroy")
	}
	if d.LostInfo().Reason != "destroyed" {
		t.Fatalf("lost: %+v", d.LostInfo())
	}
	if m, _ := q.Usage(); m != 0 {
		t.Fatalf("usage %d after destroy", m)
	}
	if _, err := d.CreateCommandEncoder(""); !errors.Is(err, ErrDestroyed) {
		t.Fatalf("encoder on a destroyed device: %v", err)
	}
}
