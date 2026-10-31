//go:build !js

package webgpu

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/go-gfx/webgpu/wire"
)

// sink is a RemoteSink that keeps what it receives and can fail.
type sink struct {
	objs      map[uint32]created
	destroyed []uint32
	submits   [][]uint32
	fail      error
}

type created struct {
	kind wire.Kind
	desc []byte
}

func newSink() *sink { return &sink{objs: map[uint32]created{}} }

func (s *sink) Create(id uint32, kind wire.Kind, desc []byte) error {
	if s.fail != nil {
		return s.fail
	}
	s.objs[id] = created{kind, append([]byte(nil), desc...)}
	return nil
}

func (s *sink) Destroy(id uint32) error { s.destroyed = append(s.destroyed, id); return nil }

func (s *sink) Submit(words []uint32, _ []byte) error {
	if s.fail != nil {
		return s.fail
	}
	s.submits = append(s.submits, append([]uint32(nil), words...))
	return nil
}

var errSink = errors.New("sink down")

// TestRemoteDevice builds a compute setup on a remote device: every
// creation reaches the sink as a descriptor that decodes, on the
// receiving side, to the descriptor the caller passed.
func TestRemoteDevice(t *testing.T) {
	s := newSink()
	d := NewRemoteDevice(s, "guest", &Quota{MemoryBytes: 1 << 20, Pipelines: 4, ShaderBytes: 1 << 10})
	must := func(o any, err error) any {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return o
	}
	buf := must(d.CreateBuffer(BufferDescriptor{Size: 64, Usage: BufferUsageStorage | BufferUsageCopyDst})).(*Buffer)
	tex := must(d.CreateTexture(TextureDescriptor{Size: Extent3D{4, 4, 1}, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageTextureBinding})).(*Texture)
	view := must(tex.CreateView(&TextureViewDescriptor{Label: "v"})).(*TextureView)
	smp := must(d.CreateSampler(nil)).(*Sampler)
	smp2 := must(d.CreateSampler(&SamplerDescriptor{Label: "s"})).(*Sampler)
	mod := must(d.CreateShaderModule(ShaderModuleDescriptor{Code: "@compute @workgroup_size(1) fn main() {}"})).(*ShaderModule)
	bgl := must(d.CreateBindGroupLayout(BindGroupLayoutDescriptor{Entries: []BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &BufferBindingLayout{Type: BufferBindingTypeStorage}}}})).(*BindGroupLayout)
	pl := must(d.CreatePipelineLayout(PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}})).(*PipelineLayout)
	bg := must(d.CreateBindGroup(BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, Buffer: buf}}})).(*BindGroup)
	cp := must(d.CreateComputePipeline(ComputePipelineDescriptor{Layout: pl, Compute: ProgrammableStage{Module: mod, EntryPoint: "main"}})).(*ComputePipeline)
	rp := must(d.CreateRenderPipeline(RenderPipelineDescriptor{Vertex: VertexState{Module: mod}})).(*RenderPipeline)
	if len(s.objs) != 11 {
		t.Fatalf("%d objects reached the sink", len(s.objs))
	}
	// The receiving side decodes every descriptor against the same ids.
	byID := map[uint32]any{}
	for _, h := range []any{buf, tex, view, smp, smp2, mod, bgl, pl, bg, cp, rp} {
		id, err := ReplayID(h)
		if err != nil {
			t.Fatal(err)
		}
		byID[id] = h
	}
	resolve := func(id uint32) (any, error) {
		if o, ok := byID[id]; ok {
			return o, nil
		}
		return nil, errors.New("unknown")
	}
	id, _ := ReplayID(bg)
	got, err := DecodeDescriptor(s.objs[id].kind, s.objs[id].desc, resolve)
	if err != nil || !reflect.DeepEqual(got, BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, Buffer: buf}}}) {
		t.Fatalf("bind group on the receiving side: %+v %v", got, err)
	}

	// Work: a write and a recorded compute submit.
	if err := d.Queue().WriteBuffer(buf, 0, []byte{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if err := d.Queue().WriteBuffer(buf, 0, nil); err != nil || len(s.submits) != 1 {
		t.Fatal("an empty write was sent")
	}
	r := d.NewRecorder()
	c := r.BeginComputePass()
	c.SetPipeline(cp)
	c.SetBindGroup(0, bg)
	c.DispatchWorkgroups(1, 1, 1)
	if err := c.End(); err != nil {
		t.Fatal(err)
	}
	seq, err := r.Submit()
	if err != nil || seq != 2 || len(s.submits) != 2 {
		t.Fatalf("submit %d %v", seq, err)
	}
	if _, err := wire.Decode(s.submits[1], 0, func(id uint32) bool { _, ok := byID[id]; return ok }); err != nil {
		t.Fatalf("the submitted stream names objects the sink does not have: %v", err)
	}
	if err := d.WaitSubmitted(context.Background(), seq); err != nil {
		t.Fatal(err)
	}
	if err := d.WaitSubmitted(context.Background(), 9); err == nil {
		t.Error("waited for a submission never made")
	}

	// Releases reach the sink once each and refund the quota.
	buf.Destroy()
	buf.Destroy()
	tex.Destroy()
	cp.Release()
	rp.Release()
	if len(s.destroyed) != 4 {
		t.Errorf("%d destroys", len(s.destroyed))
	}
	if m, p := d.Quota().Usage(); m != 0 || p != 0 {
		t.Errorf("quota after release: %d bytes, %d pipelines", m, p)
	}
	if err := buf.MapAsync(context.Background(), MapModeRead, 0, 0); !errors.Is(err, ErrRemoteUnsupported) {
		t.Error("MapAsync")
	}
	if err := d.Queue().WriteTexture(TexelCopyTextureInfo{}, nil, TexelCopyBufferLayout{}, Extent3D{}); !errors.Is(err, ErrRemoteUnsupported) {
		t.Error("WriteTexture")
	}
	if err := d.Queue().WriteBuffer(buf, 0, []byte{1, 2, 3, 4}); !errors.Is(err, ErrDestroyed) {
		t.Errorf("write to a destroyed buffer: %v", err)
	}
	if err := d.Queue().WriteBuffer(buf, 1, []byte{1}); err == nil {
		t.Error("unaligned write")
	}
}

func TestRemoteRefusals(t *testing.T) {
	s := newSink()
	d := NewRemoteDevice(s, "guest", &Quota{MemoryBytes: 1024, Pipelines: 1, ShaderBytes: 8})
	mod, _ := d.CreateShaderModule(ShaderModuleDescriptor{Code: "x"})
	bgl, _ := d.CreateBindGroupLayout(BindGroupLayoutDescriptor{})
	checks := map[string]func() error{
		"buffer over 2^53": func() error { _, err := d.CreateBuffer(BufferDescriptor{Size: 1<<53 + 1}); return err },
		"buffer quota":     func() error { _, err := d.CreateBuffer(BufferDescriptor{Size: 2048}); return err },
		"texture format": func() error {
			_, err := d.CreateTexture(TextureDescriptor{Size: Extent3D{1, 1, 1}, Format: "nope"})
			return err
		},
		"texture quota": func() error {
			_, err := d.CreateTexture(TextureDescriptor{Size: Extent3D{64, 64, 1}, Format: TextureFormatRGBA8Unorm})
			return err
		},
		"shader quota": func() error { _, err := d.CreateShaderModule(ShaderModuleDescriptor{Code: "123456789"}); return err },
		"bgl check": func() error {
			_, err := d.CreateBindGroupLayout(BindGroupLayoutDescriptor{Entries: []BindGroupLayoutEntry{{}}})
			return err
		},
		"pl check": func() error {
			_, err := d.CreatePipelineLayout(PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{nil}})
			return err
		},
		"bg check": func() error { _, err := d.CreateBindGroup(BindGroupDescriptor{}); return err },
		"rp check": func() error { _, err := d.CreateRenderPipeline(RenderPipelineDescriptor{}); return err },
		"cp check": func() error { _, err := d.CreateComputePipeline(ComputePipelineDescriptor{}); return err },
		"cp NaN (encode)": func() error {
			_, err := d.CreateComputePipeline(ComputePipelineDescriptor{Compute: ProgrammableStage{Module: mod, Constants: map[string]float64{"k": math.NaN()}}})
			return err
		},
		"view of an unattached texture": func() error { _, err := (&Texture{}).CreateView(nil); return err },
	}
	for name, f := range checks {
		if f() == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	// One pipeline fits the quota; the second is refused.
	if _, err := d.CreateComputePipeline(ComputePipelineDescriptor{Compute: ProgrammableStage{Module: mod}}); err != nil {
		t.Fatal(err)
	}
	if _, err := d.CreateRenderPipeline(RenderPipelineDescriptor{Vertex: VertexState{Module: mod}}); err == nil {
		t.Error("render pipeline over quota")
	}
	if _, err := d.CreateComputePipeline(ComputePipelineDescriptor{Compute: ProgrammableStage{Module: mod}}); err == nil {
		t.Error("compute pipeline over quota")
	}

	// A sink that refuses: every creation fails and refunds.
	s2 := newSink()
	d2 := NewRemoteDevice(s2, "guest", &Quota{MemoryBytes: 1 << 20, Pipelines: 4})
	tex, _ := d2.CreateTexture(TextureDescriptor{Size: Extent3D{1, 1, 1}, Format: TextureFormatRGBA8Unorm})
	mod2, _ := d2.CreateShaderModule(ShaderModuleDescriptor{Code: "x"})
	s2.fail = errSink
	for name, f := range map[string]func() error{
		"buffer": func() error { _, err := d2.CreateBuffer(BufferDescriptor{Size: 4}); return err },
		"texture": func() error {
			_, err := d2.CreateTexture(TextureDescriptor{Size: Extent3D{1, 1, 1}, Format: TextureFormatRGBA8Unorm})
			return err
		},
		"view":    func() error { _, err := tex.CreateView(nil); return err },
		"sampler": func() error { _, err := d2.CreateSampler(nil); return err },
		"shader":  func() error { _, err := d2.CreateShaderModule(ShaderModuleDescriptor{}); return err },
		"bgl":     func() error { _, err := d2.CreateBindGroupLayout(BindGroupLayoutDescriptor{}); return err },
		"pl":      func() error { _, err := d2.CreatePipelineLayout(PipelineLayoutDescriptor{}); return err },
		"bg":      func() error { _, err := d2.CreateBindGroup(BindGroupDescriptor{Layout: bgl}); return err },
		"render": func() error {
			_, err := d2.CreateRenderPipeline(RenderPipelineDescriptor{Vertex: VertexState{Module: mod2}})
			return err
		},
		"compute": func() error {
			_, err := d2.CreateComputePipeline(ComputePipelineDescriptor{Compute: ProgrammableStage{Module: mod2}})
			return err
		},
	} {
		if err := f(); !errors.Is(err, errSink) {
			t.Errorf("%s: %v", name, err)
		}
	}
	if m, p := d2.Quota().Usage(); m != 4 || p != 0 {
		t.Errorf("refused creations were not refunded: %d bytes, %d pipelines", m, p)
	}

	// Submits: an open pass, then a sink failure, which loses the device.
	r := d2.NewRecorder()
	r.BeginComputePass()
	if _, err := r.Submit(); !errors.Is(err, ErrRecorderState) {
		t.Errorf("open pass: %v", err)
	}
	r.BeginComputePass().End()
	if _, err := r.Submit(); !errors.Is(err, errSink) {
		t.Errorf("sink failure: %v", err)
	}
	if !d2.isLost() || d2.LostInfo().Reason != "remote" {
		t.Error("a failed submit did not lose the device")
	}
	if _, err := r.Submit(); !errors.Is(err, ErrDestroyed) {
		t.Error("submit on a lost device")
	}
	if _, err := d2.CreateSampler(nil); !errors.Is(err, ErrDestroyed) {
		t.Error("create on a lost device")
	}

	// A device that is not remote.
	plain := newDevice(jsValue{}, "plain", nil)
	if _, err := plain.CreateSampler(nil); !errors.Is(err, ErrNotRemote) {
		t.Error("create on a plain device")
	}
	if _, err := plain.NewRecorder().Submit(); !errors.Is(err, ErrNotRemote) {
		t.Error("submit on a plain device")
	}
	b := &Buffer{size: 4}
	b.dev = plain
	b.Destroy() // no sink: only the refund
}
