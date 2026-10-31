//go:build js

package webgpu

import (
	"context"
	"encoding/binary"
	"fmt"
	"math/rand/v2"
	"reflect"
	"syscall/js"
	"testing"
	"time"
	"unsafe"
)

func installReplay(t testing.TB) {
	t.Helper()
	if err := InstallReplay(context.Background(), ""); err != nil {
		t.Fatal(err)
	}
}

// fakeGPU builds JavaScript stand-ins for a device, queue, encoder and
// pass whose every method appends its name to *calls.
func fakeGPU(calls *[]string) (device, queue js.Value) {
	obj := func() js.Value { return jsGlobal.Get("Object").New() }
	method := func(o js.Value, name string, ret func() any) {
		o.Set(name, js.FuncOf(func(js.Value, []js.Value) any {
			*calls = append(*calls, name)
			if ret != nil {
				return ret()
			}
			return nil
		}))
	}
	pass := obj()
	for _, n := range []string{"setPipeline", "setBindGroup", "draw", "drawIndexed", "setVertexBuffer", "setIndexBuffer",
		"setViewport", "setScissorRect", "setBlendConstant", "setStencilReference", "executeBundles", "drawIndirect",
		"drawIndexedIndirect", "end", "dispatchWorkgroups", "dispatchWorkgroupsIndirect"} {
		method(pass, n, nil)
	}
	enc := obj()
	method(enc, "beginRenderPass", func() any { return pass })
	method(enc, "beginComputePass", func() any { return pass })
	method(enc, "copyBufferToBuffer", nil)
	method(enc, "copyTextureToTexture", nil)
	method(enc, "finish", func() any { return obj() })
	device = obj()
	method(device, "createCommandEncoder", func() any { return enc })
	queue = obj()
	method(queue, "writeBuffer", nil)
	method(queue, "submit", nil)
	return device, queue
}

// expectedCalls is what replay.js must call for decoded commands.
func expectedCalls(cmds []cmd) []string {
	names := map[opcode]string{
		opSetPipeline: "setPipeline", opSetBindGroup: "setBindGroup", opDraw: "draw", opDrawIndexed: "drawIndexed",
		opSetVertexBuffer: "setVertexBuffer", opSetIndexBuffer: "setIndexBuffer", opSetViewport: "setViewport",
		opSetScissorRect: "setScissorRect", opSetBlendConstant: "setBlendConstant", opSetStencilReference: "setStencilReference",
		opExecuteBundles: "executeBundles", opDrawIndirect: "drawIndirect", opDrawIndexedIndirect: "drawIndexedIndirect",
		opEndPass: "end", opDispatch: "dispatchWorkgroups", opDispatchIndirect: "dispatchWorkgroupsIndirect", opWriteBuffer: "writeBuffer",
	}
	var out []string
	enc := false
	open := func() {
		if !enc {
			out = append(out, "createCommandEncoder")
			enc = true
		}
	}
	for _, c := range cmds {
		switch c.Op {
		case opBeginRenderPass:
			open()
			out = append(out, "beginRenderPass")
		case opBeginComputePass:
			open()
			out = append(out, "beginComputePass")
		case opCopyBufferToBuffer:
			open()
			out = append(out, "copyBufferToBuffer")
		case opCopyTextureToTexture:
			open()
			out = append(out, "copyTextureToTexture")
		case opSubmit:
			if enc {
				out = append(out, "finish", "submit")
			}
			enc = false
		default:
			out = append(out, names[c.Op])
		}
	}
	return out
}

// TestReplayAgreesWithDecode holds replay.js and decode.go to the same
// reading: a recorded stream and thousands of mutations of it are
// accepted or refused by both, and an accepted stream makes replay.js
// call exactly what decode read, in order.
func TestReplayAgreesWithDecode(t *testing.T) {
	installReplay(t)
	r, _ := recordAll(t)
	ids := map[uint32]bool{}
	for _, o := range r.unput {
		ids[o.id] = true
	}
	known := func(id uint32) bool { return ids[id] }
	rp := replayer()
	saved := rp.Get("objs")
	table := jsGlobal.Get("Array").New()
	for id := range ids {
		table.SetIndex(int(id), jsGlobal.Get("Object").New())
	}
	rp.Set("objs", table)
	defer rp.Set("objs", saved)

	var calls []string
	device, queue := fakeGPU(&calls)
	data := jsUint8Array.New(len(r.data))
	js.CopyBytesToJS(data, r.data)
	run := func(w []uint32) (bool, []string) {
		calls = calls[:0]
		cmd := jsUint8Array.New(max(4, 4*len(w)))
		if len(w) > 0 {
			js.CopyBytesToJS(cmd, unsafe.Slice((*byte)(unsafe.Pointer(&w[0])), 4*len(w)))
		}
		err := try("replay", func() { rp.Call("replay", device, queue, cmd, len(w), data, len(r.data)) })
		return err == nil, append([]string(nil), calls...)
	}

	cmds, err := decode(r.words, uint32(len(r.data)), known)
	if err != nil {
		t.Fatal(err)
	}
	ok, got := run(r.words)
	if !ok || !reflect.DeepEqual(got, expectedCalls(cmds)) {
		t.Fatalf("replay ok=%v\n got  %v\n want %v", ok, got, expectedCalls(cmds))
	}

	rng := rand.New(rand.NewPCG(1, 2))
	interesting := []uint32{0, 1, 2, 3, 8, 9, 20, 21, 30, 40, 99, 65537, 0xffffffff}
	agreed, accepted := 0, 0
	for i := 0; i < 4000; i++ {
		w := append([]uint32(nil), r.words...)
		switch rng.IntN(3) {
		case 0:
			w[rng.IntN(len(w))] = interesting[rng.IntN(len(interesting))]
		case 1:
			w[rng.IntN(len(w))] = rng.Uint32()
		case 2:
			w = w[:rng.IntN(len(w))]
		}
		cmds, derr := decode(w, uint32(len(r.data)), known)
		ok, got := run(w)
		if ok != (derr == nil) {
			t.Fatalf("mutation %d: decode err=%v, replay ok=%v; words %v", i, derr, ok, w)
		}
		if ok {
			accepted++
			if want := expectedCalls(cmds); !reflect.DeepEqual(got, want) {
				t.Fatalf("mutation %d: calls differ\n got  %v\n want %v", i, got, want)
			}
		}
		agreed++
	}
	t.Logf("%d mutations, both readers agreed on all; %d accepted", agreed, accepted)
}

func TestQuakeFrameBatched(t *testing.T) {
	installReplay(t)
	d := newDev(t, nil)
	s := newScene(t, d)
	rec := d.NewRecorder()
	// The instance data, rewritten through the recorder before the pass.
	rec.WriteBuffer(s.instances, 0, f32s(0, 0.5, 0, 0, 1, 0.5, 0, 0, 0, 0.9, 0, 0))
	p := rec.BeginRenderPass(s.pass())
	s.record(p)
	if err := p.End(); err != nil {
		t.Fatal(err)
	}
	seq, err := rec.Submit()
	if err != nil {
		t.Fatal(err)
	}
	if err := d.WaitSubmitted(ctx(t), seq); err != nil {
		t.Fatal(err)
	}
	if d.Completed() < seq {
		t.Fatalf("completed %d < %d", d.Completed(), seq)
	}
	check(t, s.readTarget(t), func(x, y int) int { return 1 + x + 4*y })
	// The recorder is reset and reusable.
	if rec.Len() != 0 {
		t.Fatal("not reset")
	}
}

func TestDynamicOffsetsBatched(t *testing.T) {
	installReplay(t)
	d := newDev(t, nil)
	align, _ := d.Limit("minUniformBufferOffsetAlignment")
	u := must(d.CreateBuffer(BufferDescriptor{Size: 2 * align, Usage: BufferUsageUniform | BufferUsageCopyDst}))(t)
	out := must(d.CreateBuffer(BufferDescriptor{Size: 16, Usage: BufferUsageStorage | BufferUsageCopySrc}))(t)
	rb := must(d.CreateBuffer(BufferDescriptor{Size: 16, Usage: BufferUsageMapRead | BufferUsageCopyDst}))(t)
	mod := must(d.CreateShaderModule(ShaderModuleDescriptor{Code: `
struct U { slot: u32, value: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1) fn main() { o[u.slot] = u.value; }`}))(t)
	bgl := must(d.CreateBindGroupLayout(BindGroupLayoutDescriptor{Entries: []BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &BufferBindingLayout{Type: BufferBindingTypeUniform, HasDynamicOffset: true}},
		{Binding: 1, Visibility: ShaderStageCompute, Buffer: &BufferBindingLayout{Type: BufferBindingTypeStorage}},
	}}))(t)
	pl := must(d.CreatePipelineLayout(PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}}))(t)
	pipe := must(d.CreateComputePipeline(ComputePipelineDescriptor{Layout: pl, Compute: ProgrammableStage{Module: mod, EntryPoint: "main"}}))(t)
	g := must(d.CreateBindGroup(BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, Buffer: u, Size: 8}, {Binding: 1, Buffer: out}}}))(t)

	vals := make([]byte, 2*align)
	binary.LittleEndian.PutUint32(vals[0:], 0)
	binary.LittleEndian.PutUint32(vals[4:], 11)
	binary.LittleEndian.PutUint32(vals[align:], 1)
	binary.LittleEndian.PutUint32(vals[align+4:], 22)
	rec := d.NewRecorder()
	rec.WriteBuffer(u, 0, vals)
	c := rec.BeginComputePass()
	c.SetPipeline(pipe)
	c.SetBindGroup(0, g, 0)
	c.DispatchWorkgroups(1, 1, 1)
	c.SetBindGroup(0, g, uint32(align))
	c.DispatchWorkgroups(1, 1, 1)
	if err := c.End(); err != nil {
		t.Fatal(err)
	}
	rec.CopyBufferToBuffer(out, 0, rb, 0, 16)
	if _, err := rec.Submit(); err != nil {
		t.Fatal(err)
	}
	if err := rb.MapAsync(ctx(t), MapModeRead, 0, 0); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, 16)
	if err := rb.ReadMapped(0, got); err != nil {
		t.Fatal(err)
	}
	if a, b := binary.LittleEndian.Uint32(got), binary.LittleEndian.Uint32(got[4:]); a != 11 || b != 22 {
		t.Fatalf("dynamic offsets: o = [%d %d], want [11 22]", a, b)
	}
}

func TestSubmitErrors(t *testing.T) {
	installReplay(t)
	d := newDev(t, nil)
	rec := d.NewRecorder()
	rec.BeginComputePass()
	if _, err := rec.Submit(); err == nil {
		t.Fatal("open pass submitted")
	}
	if err := d.WaitSubmitted(ctx(t), d.Submitted()+1); err == nil {
		t.Fatal("waited for a submission never made")
	}
	d.Destroy()
	<-d.Lost()
	if _, err := d.NewRecorder().Submit(); err == nil {
		t.Fatal("submitted on a destroyed device")
	}
}

// Benchmarks: CPU time per frame of n draws (setBindGroup + draw each),
// direct calls against batched recording, and the same frame written in
// JavaScript as the floor.
func benchFrames(b *testing.B, mode string, n int) {
	installReplay(b)
	g, err := Navigator()
	if err != nil {
		b.Fatal(err)
	}
	a, err := g.RequestAdapter(context.Background(), &RequestAdapterOptions{AllowFallback: true})
	if err != nil {
		b.Fatal(err)
	}
	d, err := a.RequestDevice(context.Background(), nil)
	if err != nil {
		b.Fatal(err)
	}
	defer d.Destroy()
	mod, _ := d.CreateShaderModule(ShaderModuleDescriptor{Code: `
@group(0) @binding(0) var<uniform> u: vec4f;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array(vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5));
  return vec4f(p[i] * 0.01 + u.xy, 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }`})
	pipe, err := d.CreateRenderPipeline(RenderPipelineDescriptor{Vertex: VertexState{Module: mod, EntryPoint: "vs"},
		Fragment: &FragmentState{Module: mod, EntryPoint: "fs", Targets: []ColorTargetState{{Format: TextureFormatRGBA8Unorm}}}})
	if err != nil {
		b.Fatal(err)
	}
	tex, _ := d.CreateTexture(TextureDescriptor{Size: Extent3D{Width: 64, Height: 64}, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageRenderAttachment})
	view, _ := tex.CreateView(nil)
	bgl, _ := pipe.GetBindGroupLayout(0)
	groups := make([]*BindGroup, 16)
	jsGroups := make([]any, 16)
	for i := range groups {
		ub, _ := d.CreateBuffer(BufferDescriptor{Size: 256, Usage: BufferUsageUniform})
		groups[i], _ = d.CreateBindGroup(BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, Buffer: ub}}})
		jsGroups[i] = groups[i].v
	}
	desc := RenderPassDescriptor{ColorAttachments: []*RenderPassColorAttachment{{View: view, LoadOp: LoadOpClear, StoreOp: StoreOpStore}}}
	dm, err := desc.toJS()
	if err != nil {
		b.Fatal(err)
	}
	jsDesc := js.ValueOf(dm)
	rec := d.NewRecorder()
	frame := func() {
		switch mode {
		case "direct":
			enc, _ := d.CreateCommandEncoder("")
			p, _ := enc.BeginRenderPass(desc)
			p.SetPipeline(pipe)
			for k := 0; k < n; k++ {
				p.SetBindGroup(0, groups[k&15])
				p.Draw(3, 1, 0, 0)
			}
			p.End()
			cb, _ := enc.Finish()
			d.Queue().Submit(cb)
		case "batched":
			p := rec.BeginRenderPass(desc)
			p.SetPipeline(pipe)
			for k := 0; k < n; k++ {
				p.SetBindGroup(0, groups[k&15])
				p.Draw(3, 1, 0, 0)
			}
			p.End()
			if _, err := rec.Submit(); err != nil {
				b.Fatal(err)
			}
		case "js":
			replayer().Call("benchFrame", d.v, d.queue.v, jsDesc, pipe.v, js.ValueOf(jsGroups), n)
		}
	}
	if mode == "js" && replayer().Get("benchFrame").IsUndefined() {
		replayer().Set("benchFrame", jsGlobal.Get("Function").New("device", "queue", "desc", "pipe", "groups", "n", `
const enc = device.createCommandEncoder(); const pass = enc.beginRenderPass(desc); pass.setPipeline(pipe);
for (let k = 0; k < n; k++) { pass.setBindGroup(0, groups[k & 15]); pass.draw(3, 1, 0, 0); }
pass.end(); queue.submit([enc.finish()]);`))
	}
	frame() // warm: ids registered, arrays grown
	d.Queue().OnSubmittedWorkDone(context.Background())
	var cpu time.Duration
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := time.Now()
		frame()
		cpu += time.Since(t)
		b.StopTimer()
		d.Queue().OnSubmittedWorkDone(context.Background()) // keep the GPU queue from backing up; not timed
		b.StartTimer()
	}
	b.ReportMetric(float64(cpu.Microseconds())/float64(b.N), "cpu-µs/frame")
}

func BenchmarkFrame(b *testing.B) {
	for _, n := range []int{10, 100, 1000, 5000} {
		for _, mode := range []string{"direct", "batched", "js"} {
			b.Run(fmt.Sprintf("%s/draws=%d", mode, n), func(b *testing.B) { benchFrames(b, mode, n) })
		}
	}
}

// BenchmarkWriteBuffer64KiBSource measures queue.writeBuffer of 64 KiB
// from a view into ArrayBuffers of growing size: the open question of why
// writing from a view of the Go memory (tens of MiB) was slower than from
// a 64 KiB staging array.
func BenchmarkWriteBuffer64KiBSource(b *testing.B) {
	g, _ := Navigator()
	a, err := g.RequestAdapter(context.Background(), &RequestAdapterOptions{AllowFallback: true})
	if err != nil {
		b.Fatal(err)
	}
	d, err := a.RequestDevice(context.Background(), nil)
	if err != nil {
		b.Fatal(err)
	}
	defer d.Destroy()
	dst, _ := d.CreateBuffer(BufferDescriptor{Size: 65536, Usage: BufferUsageCopyDst})
	type source struct {
		name string
		src  func() js.Value
	}
	var sources []source
	for _, size := range []int{64 << 10, 1 << 20, 16 << 20, 64 << 20} {
		u := jsUint8Array.New(size)
		sources = append(sources, source{fmt.Sprintf("arraybuffer=%dKiB", size>>10), func() js.Value { return u }})
	}
	// The Go instance's own memory (a WebAssembly.Memory buffer), as the
	// prototype's zero-copy path used it; reachable when the runner
	// exposes the instance.
	if g := jsGlobal.Get("goRunner"); !nullish(g) {
		sources = append(sources, source{"wasm-memory", func() js.Value {
			return jsUint8Array.New(g.Get("_inst").Get("exports").Get("mem").Get("buffer"))
		}})
	}
	data := make([]byte, 65536)
	for _, s := range sources {
		b.Run(s.name, func(b *testing.B) {
			src := s.src()
			off := 0
			if s.name == "wasm-memory" {
				off = int(uintptr(unsafe.Pointer(&data[0])))
			}
			d.queue.v.Call("writeBuffer", dst.v, 0, src, off, 65536) // warm
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				d.queue.v.Call("writeBuffer", dst.v, 0, src, off, 65536)
			}
			b.StopTimer()
			d.Queue().OnSubmittedWorkDone(context.Background())
		})
	}
}
