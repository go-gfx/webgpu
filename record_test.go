package webgpu

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

func all(uint32) bool { return true }

// recordAll records one of every command, and returns the recorder.
func recordAll(t *testing.T) (*Recorder, []opcode) {
	t.Helper()
	d := newDevice(jsValue{}, "d", nil)
	r := d.NewRecorder()
	vb := buf(64, BufferUsageVertex|BufferUsageCopyDst|BufferUsageCopySrc)
	ib := buf(64, BufferUsageIndex)
	ind := buf(64, BufferUsageIndirect)
	g := &BindGroup{}
	rp := &RenderPipeline{}
	cp := &ComputePipeline{}
	v, rv, dv := &TextureView{}, &TextureView{}, &TextureView{}
	bundle := &RenderBundle{}

	r.WriteBuffer(vb, 4, []byte{1, 2, 3, 4, 5, 6, 7, 8})
	r.CopyBufferToBuffer(vb, 0, vb, 32, 16)
	p := r.BeginRenderPass(RenderPassDescriptor{
		ColorAttachments: []*RenderPassColorAttachment{
			{View: v, ResolveTarget: rv, LoadOp: LoadOpClear, StoreOp: StoreOpStore, ClearValue: Color{0.5, 0, 0, 1}},
			nil,
		},
		DepthStencilAttachment: &RenderPassDepthStencilAttachment{View: dv, DepthLoadOp: LoadOpClear, DepthStoreOp: StoreOpDiscard, DepthClearValue: 1,
			StencilLoadOp: LoadOpLoad, StencilStoreOp: StoreOpStore, StencilClearValue: 7, DepthReadOnly: false, StencilReadOnly: false},
	})
	p.SetPipeline(rp)
	p.SetBindGroup(0, g)
	p.SetBindGroup(1, g, 256, 512)
	p.SetBindGroup(2, nil)
	p.SetVertexBuffer(0, vb, 0, 0)
	p.SetIndexBuffer(ib, IndexFormatUint32, 8, 16)
	p.SetViewport(0, 0, 640, 480, 0, 1)
	p.SetScissorRect(1, 2, 3, 4)
	p.SetBlendConstant(Color{1, 1, 1, 1})
	p.SetStencilReference(9)
	p.Draw(3, 1, 0, 0)
	p.DrawIndexed(6, 2, 0, -4, 1)
	p.DrawIndirect(ind, 16)
	p.DrawIndexedIndirect(ind, 4)
	p.ExecuteBundles(bundle, bundle)
	if err := p.End(); err != nil {
		t.Fatal(err)
	}
	c := r.BeginComputePass()
	c.SetPipeline(cp)
	c.SetBindGroup(0, g, 64)
	c.DispatchWorkgroups(4, 5, 6)
	c.DispatchWorkgroupsIndirect(ind, 12)
	if err := c.End(); err != nil {
		t.Fatal(err)
	}
	if err := r.finish(); err != nil {
		t.Fatal(err)
	}
	return r, []opcode{opWriteBuffer, opCopyBufferToBuffer, opBeginRenderPass, opSetPipeline, opSetBindGroup, opSetBindGroup, opSetBindGroup,
		opSetVertexBuffer, opSetIndexBuffer, opSetViewport, opSetScissorRect, opSetBlendConstant, opSetStencilReference, opDraw, opDrawIndexed,
		opDrawIndirect, opDrawIndexedIndirect, opExecuteBundles, opEndPass, opBeginComputePass, opSetPipeline, opSetBindGroup, opDispatch,
		opDispatchIndirect, opEndPass, opSubmit}
}

func TestRecordDecodeRoundTrip(t *testing.T) {
	r, want := recordAll(t)
	if r.Err() != nil {
		t.Fatal(r.Err())
	}
	cmds, err := decode(r.words, uint32(len(r.data)), all)
	if err != nil {
		t.Fatal(err)
	}
	var got []opcode
	for _, c := range cmds {
		got = append(got, c.op)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ops %v, want %v", got, want)
	}
	// Spot checks of arguments.
	if a := cmds[0].args; a[1] != 4 || a[2] != 0 || a[3] != 0 || a[4] != 8 {
		t.Fatalf("writeBuffer args %v", a)
	}
	if string(r.data) != "\x01\x02\x03\x04\x05\x06\x07\x08" {
		t.Fatalf("data %v", r.data)
	}
	if a := cmds[5].args; a[0] != 1 || a[2] != 2 || a[3] != 256 || a[4] != 512 {
		t.Fatalf("dynamic offsets %v", a)
	}
	if a := cmds[6].args; a[1] != 0 {
		t.Fatalf("unbind %v", a)
	}
	if a := cmds[7].args; a[4] != 64 { // size 0 resolved to the end
		t.Fatalf("vertex buffer size %v", a)
	}
	if a := cmds[14].args; int32(a[3]) != -4 {
		t.Fatalf("baseVertex %v", a)
	}
	if a := cmds[9].args; math.Float32frombits(a[2]) != 640 {
		t.Fatalf("viewport %v", a)
	}
	if a := cmds[2].args; a[0] != 2 || a[3] != 2 || a[4] != 1 || math.Float32frombits(a[5]) != 0.5 || a[9] != 0 || a[17] != 1 {
		t.Fatalf("render pass %v", a)
	}
	if r.Len() != len(r.words) {
		t.Fatal("Len")
	}
	r.Reset()
	if r.Len() != 0 || len(r.data) != 0 || len(r.refs) != 0 || len(r.unput) != 0 {
		t.Fatal("Reset")
	}
}

func TestRecorderRefusals(t *testing.T) {
	d := newDevice(jsValue{}, "d", nil)
	vb := buf(64, BufferUsageVertex|BufferUsageCopyDst)
	noUsage := buf(64, 0)
	var be *BoundsError
	for name, f := range map[string]func(r *Recorder){
		"write out of range": func(r *Recorder) { r.WriteBuffer(vb, 64, []byte{1, 2, 3, 4}) },
		"copy unaligned":     func(r *Recorder) { r.CopyBufferToBuffer(vb, 1, vb, 0, 4) },
		"9 colors": func(r *Recorder) {
			r.BeginRenderPass(RenderPassDescriptor{ColorAttachments: make([]*RenderPassColorAttachment, 9)})
		},
		"no view": func(r *Recorder) {
			r.BeginRenderPass(RenderPassDescriptor{ColorAttachments: []*RenderPassColorAttachment{{}}})
		},
		"nil pipeline": func(r *Recorder) { r.BeginRenderPass(RenderPassDescriptor{}).SetPipeline(nil) },
		"vertex range": func(r *Recorder) { r.BeginRenderPass(RenderPassDescriptor{}).SetVertexBuffer(0, vb, 32, 64) },
		"index usage": func(r *Recorder) {
			r.BeginRenderPass(RenderPassDescriptor{}).SetIndexBuffer(vb, IndexFormatUint16, 0, 0)
		},
		"index format": func(r *Recorder) {
			r.BeginRenderPass(RenderPassDescriptor{}).SetIndexBuffer(buf(8, BufferUsageIndex), "uint8", 0, 0)
		},
		"indirect usage":   func(r *Recorder) { r.BeginRenderPass(RenderPassDescriptor{}).DrawIndirect(noUsage, 0) },
		"nil bundle":       func(r *Recorder) { r.BeginRenderPass(RenderPassDescriptor{}).ExecuteBundles(nil) },
		"compute nil":      func(r *Recorder) { r.BeginComputePass().SetPipeline(nil) },
		"compute indirect": func(r *Recorder) { r.BeginComputePass().DispatchWorkgroupsIndirect(buf(8, BufferUsageIndirect), 0) },
	} {
		r := d.NewRecorder()
		f(r)
		if r.Err() == nil {
			t.Errorf("%s: accepted", name)
			continue
		}
		if !errors.As(r.Err(), &be) && !strings.Contains(r.Err().Error(), "no view") {
			t.Errorf("%s: %v", name, r.Err())
		}
	}
	// A refused command is not recorded; the stream stays decodable.
	r := d.NewRecorder()
	p := r.BeginRenderPass(RenderPassDescriptor{})
	p.SetVertexBuffer(0, vb, 32, 64)
	p.Draw(3, 1, 0, 0)
	if err := p.End(); err == nil {
		t.Fatal("End hid the refusal")
	}
	if err := r.finish(); err != nil {
		t.Fatal(err)
	}
	if _, err := decode(r.words, 0, all); err != nil {
		t.Fatal(err)
	}
}

func TestRecorderState(t *testing.T) {
	d := newDevice(jsValue{}, "d", nil)
	vb := buf(64, BufferUsageVertex|BufferUsageCopyDst|BufferUsageCopySrc)
	state := func(name string, f func(r *Recorder)) {
		t.Helper()
		r := d.NewRecorder()
		f(r)
		if !errors.Is(r.Err(), ErrRecorderState) {
			t.Errorf("%s: %v", name, r.Err())
		}
	}
	state("write in pass", func(r *Recorder) { r.BeginRenderPass(RenderPassDescriptor{}); r.WriteBuffer(vb, 0, []byte{1, 2, 3, 4}) })
	state("copy in pass", func(r *Recorder) { r.BeginComputePass(); r.CopyBufferToBuffer(vb, 0, vb, 4, 4) })
	state("nested render", func(r *Recorder) {
		r.BeginRenderPass(RenderPassDescriptor{})
		r.BeginRenderPass(RenderPassDescriptor{})
	})
	state("nested compute", func(r *Recorder) { r.BeginRenderPass(RenderPassDescriptor{}); r.BeginComputePass() })
	state("open at submit", func(r *Recorder) { r.BeginComputePass(); _ = r.finish() })
	ended := func(r *Recorder) *RenderPassRecorder {
		p := r.BeginRenderPass(RenderPassDescriptor{})
		p.End()
		return p
	}
	for name, f := range map[string]func(p *RenderPassRecorder){
		"setPipeline":         func(p *RenderPassRecorder) { p.SetPipeline(&RenderPipeline{}) },
		"setBindGroup":        func(p *RenderPassRecorder) { p.SetBindGroup(0, nil) },
		"setVertexBuffer":     func(p *RenderPassRecorder) { p.SetVertexBuffer(0, vb, 0, 0) },
		"setIndexBuffer":      func(p *RenderPassRecorder) { p.SetIndexBuffer(vb, IndexFormatUint16, 0, 0) },
		"draw":                func(p *RenderPassRecorder) { p.Draw(1, 1, 0, 0) },
		"drawIndexed":         func(p *RenderPassRecorder) { p.DrawIndexed(1, 1, 0, 0, 0) },
		"drawIndirect":        func(p *RenderPassRecorder) { p.DrawIndirect(vb, 0) },
		"setViewport":         func(p *RenderPassRecorder) { p.SetViewport(0, 0, 1, 1, 0, 1) },
		"setScissorRect":      func(p *RenderPassRecorder) { p.SetScissorRect(0, 0, 1, 1) },
		"setBlendConstant":    func(p *RenderPassRecorder) { p.SetBlendConstant(Color{}) },
		"setStencilReference": func(p *RenderPassRecorder) { p.SetStencilReference(0) },
		"executeBundles":      func(p *RenderPassRecorder) { p.ExecuteBundles() },
		"end":                 func(p *RenderPassRecorder) { p.End() },
	} {
		state("render "+name+" after end", func(r *Recorder) { f(ended(r)) })
	}
	cended := func(r *Recorder) *ComputePassRecorder {
		p := r.BeginComputePass()
		p.End()
		return p
	}
	for name, f := range map[string]func(p *ComputePassRecorder){
		"setPipeline":      func(p *ComputePassRecorder) { p.SetPipeline(&ComputePipeline{}) },
		"setBindGroup":     func(p *ComputePassRecorder) { p.SetBindGroup(0, nil) },
		"dispatch":         func(p *ComputePassRecorder) { p.DispatchWorkgroups(1, 1, 1) },
		"dispatchIndirect": func(p *ComputePassRecorder) { p.DispatchWorkgroupsIndirect(vb, 0) },
		"end":              func(p *ComputePassRecorder) { p.End() },
	} {
		state("compute "+name+" after end", func(r *Recorder) { f(cended(r)) })
	}
	// An empty write records nothing; a huge one is refused.
	r := d.NewRecorder()
	r.WriteBuffer(vb, 0, nil)
	if r.Len() != 0 || r.Err() != nil {
		t.Fatal("empty write")
	}
	if !dataFits(1<<30, 1<<30) || maxData != math.MaxUint32 {
		t.Fatal("dataFits")
	}
	saved := maxData
	maxData = 8
	defer func() { maxData = saved }()
	r.WriteBuffer(vb, 0, make([]byte, 8))
	r.WriteBuffer(vb, 8, make([]byte, 4))
	if r.Err() == nil || !strings.Contains(r.Err().Error(), "4 GiB") {
		t.Fatalf("data limit: %v", r.Err())
	}
}

func TestRegistry(t *testing.T) {
	var dropped []uint32
	registry.mu.Lock()
	registry.drop = func(id uint32) { dropped = append(dropped, id) }
	registry.mu.Unlock()
	defer func() {
		registry.mu.Lock()
		registry.drop = nil
		registry.mu.Unlock()
	}()
	o := &object{}
	id, unput := o.batchID()
	if id == 0 || !unput {
		t.Fatal(id, unput)
	}
	if id2, unput2 := o.batchID(); id2 != id || !unput2 {
		t.Fatal("second batchID")
	}
	if !o.markPut() || o.markPut() {
		t.Fatal("markPut must succeed once")
	}
	if _, unput3 := o.batchID(); unput3 {
		t.Fatal("put object reported unput")
	}
	freeID(id)
	if len(dropped) != 1 || dropped[0] != id {
		t.Fatal(dropped)
	}
	p := &object{}
	if got, _ := p.batchID(); got != id {
		t.Fatalf("freed id %d not reused (got %d)", id, got)
	}
	registry.mu.Lock()
	registry.drop = nil
	registry.mu.Unlock()
	freeID(999999) // no drop function: only recycles
}

func TestDecodeRejects(t *testing.T) {
	known := func(id uint32) bool { return id >= 1 && id <= 3 }
	rp := []uint32{20, 0, 0} // empty render pass
	for _, c := range []struct {
		name  string
		w     []uint32
		ndata uint32
		err   string
	}{
		{"bad opcode", []uint32{99}, 0, "bad opcode"},
		{"draw outside pass", []uint32{3, 1, 1, 0, 0}, 0, "out of place"},
		{"pipeline outside pass", []uint32{1, 1}, 0, "out of place"},
		{"dispatch in render", append(append([]uint32{}, rp...), 23, 1, 1, 1), 0, "out of place"},
		{"write in pass", append(append([]uint32{}, rp...), 30, 1, 0, 0, 0, 4), 4, "out of place"},
		{"truncated draw", append(append([]uint32{}, rp...), 3, 1), 0, "truncated"},
		{"truncated bind group", append(append([]uint32{}, rp...), 2, 0), 0, "truncated"},
		{"too many offsets", append(append([]uint32{}, rp...), 2, 0, 1, 70000), 0, "dynamic offsets"},
		{"offsets past end", append(append([]uint32{}, rp...), 2, 0, 1, 5, 1), 0, "truncated"},
		{"truncated bundles", append(append([]uint32{}, rp...), 11), 0, "truncated"},
		{"too many bundles", append(append([]uint32{}, rp...), 11, 70000), 0, "bundles"},
		{"truncated pass", []uint32{20}, 0, "truncated"},
		{"9 colors", []uint32{20, 9}, 0, "color attachments"},
		{"truncated colors", []uint32{20, 1, 1}, 0, "truncated"},
		{"bad depth flag", []uint32{20, 0, 2}, 0, "depth flag"},
		{"truncated depth", []uint32{20, 0, 1, 1}, 0, "truncated"},
		{"unknown view", []uint32{20, 1, 9, 0, 1, 1, 0, 0, 0, 0, 0}, 0, "unknown object 9"},
		{"unknown resolve", []uint32{20, 1, 1, 9, 1, 1, 0, 0, 0, 0, 0}, 0, "unknown object 9"},
		{"dirty unused slot", []uint32{20, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0}, 0, "not empty"},
		{"bad load op", []uint32{20, 1, 1, 0, 3, 1, 0, 0, 0, 0, 0}, 0, "load or store"},
		{"unknown depth view", []uint32{20, 0, 1, 9, 1, 1, 0, 0, 1, 1, 0, 0}, 0, "unknown object 9"},
		{"bad depth op", []uint32{20, 0, 1, 1, 3, 1, 0, 0, 1, 1, 0, 0}, 0, "depth-stencil ops"},
		{"bad read-only", []uint32{20, 0, 1, 1, 1, 1, 0, 2, 1, 1, 0, 0}, 0, "depth-stencil ops"},
		{"pipeline id 0", append(append([]uint32{}, rp...), 1, 0), 0, "unknown object 0"},
		{"bind group unknown", append(append([]uint32{}, rp...), 2, 0, 9, 0), 0, "unknown object 9"},
		{"vertex unknown", append(append([]uint32{}, rp...), 5, 0, 9, 0, 0, 0, 0), 0, "unknown object 9"},
		{"index unknown", append(append([]uint32{}, rp...), 6, 9, 1, 0, 0, 0, 0), 0, "unknown object 9"},
		{"index format", append(append([]uint32{}, rp...), 6, 1, 3, 0, 0, 0, 0), 0, "index format"},
		{"bundle unknown", append(append([]uint32{}, rp...), 11, 1, 9), 0, "unknown object 9"},
		{"indirect unknown", append(append([]uint32{}, rp...), 12, 9, 0, 0), 0, "unknown object 9"},
		{"write unknown", []uint32{30, 9, 0, 0, 0, 4}, 4, "unknown object 9"},
		{"write past data", []uint32{30, 1, 0, 0, 2, 4}, 4, "past 4 bytes"},
		{"write overflow", []uint32{30, 1, 0, 0, math.MaxUint32, 2}, 4, "past"},
		{"copy unknown src", []uint32{31, 9, 0, 0, 1, 0, 0, 4, 0}, 0, "unknown object 9"},
		{"copy unknown dst", []uint32{31, 1, 0, 0, 9, 0, 0, 4, 0}, 0, "unknown object 9"},
		{"pass left open", []uint32{22}, 0, "left open"},
	} {
		_, err := decode(c.w, c.ndata, known)
		if err == nil || !strings.Contains(err.Error(), c.err) {
			t.Errorf("%s: %v, want %q", c.name, err, c.err)
		}
	}
	if _, err := decode(nil, 0, known); err != nil {
		t.Fatal(err)
	}
}

// BenchmarkRecordDraws: the Go side of a batched frame, per draw
// (setBindGroup + draw), objects already in the table.
func BenchmarkRecordDraws(b *testing.B) {
	d := newDevice(jsValue{}, "d", nil)
	r := d.NewRecorder()
	groups := make([]*BindGroup, 16)
	for i := range groups {
		groups[i] = &BindGroup{}
		groups[i].batchID()
		groups[i].markPut()
	}
	pipe := &RenderPipeline{}
	pipe.batchID()
	pipe.markPut()
	for b.Loop() {
		p := r.BeginRenderPass(RenderPassDescriptor{})
		p.SetPipeline(pipe)
		for k := 0; k < 1000; k++ {
			p.SetBindGroup(0, groups[k&15])
			p.Draw(3, 1, 0, 0)
		}
		p.End()
		r.Reset()
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/1000, "ns/draw")
}

func TestRangeHelpersAndTimeline(t *testing.T) {
	size := uint64(0)
	wantBounds(t, bufferRange("op", nil, BufferUsageVertex, "VERTEX", 0, &size), "nil buffer")
	wantBounds(t, bufferRange("op", buf(8, BufferUsageVertex), BufferUsageVertex, "VERTEX", 9, &size), "past the buffer")
	wantBounds(t, checkIndirect("op", nil, BufferUsageIndirect, 0, 16), "nil buffer")
	wantBounds(t, checkIndirect("op", buf(64, BufferUsageIndirect), BufferUsageIndirect, 2, 16), "multiple of 4")
	if b2u(true) != 1 || b2u(false) != 0 {
		t.Fatal("b2u")
	}
	d := newDevice(jsValue{}, "d", nil)
	if d.Submitted() != 0 || d.Completed() != 0 {
		t.Fatal("fresh device timeline")
	}
	d.tl.Next()
	d.tl.Signal(1)
	if d.Submitted() != 1 || d.Completed() != 1 {
		t.Fatal("timeline")
	}
}
