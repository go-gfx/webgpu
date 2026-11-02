//go:build js

package webgpu

import "testing"

// frameAllocBudget is the allocations of one presented frame through the
// batched path: a recorded texture write, the surface's view (CurrentView),
// a one-draw pass and the submit. What remains is the syscall/js calls
// (a Call's argument slice and the closure of try).
const frameAllocBudget = 4

// TestFrameAllocs holds the per-frame allocations of the batched path to a
// budget, and logs where they come from.
func TestFrameAllocs(t *testing.T) {
	installReplay(t)
	d := newDev(t, &Quota{SurfaceWidth: 64, SurfaceHeight: 64})
	s := must(d.NewSurface(SurfaceDescriptor{Width: 32, Height: 16}))(t)
	defer s.Destroy()
	tex := must(d.CreateTexture(TextureDescriptor{Size: Extent3D{Width: 32, Height: 16}, Format: TextureFormatR8Uint,
		Usage: TextureUsageTextureBinding | TextureUsageCopyDst}))(t)
	px := make([]byte, 32*16)
	rec := d.NewRecorder()
	att := &RenderPassColorAttachment{LoadOp: LoadOpClear, StoreOp: StoreOpStore}
	desc := RenderPassDescriptor{ColorAttachments: []*RenderPassColorAttachment{att}}
	measure := func(name string, f func()) float64 {
		f() // warm up
		n := testing.AllocsPerRun(200, f)
		t.Logf("%-28s %5.1f allocs", name, n)
		return n
	}
	measure("Recorder.WriteTexture", func() {
		rec.Reset()
		rec.WriteTexture(TexelCopyTextureInfo{Texture: tex}, px, TexelCopyBufferLayout{BytesPerRow: 32}, Extent3D{Width: 32, Height: 16})
	})
	measure("CurrentView", func() { _, _ = s.CurrentView() })
	v := must(s.CurrentView())(t)
	att.View = v
	measure("BeginRenderPass+End", func() {
		rec.Reset()
		p := rec.BeginRenderPass(desc)
		_ = p.End()
	})
	n := measure("whole frame", func() {
		v, _ := s.CurrentView()
		att.View = v
		rec.Reset()
		rec.WriteTexture(TexelCopyTextureInfo{Texture: tex}, px, TexelCopyBufferLayout{BytesPerRow: 32}, Extent3D{Width: 32, Height: 16})
		p := rec.BeginRenderPass(desc)
		_ = p.End()
		_, _ = rec.Submit()
		_, _ = s.Present()
	})
	if n > frameAllocBudget {
		t.Fatalf("a frame allocates %.0f, budget %d", n, frameAllocBudget)
	}
}

// TestCurrentViewAcrossFrames: the one view handle stands for a new browser
// texture every frame, and each frame's pixels are the frame's own.
func TestCurrentViewAcrossFrames(t *testing.T) {
	installReplay(t)
	d := newDev(t, &Quota{SurfaceWidth: 64, SurfaceHeight: 64})
	s := must(d.NewSurface(SurfaceDescriptor{Width: 8, Height: 8}))(t)
	defer s.Destroy()
	rec := d.NewRecorder()
	var first *TextureView
	for i, c := range []Color{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}} {
		v := must(s.CurrentView())(t)
		if first == nil {
			first = v
		} else if v != first {
			t.Fatalf("frame %d: a new handle", i)
		}
		p := rec.BeginRenderPass(RenderPassDescriptor{ColorAttachments: []*RenderPassColorAttachment{{View: v, LoadOp: LoadOpClear, StoreOp: StoreOpStore, ClearValue: c}}})
		if err := p.End(); err != nil {
			t.Fatal(err)
		}
		if _, err := rec.Submit(); err != nil {
			t.Fatal(err)
		}
		bm := must(s.Present())(t)
		px := pixels(t, bm, 8, 8)
		bm.Call("close")
		want := [4]byte{byte(255 * c.R), byte(255 * c.G), byte(255 * c.B), 255}
		if [4]byte(px[:4]) != want || [4]byte(px[len(px)-4:]) != want {
			t.Fatalf("frame %d: pixels %v, want %v", i, px[:4], want)
		}
	}
	d.Destroy()
	<-d.Lost()
	if _, err := s.CurrentView(); err == nil {
		t.Fatal("a view on a lost device")
	}
}
