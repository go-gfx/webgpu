//go:build js

package webgpu

import (
	"errors"
	"syscall/js"
	"testing"
)

func TestCanvasSupport(t *testing.T) {
	if err := CanvasSupport(); err != nil {
		t.Fatalf("Chromium offers a WebGPU OffscreenCanvas context: %v", err)
	}
}

// pixels draws an ImageBitmap on a 2D canvas and returns its RGBA bytes:
// what a page showing the frame would show.
func pixels(t *testing.T, bm js.Value, w, h int) []byte {
	t.Helper()
	c := jsGlobal.Get("OffscreenCanvas").New(w, h)
	g := c.Call("getContext", "2d")
	g.Call("drawImage", bm, 0, 0)
	data := g.Call("getImageData", 0, 0, w, h).Get("data")
	out := make([]byte, 4*w*h)
	js.CopyBytesToGo(out, data)
	return out
}

func TestSurfaceFrames(t *testing.T) {
	installReplay(t)
	d := newDev(t, &Quota{SurfaceWidth: 64, SurfaceHeight: 64})
	s, err := d.NewSurface(SurfaceDescriptor{Width: 32, Height: 16})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Destroy()
	if w, h := s.Size(); w != 32 || h != 16 || s.Format() == "" || s.Canvas().IsUndefined() {
		t.Fatalf("surface %dx%d %q", w, h, s.Format())
	}
	clear := func(c Color) RenderPassDescriptor {
		tex, err := s.CurrentTexture()
		if err != nil {
			t.Fatal(err)
		}
		if tex.Width() != uint32(32) && tex.Width() != uint32(48) {
			t.Fatalf("texture width %d", tex.Width())
		}
		tex.Destroy() // the canvas's: left alone
		v := must(tex.CreateView(nil))(t)
		return RenderPassDescriptor{ColorAttachments: []*RenderPassColorAttachment{{View: v, LoadOp: LoadOpClear, StoreOp: StoreOpStore, ClearValue: c}}}
	}
	frame := func(name string, want [4]byte, w, h int) {
		t.Helper()
		bm, err := s.Present()
		if err != nil {
			t.Fatal(err)
		}
		if bm.Get("width").Int() != w || bm.Get("height").Int() != h {
			t.Fatalf("%s: bitmap %dx%d", name, bm.Get("width").Int(), bm.Get("height").Int())
		}
		px := pixels(t, bm, w, h)
		bm.Call("close")
		for i := 0; i < len(px); i += 4 {
			if px[i] != want[0] || px[i+1] != want[1] || px[i+2] != want[2] || px[i+3] != want[3] {
				t.Fatalf("%s: pixel %d is %v, want %v (a blank frame reads 0 0 0 0)", name, i/4, px[i:i+4], want)
			}
		}
	}

	// Direct calls.
	enc := must(d.CreateCommandEncoder(""))(t)
	p := must(enc.BeginRenderPass(clear(Color{1, 0, 0, 1})))(t)
	if err := p.End(); err != nil {
		t.Fatal(err)
	}
	if err := d.Queue().Submit(must(enc.Finish())(t)); err != nil {
		t.Fatal(err)
	}
	frame("direct", [4]byte{255, 0, 0, 255}, 32, 16)

	// Batched, after a resize.
	if err := s.Resize(48, 16); err != nil {
		t.Fatal(err)
	}
	rec := d.NewRecorder()
	rp := rec.BeginRenderPass(clear(Color{0, 0, 1, 1}))
	if err := rp.End(); err != nil {
		t.Fatal(err)
	}
	if _, err := rec.Submit(); err != nil {
		t.Fatal(err)
	}
	frame("batched", [4]byte{0, 0, 255, 255}, 48, 16)

	var qe *QuotaError
	if err := s.Resize(65, 16); !errors.As(err, &qe) {
		t.Fatalf("resize past the quota: %v", err)
	}
	if _, err := d.NewSurface(SurfaceDescriptor{Width: 64, Height: 65}); !errors.As(err, &qe) {
		t.Fatalf("surface past the quota: %v", err)
	}
}

func TestSurfaceFromCanvas(t *testing.T) {
	d := newDev(t, nil)
	c := jsGlobal.Get("OffscreenCanvas").New(8, 8)
	s, err := d.NewSurfaceFromCanvas(c, SurfaceDescriptor{Format: TextureFormatRGBA8Unorm, AlphaMode: CanvasAlphaModePremultiplied})
	if err != nil {
		t.Fatal(err)
	}
	if w, h := s.Size(); w != 8 || h != 8 || s.Format() != TextureFormatRGBA8Unorm {
		t.Fatalf("%dx%d %s", w, h, s.Format())
	}
	if bm, err := s.Present(); err != nil || !bm.IsUndefined() {
		t.Fatalf("a transferred canvas presents by itself: %v %v", bm, err)
	}
	d.Destroy()
	<-d.Lost()
	if _, err := s.CurrentTexture(); !errors.Is(err, ErrDestroyed) {
		t.Fatalf("current texture on a lost device: %v", err)
	}
	if _, err := d.NewSurfaceFromCanvas(jsGlobal.Get("OffscreenCanvas").New(4, 4), SurfaceDescriptor{}); !errors.Is(err, ErrDestroyed) {
		t.Fatalf("surface on a lost device: %v", err)
	}
}
