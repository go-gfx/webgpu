//go:build js

package webgpu

import (
	"fmt"
	"syscall/js"
)

// CanvasSupport reports whether this window or worker can render WebGPU
// into an OffscreenCanvas: nil if so, ErrNoCanvasContext (wrapped, with
// the reason) if not. MDN's compatibility data lists the webgpu context
// of OffscreenCanvas in Chrome 113+ (all platforms from 144), Safari 26,
// and Firefox 141 on Windows only: elsewhere a program should fall back
// to another renderer, and this is the signal.
func CanvasSupport() error {
	oc := jsGlobal.Get("OffscreenCanvas")
	if oc.Type() != js.TypeFunction {
		return fmt.Errorf("%w: OffscreenCanvas is not available", ErrNoCanvasContext)
	}
	if _, err := Navigator(); err != nil {
		return fmt.Errorf("%w: %v", ErrNoCanvasContext, err)
	}
	var ctx js.Value
	if err := try("getContext", func() { ctx = oc.New(1, 1).Call("getContext", "webgpu") }); err != nil {
		return fmt.Errorf("%w: %v", ErrNoCanvasContext, err)
	}
	if nullish(ctx) {
		return fmt.Errorf("%w: getContext(\"webgpu\") returned null", ErrNoCanvasContext)
	}
	return nil
}

// Surface is an OffscreenCanvas with a WebGPU context, configured for a
// device. A standalone surface (NewSurface) never shows itself: Present
// moves the frame out as an ImageBitmap (transferToImageBitmap, without a
// copy) for the caller to post to a page and draw. A surface over a canvas
// a page transferred (NewSurfaceFromCanvas) is presented by the browser at
// the end of each task, and Present returns no bitmap.
type Surface struct {
	dev        *Device
	canvas     js.Value
	ctx        js.Value
	desc       SurfaceDescriptor
	format     TextureFormat
	standalone bool
}

// NewSurface creates a standalone OffscreenCanvas of desc's size, within
// the device's quota (SurfaceWidth, SurfaceHeight).
func (d *Device) NewSurface(desc SurfaceDescriptor) (*Surface, error) {
	if err := d.quota.checkSurfaceSize(desc.Width, desc.Height); err != nil {
		return nil, err
	}
	if err := CanvasSupport(); err != nil {
		return nil, err
	}
	var c js.Value
	if err := try("OffscreenCanvas", func() { c = jsGlobal.Get("OffscreenCanvas").New(desc.Width, desc.Height) }); err != nil {
		return nil, err
	}
	return d.surface(c, desc, true)
}

// NewSurfaceFromCanvas configures an OffscreenCanvas the caller owns (one
// a page made with transferControlToOffscreen and posted to a worker); its
// size is the canvas's, checked against the quota.
func (d *Device) NewSurfaceFromCanvas(canvas js.Value, desc SurfaceDescriptor) (*Surface, error) {
	desc.Width, desc.Height = canvas.Get("width").Int(), canvas.Get("height").Int()
	if err := d.quota.checkSurfaceSize(desc.Width, desc.Height); err != nil {
		return nil, err
	}
	return d.surface(canvas, desc, false)
}

func (d *Device) surface(c js.Value, desc SurfaceDescriptor, standalone bool) (*Surface, error) {
	if d.isLost() {
		return nil, ErrDestroyed
	}
	var ctx js.Value
	if err := try("getContext", func() { ctx = c.Call("getContext", "webgpu") }); err != nil {
		return nil, err
	}
	if nullish(ctx) {
		return nil, fmt.Errorf("%w: getContext(\"webgpu\") returned null", ErrNoCanvasContext)
	}
	s := &Surface{dev: d, canvas: c, ctx: ctx, desc: desc, standalone: standalone}
	g, err := Navigator()
	if err != nil {
		return nil, err
	}
	s.format = desc.Format
	if s.format == "" {
		s.format = g.PreferredCanvasFormat()
	}
	if err := s.configure(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Surface) configure() error {
	return try("configure", func() { s.ctx.Call("configure", js.ValueOf(s.desc.config(s.dev.v, s.format))) })
}

// Format is the format of the surface's textures.
func (s *Surface) Format() TextureFormat { return s.format }

// Size is the canvas size.
func (s *Surface) Size() (width, height int) { return s.desc.Width, s.desc.Height }

// Canvas returns the OffscreenCanvas.
func (s *Surface) Canvas() js.Value { return s.canvas }

// Resize changes the canvas size, within the quota.
func (s *Surface) Resize(width, height int) error {
	if err := s.dev.quota.checkSurfaceSize(width, height); err != nil {
		return err
	}
	if err := try("resize", func() {
		s.canvas.Set("width", width)
		s.canvas.Set("height", height)
	}); err != nil {
		return err
	}
	s.desc.Width, s.desc.Height = width, height
	return nil
}

// CurrentTexture is context.getCurrentTexture(): the texture to render
// this frame into. It is the canvas's, not the caller's: not charged to
// the quota, and Destroy leaves it alone.
func (s *Surface) CurrentTexture() (*Texture, error) {
	if s.dev.isLost() {
		return nil, ErrDestroyed
	}
	var v js.Value
	if err := try("getCurrentTexture", func() { v = s.ctx.Call("getCurrentTexture") }); err != nil {
		return nil, err
	}
	usage := s.desc.Usage
	if usage == 0 {
		usage = TextureUsageRenderAttachment
	}
	t := &Texture{canvas: true, desc: TextureDescriptor{
		Size: Extent3D{Width: uint32(s.desc.Width), Height: uint32(s.desc.Height)}, Format: s.format, Usage: usage,
	}.norm()}
	t.v = v
	return t, nil
}

// Present ends the frame. For a standalone surface it returns the frame
// as an ImageBitmap (transferToImageBitmap: the image moves out without a
// copy); the caller must transfer it (postMessage) or close it. For a
// transferred canvas the browser presents by itself and Present returns
// js.Undefined().
func (s *Surface) Present() (js.Value, error) {
	if !s.standalone {
		return js.Undefined(), nil
	}
	var bm js.Value
	if err := try("transferToImageBitmap", func() { bm = s.canvas.Call("transferToImageBitmap") }); err != nil {
		return js.Value{}, err
	}
	return bm, nil
}

// Destroy unconfigures the context, destroying its textures.
func (s *Surface) Destroy() {
	_ = try("unconfigure", func() { s.ctx.Call("unconfigure") })
}
