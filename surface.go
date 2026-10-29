package webgpu

// SurfaceDescriptor configures a Surface: an OffscreenCanvas with a WebGPU
// context (GPUCanvasConfiguration, https://www.w3.org/TR/webgpu/#canvas-configuration).
type SurfaceDescriptor struct {
	Width, Height int
	// Format is the canvas textures' format; empty is the browser's
	// preferred canvas format.
	Format TextureFormat
	// Usage of the canvas textures; zero is RENDER_ATTACHMENT.
	Usage TextureUsage
	// AlphaMode; empty is "opaque".
	AlphaMode CanvasAlphaMode
}

// checkSurfaceSize bounds a surface by the quota (SurfaceWidth,
// SurfaceHeight) and refuses an empty one.
func (q *Quota) checkSurfaceSize(w, h int) error {
	if w < 1 || h < 1 {
		return bounds("surface", "size %dx%d", w, h)
	}
	if q == nil {
		return nil
	}
	if (q.SurfaceWidth > 0 && w > q.SurfaceWidth) || (q.SurfaceHeight > 0 && h > q.SurfaceHeight) {
		return &QuotaError{Resource: "surface", Limit: int64(q.SurfaceWidth) * int64(q.SurfaceHeight), Want: int64(w) * int64(h)}
	}
	return nil
}

func (s SurfaceDescriptor) config(dev any, preferred TextureFormat) map[string]any {
	f := s.Format
	if f == "" {
		f = preferred
	}
	d := map[string]any{"device": dev, "format": string(f)}
	if s.Usage != 0 {
		d["usage"] = uint32(s.Usage)
	}
	setStr(d, "alphaMode", s.AlphaMode)
	return d
}
