//go:build js

package webgpu

import (
	"context"
	"fmt"
)

// ReadTexture copies a 2D region of a texture to Go memory: width x height
// texels of bytesPerTexel bytes each, row after row without padding. It
// does what every readback needs: a copy to a mappable buffer whose rows
// are padded to 256 bytes (bytesPerRow must be a multiple of 256), a
// submit, a map, and the unpadding. The texture needs COPY_SRC. Meant for
// tests and screenshots, not for a frame loop (it waits for the GPU).
func (d *Device) ReadTexture(ctx context.Context, src TexelCopyTextureInfo, width, height, bytesPerTexel int) ([]byte, error) {
	if width < 1 || height < 1 || bytesPerTexel < 1 {
		return nil, bounds("readTexture", "size %dx%d of %d-byte texels", width, height, bytesPerTexel)
	}
	row := width * bytesPerTexel
	pitch := (row + 255) &^ 255
	buf, err := d.CreateBuffer(BufferDescriptor{Label: "readTexture", Size: uint64(pitch) * uint64(height), Usage: BufferUsageMapRead | BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	defer buf.Destroy()
	enc, err := d.CreateCommandEncoder("readTexture")
	if err != nil {
		return nil, err
	}
	if err := enc.CopyTextureToBuffer(src, TexelCopyBufferInfo{Buffer: buf, Layout: TexelCopyBufferLayout{BytesPerRow: uint32(pitch)}},
		Extent3D{Width: uint32(width), Height: uint32(height)}); err != nil {
		return nil, err
	}
	cb, err := enc.Finish()
	if err != nil {
		return nil, err
	}
	if err := d.Queue().Submit(cb); err != nil {
		return nil, err
	}
	if err := buf.MapAsync(ctx, MapModeRead, 0, 0); err != nil {
		return nil, fmt.Errorf("readTexture: %w", err)
	}
	raw := make([]byte, pitch*height)
	if err := buf.ReadMapped(0, raw); err != nil {
		return nil, err
	}
	if err := buf.Unmap(); err != nil {
		return nil, err
	}
	out := make([]byte, row*height)
	for y := 0; y < height; y++ {
		copy(out[y*row:], raw[y*pitch:y*pitch+row])
	}
	return out, nil
}
