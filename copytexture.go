package webgpu

import "github.com/go-gfx/webgpu/wire"

// Texture read-back: encoder.copyTextureToBuffer through the batched stream
// (wire.CopyTextureToBuffer, stream version 5). A frame can be read without a
// compute pass that loads every texel into a storage buffer: copy the texture
// into a COPY_DST buffer (rows padded to 256 bytes), then Recorder.ReadBuffer
// it, if it is a MAP_READ buffer or after a CopyBufferToBuffer into one.

// CheckCopyTextureToBuffer applies WebGPU's validation of a texture-to-buffer
// copy, before any GPU is involved: the texture has COPY_SRC and is not
// multisampled, the mip level exists, the region lies inside the mip level
// and on texel block boundaries, the buffer has COPY_DST usage, bytesPerRow is
// a multiple of 256 (wire.CopyBytesPerRowAlignment), and the whole layout
// (offset, rows, images) lies inside bufSize bytes. The arithmetic is 64-bit
// and overflow is a refusal. desc is the texture's descriptor as the device
// sent it; the error is a bounds error.
func CheckCopyTextureToBuffer(desc TextureDescriptor, src TexelCopyTextureInfo, bufSize uint64, bufUsage BufferUsage, layout TexelCopyBufferLayout, size Extent3D) error {
	const op = "copyTextureToBuffer"
	if bufUsage&BufferUsageCopyDst == 0 {
		return bounds(op, "buffer lacks the COPY_DST usage")
	}
	if layout.BytesPerRow%wire.CopyBytesPerRowAlignment != 0 {
		return bounds(op, "bytesPerRow %d is not a multiple of %d", layout.BytesPerRow, wire.CopyBytesPerRowAlignment)
	}
	src.Texture = &Texture{desc: desc.norm()}
	return checkTextureCopy(op, src, layout, bufSize, size, false)
}

// CopyTextureToBuffer records encoder.copyTextureToBuffer, checked like the
// direct path: see CheckCopyTextureToBuffer. The bytes are in dst once the
// submit's GPU work has run; Recorder.ReadBuffer brings them back.
func (r *Recorder) CopyTextureToBuffer(src TexelCopyTextureInfo, dst *Buffer, layout TexelCopyBufferLayout, size Extent3D) {
	const op = "copyTextureToBuffer"
	if !r.in(0, op+" inside a pass") {
		return
	}
	if src.Texture == nil || dst == nil {
		r.fail(bounds(op, "nil texture or buffer"))
		return
	}
	if err := CheckCopyTextureToBuffer(src.Texture.desc, src, dst.size, dst.usage, layout, size); err != nil {
		r.fail(err)
		return
	}
	asp, ok := aspectCode[src.Aspect]
	if !ok {
		r.fail(bounds(op, "aspect %q", src.Aspect))
		return
	}
	size = size.norm()
	lo, hi := lohi(layout.Offset)
	r.words = append(r.words, uint32(opCopyTextureToBuffer), r.id(&src.Texture.object), src.MipLevel,
		src.Origin.X, src.Origin.Y, src.Origin.Z, asp, r.id(&dst.object), lo, hi,
		layout.BytesPerRow, layout.RowsPerImage, size.Width, size.Height, size.DepthOrArrayLayers)
}
