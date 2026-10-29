package webgpu

import (
	"fmt"
	"math"
	"math/bits"
)

// Checks done in Go before a call reaches the browser. WebGPU validates
// these too, but reports a failure asynchronously, as an error on the
// device that a caller who did not push an error scope never sees; here
// an out-of-range write is an error at the call. Arithmetic is overflow
// checked: sizes come from untrusted data (game maps, container guests).

// checkRange: [off, off+size) lies inside b.
func checkRange(op string, b *Buffer, off, size uint64) error {
	if b == nil {
		return bounds(op, "nil buffer")
	}
	end, carry := bits.Add64(off, size, 0)
	if carry != 0 || end > b.size {
		return bounds(op, "range [%d, %d+%d) outside the buffer's %d bytes", off, off, size, b.size)
	}
	return nil
}

func checkUsage(op string, b *Buffer, want BufferUsage, name string) error {
	if b.usage&want == 0 {
		return bounds(op, "buffer %q lacks the %s usage", b.label, name)
	}
	return nil
}

// checkWriteBuffer validates Queue.WriteBuffer (GPUQueue.writeBuffer):
// COPY_DST usage, 4-byte aligned offset and size, inside the buffer.
func checkWriteBuffer(b *Buffer, off uint64, n int) error {
	const op = "writeBuffer"
	if b == nil {
		return bounds(op, "nil buffer")
	}
	if err := checkUsage(op, b, BufferUsageCopyDst, "COPY_DST"); err != nil {
		return err
	}
	if off%4 != 0 || n%4 != 0 {
		return bounds(op, "offset %d and size %d must be multiples of 4", off, n)
	}
	return checkRange(op, b, off, uint64(n))
}

// checkCopyBufferToBuffer validates CommandEncoder.CopyBufferToBuffer.
func checkCopyBufferToBuffer(src *Buffer, so uint64, dst *Buffer, do, size uint64) error {
	const op = "copyBufferToBuffer"
	if src == nil || dst == nil {
		return bounds(op, "nil buffer")
	}
	if err := checkUsage(op, src, BufferUsageCopySrc, "COPY_SRC"); err != nil {
		return err
	}
	if err := checkUsage(op, dst, BufferUsageCopyDst, "COPY_DST"); err != nil {
		return err
	}
	if so%4 != 0 || do%4 != 0 || size%4 != 0 {
		return bounds(op, "offsets %d, %d and size %d must be multiples of 4", so, do, size)
	}
	if err := checkRange(op, src, so, size); err != nil {
		return err
	}
	return checkRange(op, dst, do, size)
}

// mipExtent is the logical miplevel-specific texture extent
// (#abstract-opdef-logical-miplevel-specific-texture-extent).
func mipExtent(d TextureDescriptor, level uint32) Extent3D {
	shr := func(v uint32) uint32 {
		if level >= 32 {
			return 1
		}
		return max(1, v>>level)
	}
	switch d.Dimension {
	case TextureDimension1D:
		return Extent3D{shr(d.Size.Width), 1, 1}
	case TextureDimension3D:
		return Extent3D{shr(d.Size.Width), shr(d.Size.Height), shr(d.Size.DepthOrArrayLayers)}
	}
	return Extent3D{shr(d.Size.Width), shr(d.Size.Height), d.Size.DepthOrArrayLayers}
}

func roundUp(v, m uint32) uint64 { return (uint64(v) + uint64(m) - 1) / uint64(m) * uint64(m) }

// TextureBytes is what a texture costs in GPU memory as this package
// counts it: every mip level's physical extent, in texel blocks, times the
// format's memory cost, times the sample count. It is an upper bound for
// quota purposes, not a promise about the driver's allocation.
func TextureBytes(d TextureDescriptor) (int64, error) {
	d = d.norm()
	fi, err := d.Format.Info()
	if err != nil {
		return 0, err
	}
	if d.MipLevelCount > 32 {
		return 0, fmt.Errorf("webgpu: %d mip levels", d.MipLevelCount)
	}
	var total uint64
	for l := uint32(0); l < d.MipLevelCount; l++ {
		e := mipExtent(d, l)
		w := roundUp(e.Width, fi.BlockWidth) / uint64(fi.BlockWidth)
		h := roundUp(e.Height, fi.BlockHeight) / uint64(fi.BlockHeight)
		hi, n := bits.Mul64(w, h)
		hi2, n2 := bits.Mul64(n, uint64(e.DepthOrArrayLayers))
		hi3, n3 := bits.Mul64(n2, uint64(fi.MemoryBytes)*uint64(d.SampleCount))
		var carry uint64
		total, carry = bits.Add64(total, n3, 0)
		if hi|hi2|hi3|carry != 0 || total > math.MaxInt64 {
			return 0, fmt.Errorf("webgpu: texture size overflows")
		}
	}
	return int64(total), nil
}

// copyFootprint returns the texel block copy footprint for a copy of the
// given aspect of t, and the block size.
func copyFootprint(op string, t *Texture, aspect TextureAspect, dst bool) (bytes, bw, bh uint32, err error) {
	fi, err := t.desc.Format.Info()
	if err != nil {
		return 0, 0, 0, err
	}
	if !fi.Depth && !fi.Stencil {
		return fi.CopyBytes, fi.BlockWidth, fi.BlockHeight, nil
	}
	// Depth-stencil formats (#depth-formats): one aspect at a time.
	if aspect == "" || aspect == TextureAspectAll {
		switch {
		case fi.Depth && fi.Stencil:
			return 0, 0, 0, bounds(op, "%s: name the aspect to copy (depth-only or stencil-only)", t.desc.Format)
		case fi.Stencil:
			aspect = TextureAspectStencilOnly
		default:
			aspect = TextureAspectDepthOnly
		}
	}
	switch {
	case aspect == TextureAspectStencilOnly && fi.Stencil:
		return 1, 1, 1, nil
	case aspect == TextureAspectDepthOnly && fi.Depth:
		switch t.desc.Format {
		case TextureFormatDepth16Unorm:
			return 2, 1, 1, nil
		case TextureFormatDepth32Float, "depth32float-stencil8":
			if dst {
				return 0, 0, 0, bounds(op, "the depth aspect of %s cannot be a copy destination", t.desc.Format)
			}
			return 4, 1, 1, nil
		}
		return 0, 0, 0, bounds(op, "the depth aspect of %s cannot be copied", t.desc.Format)
	}
	return 0, 0, 0, bounds(op, "%s has no %s aspect", t.desc.Format, aspect)
}

// checkTextureCopy validates a texel copy between linear data of dataLen
// bytes and a texture region: the region inside the mip level, block
// alignment, and the layout fitting the data (#abstract-opdef-validating-
// linear-texture-data). dst says the texture is written.
func checkTextureCopy(op string, info TexelCopyTextureInfo, layout TexelCopyBufferLayout, dataLen uint64, size Extent3D, dst bool) error {
	t := info.Texture
	if t == nil {
		return bounds(op, "nil texture")
	}
	size = size.norm()
	want := TextureUsageCopySrc
	if dst {
		want = TextureUsageCopyDst
	}
	if t.desc.Usage&want == 0 {
		return bounds(op, "texture %q lacks the %s usage", t.label, map[bool]string{true: "COPY_DST", false: "COPY_SRC"}[dst])
	}
	if t.desc.SampleCount > 1 {
		return bounds(op, "multisampled texture")
	}
	if info.MipLevel >= t.desc.MipLevelCount {
		return bounds(op, "mip level %d of %d", info.MipLevel, t.desc.MipLevelCount)
	}
	fb, bw, bh, err := copyFootprint(op, t, info.Aspect, dst)
	if err != nil {
		return err
	}
	ext := mipExtent(t.desc, info.MipLevel)
	phys := Extent3D{uint32(roundUp(ext.Width, bw)), uint32(roundUp(ext.Height, bh)), ext.DepthOrArrayLayers}
	o := info.Origin
	if uint64(o.X)+uint64(size.Width) > uint64(phys.Width) || uint64(o.Y)+uint64(size.Height) > uint64(phys.Height) ||
		uint64(o.Z)+uint64(size.DepthOrArrayLayers) > uint64(phys.DepthOrArrayLayers) {
		return bounds(op, "region %dx%dx%d at (%d,%d,%d) outside mip level %d's %dx%dx%d",
			size.Width, size.Height, size.DepthOrArrayLayers, o.X, o.Y, o.Z, info.MipLevel, phys.Width, phys.Height, phys.DepthOrArrayLayers)
	}
	if o.X%bw != 0 || o.Y%bh != 0 || size.Width%bw != 0 || size.Height%bh != 0 {
		return bounds(op, "origin and size must be multiples of the %dx%d texel block", bw, bh)
	}
	return checkLinear(op, layout, dataLen, size, fb, bw, bh)
}

// checkLinear is "validating linear texture data".
func checkLinear(op string, l TexelCopyBufferLayout, byteSize uint64, e Extent3D, footprint, bw, bh uint32) error {
	wBlocks := uint64(e.Width / bw)
	hBlocks := uint64(e.Height / bh)
	lastRow := wBlocks * uint64(footprint)
	if hBlocks > 1 && l.BytesPerRow == 0 {
		return bounds(op, "bytesPerRow is required for %d block rows", hBlocks)
	}
	if e.DepthOrArrayLayers > 1 && (l.BytesPerRow == 0 || l.RowsPerImage == 0) {
		return bounds(op, "bytesPerRow and rowsPerImage are required for %d layers", e.DepthOrArrayLayers)
	}
	if l.BytesPerRow != 0 && uint64(l.BytesPerRow) < lastRow {
		return bounds(op, "bytesPerRow %d < %d bytes in a row", l.BytesPerRow, lastRow)
	}
	if l.RowsPerImage != 0 && uint64(l.RowsPerImage) < hBlocks {
		return bounds(op, "rowsPerImage %d < %d block rows", l.RowsPerImage, hBlocks)
	}
	bpr, rpi := uint64(l.BytesPerRow), uint64(l.RowsPerImage)
	var required uint64
	if e.DepthOrArrayLayers > 0 {
		// bpr, rpi < 2^32 and layers < 2^32: the products fit below 2^96;
		// check with bits.Mul64.
		hi, img := bits.Mul64(bpr, rpi)
		hi2, imgs := bits.Mul64(img, uint64(e.DepthOrArrayLayers-1))
		if hi|hi2 != 0 {
			return bounds(op, "layout overflows")
		}
		required = imgs
		if hBlocks > 0 {
			// bpr and hBlocks are below 2^32, lastRow below 2^36.
			var c uint64
			required, c = bits.Add64(required, bpr*(hBlocks-1)+lastRow, 0)
			if c != 0 {
				return bounds(op, "layout overflows")
			}
		}
	}
	end, carry := bits.Add64(l.Offset, required, 0)
	if carry != 0 || end > byteSize {
		return bounds(op, "layout needs %d bytes from offset %d; the data has %d", required, l.Offset, byteSize)
	}
	return nil
}
