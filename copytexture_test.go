package webgpu

import (
	"errors"
	"math"
	"strings"
	"testing"
)

// The accepted and refused shapes of a texture-to-buffer copy: the checks are
// the direct path's, plus the buffer side (COPY_DST, size, 256-byte rows).
func TestCopyTextureToBufferChecks(t *testing.T) {
	src := func(w, h, layers, mips uint32, u TextureUsage) *Texture {
		return tex(TextureDescriptor{Size: Extent3D{Width: w, Height: h, DepthOrArrayLayers: layers}, MipLevelCount: mips,
			Format: TextureFormatRGBA8Unorm, Usage: u})
	}
	rgba := src(8, 4, 1, 2, TextureUsageCopySrc)
	layered := src(8, 4, 3, 1, TextureUsageCopySrc)
	dstOK := buf(1024, BufferUsageCopyDst|BufferUsageMapRead)
	full := Extent3D{Width: 8, Height: 4}
	l256 := TexelCopyBufferLayout{BytesPerRow: 256}

	type tc struct {
		name   string
		t      *Texture
		info   TexelCopyTextureInfo
		b      *Buffer
		layout TexelCopyBufferLayout
		size   Extent3D
		want   string // "" accepted, else a substring of the refusal
	}
	for _, c := range []tc{
		{"whole texture", rgba, TexelCopyTextureInfo{}, dstOK, l256, full, ""},
		{"mip 1", rgba, TexelCopyTextureInfo{MipLevel: 1}, dstOK, l256, Extent3D{Width: 4, Height: 2}, ""},
		{"one row, no bytesPerRow", rgba, TexelCopyTextureInfo{}, dstOK, TexelCopyBufferLayout{}, Extent3D{Width: 8, Height: 1}, ""},
		{"layout ends at the buffer's end", rgba, TexelCopyTextureInfo{}, buf(1024-224, BufferUsageCopyDst), TexelCopyBufferLayout{Offset: 0, BytesPerRow: 256}, full, ""},
		{"three layers", layered, TexelCopyTextureInfo{}, buf(3*1024, BufferUsageCopyDst), TexelCopyBufferLayout{BytesPerRow: 256, RowsPerImage: 4}, Extent3D{Width: 8, Height: 4, DepthOrArrayLayers: 3}, ""},

		{"texture without COPY_SRC", src(8, 4, 1, 1, TextureUsageCopyDst), TexelCopyTextureInfo{}, dstOK, l256, full, "COPY_SRC"},
		{"buffer without COPY_DST", rgba, TexelCopyTextureInfo{}, buf(1024, BufferUsageMapRead), l256, full, "COPY_DST"},
		{"bytesPerRow 255", rgba, TexelCopyTextureInfo{}, dstOK, TexelCopyBufferLayout{BytesPerRow: 255}, full, "multiple of 256"},
		{"bytesPerRow 512+4", rgba, TexelCopyTextureInfo{}, dstOK, TexelCopyBufferLayout{BytesPerRow: 516}, full, "multiple of 256"},
		{"bytesPerRow 0 with rows", rgba, TexelCopyTextureInfo{}, dstOK, TexelCopyBufferLayout{}, full, "bytesPerRow is required"},
		{"mip level out of range", rgba, TexelCopyTextureInfo{MipLevel: 2}, dstOK, l256, Extent3D{Width: 1, Height: 1}, "mip level 2"},
		{"mip 1 is smaller", rgba, TexelCopyTextureInfo{MipLevel: 1}, dstOK, l256, full, "outside mip level 1"},
		{"origin past width", rgba, TexelCopyTextureInfo{Origin: Origin3D{X: 4}}, dstOK, l256, full, "outside mip level"},
		{"origin overflow", rgba, TexelCopyTextureInfo{Origin: Origin3D{Y: math.MaxUint32}}, dstOK, l256, full, "outside mip level"},
		{"extent overflow", rgba, TexelCopyTextureInfo{Origin: Origin3D{X: 1}}, dstOK, l256, Extent3D{Width: math.MaxUint32, Height: 1}, "outside mip level"},
		{"layers past the texture", rgba, TexelCopyTextureInfo{}, dstOK, TexelCopyBufferLayout{BytesPerRow: 256, RowsPerImage: 4}, Extent3D{Width: 8, Height: 4, DepthOrArrayLayers: 2}, "outside mip level"},
		{"layout past the buffer", rgba, TexelCopyTextureInfo{}, buf(512, BufferUsageCopyDst), l256, full, "needs"},
		{"offset past the buffer", rgba, TexelCopyTextureInfo{}, dstOK, TexelCopyBufferLayout{Offset: 1 << 20, BytesPerRow: 256}, full, "needs"},
		{"offset overflows uint64", rgba, TexelCopyTextureInfo{}, dstOK, TexelCopyBufferLayout{Offset: math.MaxUint64 - 8, BytesPerRow: 256}, full, "needs"},
		{"layout overflows uint64", layered, TexelCopyTextureInfo{}, dstOK, TexelCopyBufferLayout{BytesPerRow: math.MaxUint32 - 255, RowsPerImage: math.MaxUint32}, Extent3D{Width: 8, Height: 4, DepthOrArrayLayers: 3}, "overflows"},
		{"rowsPerImage below the rows", layered, TexelCopyTextureInfo{}, dstOK, TexelCopyBufferLayout{BytesPerRow: 256, RowsPerImage: 2}, Extent3D{Width: 8, Height: 4, DepthOrArrayLayers: 2}, "rowsPerImage"},
		{"bad aspect", rgba, TexelCopyTextureInfo{Aspect: "plane0-only"}, dstOK, l256, full, "aspect"},
		{"depth-stencil without an aspect", tex(TextureDescriptor{Size: Extent3D{Width: 8, Height: 4}, Format: "depth24plus-stencil8", Usage: TextureUsageCopySrc}),
			TexelCopyTextureInfo{}, dstOK, l256, full, "aspect"},
	} {
		r := newDevice(jsValue{}, "d", nil).NewRecorder()
		info := c.info
		info.Texture = c.t
		r.CopyTextureToBuffer(info, c.b, c.layout, c.size)
		if c.want == "" {
			if r.Err() != nil || len(r.words) != 15 {
				t.Errorf("%s: err %v, %d words", c.name, r.Err(), len(r.words))
			}
			continue
		}
		var be *BoundsError
		if !errors.As(r.Err(), &be) || !strings.Contains(r.Err().Error(), c.want) {
			t.Errorf("%s: %v, want a bounds error with %q", c.name, r.Err(), c.want)
		}
		if len(r.words) != 0 {
			t.Errorf("%s: refused but recorded %d words", c.name, len(r.words))
		}
	}

	// Not a size problem: nil objects and a pass.
	r := newDevice(jsValue{}, "d", nil).NewRecorder()
	r.CopyTextureToBuffer(TexelCopyTextureInfo{}, dstOK, l256, full)
	if r.Err() == nil {
		t.Error("nil texture accepted")
	}
	r = newDevice(jsValue{}, "d", nil).NewRecorder()
	r.CopyTextureToBuffer(TexelCopyTextureInfo{Texture: rgba}, nil, l256, full)
	if r.Err() == nil {
		t.Error("nil buffer accepted")
	}
	r = newDevice(jsValue{}, "d", nil).NewRecorder()
	r.BeginComputePass()
	r.CopyTextureToBuffer(TexelCopyTextureInfo{Texture: rgba}, dstOK, l256, full)
	if !errors.Is(r.Err(), ErrRecorderState) {
		t.Errorf("in a pass: %v", r.Err())
	}
}
