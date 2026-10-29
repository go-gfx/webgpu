package webgpu

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func buf(size uint64, usage BufferUsage) *Buffer {
	b := &Buffer{size: size, usage: usage}
	b.label = "b"
	return b
}

func tex(d TextureDescriptor) *Texture {
	t := &Texture{desc: d.norm()}
	t.label = "t"
	return t
}

func wantBounds(t *testing.T, err error, sub string) {
	t.Helper()
	var be *BoundsError
	if !errors.As(err, &be) {
		t.Fatalf("got %v, want a *BoundsError containing %q", err, sub)
	}
	if !strings.Contains(err.Error(), sub) {
		t.Fatalf("got %q, want it to contain %q", err, sub)
	}
}

func TestCheckRange(t *testing.T) {
	b := buf(64, BufferUsageCopyDst)
	if err := checkRange("op", b, 0, 64); err != nil {
		t.Fatal(err)
	}
	if err := checkRange("op", b, 64, 0); err != nil {
		t.Fatal(err)
	}
	wantBounds(t, checkRange("op", b, 1, 64), "outside")
	wantBounds(t, checkRange("op", b, math.MaxUint64, 2), "outside") // overflow
	wantBounds(t, checkRange("op", nil, 0, 0), "nil buffer")
}

func TestCheckWriteBuffer(t *testing.T) {
	b := buf(64, BufferUsageCopyDst|BufferUsageUniform)
	if err := checkWriteBuffer(b, 60, 4); err != nil {
		t.Fatal(err)
	}
	wantBounds(t, checkWriteBuffer(b, 64, 4), "outside")
	wantBounds(t, checkWriteBuffer(b, 2, 4), "multiples of 4")
	wantBounds(t, checkWriteBuffer(b, 0, 3), "multiples of 4")
	wantBounds(t, checkWriteBuffer(buf(64, BufferUsageUniform), 0, 4), "COPY_DST")
	wantBounds(t, checkWriteBuffer(nil, 0, 4), "nil buffer")
}

func TestCheckCopyBufferToBuffer(t *testing.T) {
	src := buf(64, BufferUsageCopySrc)
	dst := buf(32, BufferUsageCopyDst)
	if err := checkCopyBufferToBuffer(src, 32, dst, 0, 32); err != nil {
		t.Fatal(err)
	}
	wantBounds(t, checkCopyBufferToBuffer(nil, 0, dst, 0, 4), "nil")
	wantBounds(t, checkCopyBufferToBuffer(dst, 0, dst, 0, 4), "COPY_SRC")
	wantBounds(t, checkCopyBufferToBuffer(src, 0, src, 0, 4), "COPY_DST")
	wantBounds(t, checkCopyBufferToBuffer(src, 1, dst, 0, 4), "multiples of 4")
	wantBounds(t, checkCopyBufferToBuffer(src, 64, dst, 0, 4), "outside")
	wantBounds(t, checkCopyBufferToBuffer(src, 0, dst, 0, 64), "outside")
}

func TestMipExtent(t *testing.T) {
	for _, c := range []struct {
		d     TextureDescriptor
		level uint32
		want  Extent3D
	}{
		{TextureDescriptor{Size: Extent3D{256, 64, 6}}, 2, Extent3D{64, 16, 6}},
		{TextureDescriptor{Size: Extent3D{256, 64, 6}}, 7, Extent3D{2, 1, 6}},
		{TextureDescriptor{Size: Extent3D{256, 64, 6}}, 40, Extent3D{1, 1, 6}},
		{TextureDescriptor{Size: Extent3D{256, 1, 1}, Dimension: TextureDimension1D}, 3, Extent3D{32, 1, 1}},
		{TextureDescriptor{Size: Extent3D{16, 8, 4}, Dimension: TextureDimension3D}, 2, Extent3D{4, 2, 1}},
	} {
		if got := mipExtent(c.d.norm(), c.level); got != c.want {
			t.Errorf("mipExtent(%+v, %d) = %+v, want %+v", c.d, c.level, got, c.want)
		}
	}
}

func TestTextureBytes(t *testing.T) {
	for _, c := range []struct {
		d    TextureDescriptor
		want int64
	}{
		{TextureDescriptor{Size: Extent3D{Width: 64, Height: 64}, Format: "r8uint"}, 4096},
		// 64x64 rgba8 with a full chain: 4*(4096+1024+256+64+16+4+1).
		{TextureDescriptor{Size: Extent3D{Width: 64, Height: 64}, Format: "rgba8unorm", MipLevelCount: 7}, 4 * 5461},
		{TextureDescriptor{Size: Extent3D{Width: 8, Height: 8}, Format: "depth24plus-stencil8", SampleCount: 4}, 8 * 64 * 4},
		// bc1: 4x4 blocks of 8 bytes; 6x6 rounds up to 2x2 blocks.
		{TextureDescriptor{Size: Extent3D{Width: 6, Height: 6}, Format: "bc1-rgba-unorm"}, 32},
		{TextureDescriptor{Size: Extent3D{Width: 16, Height: 16, DepthOrArrayLayers: 6}, Format: "r8unorm"}, 1536},
	} {
		got, err := TextureBytes(c.d)
		if err != nil || got != c.want {
			t.Errorf("TextureBytes(%+v) = %d, %v; want %d", c.d, got, err, c.want)
		}
	}
	if _, err := TextureBytes(TextureDescriptor{Format: "nope"}); err == nil {
		t.Error("unknown format accepted")
	}
	if _, err := TextureBytes(TextureDescriptor{Format: "r8unorm", MipLevelCount: 33}); err == nil {
		t.Error("33 mip levels accepted")
	}
	huge := TextureDescriptor{Size: Extent3D{math.MaxUint32, math.MaxUint32, math.MaxUint32}, Format: "rgba32float"}
	if _, err := TextureBytes(huge); err == nil {
		t.Error("overflowing texture accepted")
	}
	// Sum overflow past MaxInt64 without a single product overflowing.
	big := TextureDescriptor{Size: Extent3D{1 << 31, 1 << 31, 1}, Format: "rg8unorm"}
	if _, err := TextureBytes(big); err == nil {
		t.Error("texture past MaxInt64 accepted")
	}
}

func TestCheckTextureCopy(t *testing.T) {
	t8 := tex(TextureDescriptor{Size: Extent3D{Width: 64, Height: 32}, Format: "r8uint", Usage: TextureUsageCopyDst, MipLevelCount: 2})
	dst := TexelCopyTextureInfo{Texture: t8}
	ok := func(info TexelCopyTextureInfo, l TexelCopyBufferLayout, n uint64, e Extent3D) {
		t.Helper()
		if err := checkTextureCopy("writeTexture", info, l, n, e, true); err != nil {
			t.Fatal(err)
		}
	}
	bad := func(info TexelCopyTextureInfo, l TexelCopyBufferLayout, n uint64, e Extent3D, dstSide bool, sub string) {
		t.Helper()
		wantBounds(t, checkTextureCopy("writeTexture", info, l, n, e, dstSide), sub)
	}
	ok(dst, TexelCopyBufferLayout{BytesPerRow: 64}, 64*32, Extent3D{Width: 64, Height: 32})
	// A dirty rectangle: 16x8 at (8,4), rows 16 bytes apart.
	ok(TexelCopyTextureInfo{Texture: t8, Origin: Origin3D{X: 8, Y: 4}}, TexelCopyBufferLayout{BytesPerRow: 16}, 128, Extent3D{Width: 16, Height: 8})
	// One row needs no bytesPerRow.
	ok(dst, TexelCopyBufferLayout{}, 64, Extent3D{Width: 64})
	// Mip level 1 is 32x16.
	ok(TexelCopyTextureInfo{Texture: t8, MipLevel: 1}, TexelCopyBufferLayout{BytesPerRow: 32}, 512, Extent3D{Width: 32, Height: 16})

	bad(dst, TexelCopyBufferLayout{BytesPerRow: 64}, 64*32-1, Extent3D{Width: 64, Height: 32}, true, "the data has")
	bad(TexelCopyTextureInfo{Texture: t8, Origin: Origin3D{X: 60}}, TexelCopyBufferLayout{}, 64, Extent3D{Width: 8}, true, "outside")
	bad(TexelCopyTextureInfo{Texture: t8, Origin: Origin3D{Y: 32}}, TexelCopyBufferLayout{}, 64, Extent3D{Width: 8}, true, "outside")
	bad(TexelCopyTextureInfo{Texture: t8, Origin: Origin3D{Z: 1}}, TexelCopyBufferLayout{}, 64, Extent3D{Width: 8}, true, "outside")
	bad(TexelCopyTextureInfo{Texture: t8, MipLevel: 1}, TexelCopyBufferLayout{BytesPerRow: 64}, 64*32, Extent3D{Width: 64, Height: 32}, true, "outside")
	bad(TexelCopyTextureInfo{Texture: t8, MipLevel: 2}, TexelCopyBufferLayout{}, 1, Extent3D{Width: 1}, true, "mip level")
	bad(dst, TexelCopyBufferLayout{}, 64*32, Extent3D{Width: 64, Height: 32}, true, "bytesPerRow is required")
	bad(dst, TexelCopyBufferLayout{BytesPerRow: 32}, 64*32, Extent3D{Width: 64, Height: 32}, true, "bytesPerRow 32 <")
	bad(dst, TexelCopyBufferLayout{}, 64, Extent3D{Width: 64}, false, "COPY_SRC")
	bad(TexelCopyTextureInfo{}, TexelCopyBufferLayout{}, 64, Extent3D{Width: 64}, true, "nil texture")
	if err := checkTextureCopy("w", TexelCopyTextureInfo{Texture: tex(TextureDescriptor{Format: "nope", Usage: TextureUsageCopyDst})}, TexelCopyBufferLayout{}, 4, Extent3D{Width: 1}, true); err == nil {
		t.Fatal("unknown format accepted")
	}

	// Offset past the data, and overflowing offsets.
	bad(dst, TexelCopyBufferLayout{Offset: 1, BytesPerRow: 64}, 64*32, Extent3D{Width: 64, Height: 32}, true, "from offset 1")
	bad(dst, TexelCopyBufferLayout{Offset: math.MaxUint64, BytesPerRow: 64}, 64*32, Extent3D{Width: 64, Height: 32}, true, "from offset")

	// Multisampled.
	ms := tex(TextureDescriptor{Size: Extent3D{Width: 4, Height: 4}, Format: "rgba8unorm", Usage: TextureUsageCopyDst, SampleCount: 4})
	bad(TexelCopyTextureInfo{Texture: ms}, TexelCopyBufferLayout{BytesPerRow: 16}, 64, Extent3D{Width: 4, Height: 4}, true, "multisampled")

	// Arrays: rowsPerImage required, then sized.
	arr := tex(TextureDescriptor{Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 3}, Format: "rgba8unorm", Usage: TextureUsageCopyDst})
	ai := TexelCopyTextureInfo{Texture: arr}
	if err := checkTextureCopy("w", ai, TexelCopyBufferLayout{BytesPerRow: 16, RowsPerImage: 4}, 16*4*3, Extent3D{4, 4, 3}, true); err != nil {
		t.Fatal(err)
	}
	bad(ai, TexelCopyBufferLayout{BytesPerRow: 16}, 16*4*3, Extent3D{4, 4, 3}, true, "rowsPerImage are required")
	bad(ai, TexelCopyBufferLayout{BytesPerRow: 16, RowsPerImage: 2}, 16*4*3, Extent3D{4, 4, 3}, true, "rowsPerImage 2 <")
	bad(ai, TexelCopyBufferLayout{BytesPerRow: 16, RowsPerImage: 4}, 16*4*3-1, Extent3D{4, 4, 3}, true, "the data has")
	// bytesPerRow*rowsPerImage*layers overflowing 64 bits.
	big := tex(TextureDescriptor{Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: math.MaxUint32}, Format: "rgba8unorm", Usage: TextureUsageCopyDst})
	bad(TexelCopyTextureInfo{Texture: big}, TexelCopyBufferLayout{BytesPerRow: math.MaxUint32, RowsPerImage: math.MaxUint32}, math.MaxUint64, Extent3D{4, 4, math.MaxUint32}, true, "overflows")

	// Compressed: block alignment.
	bc := tex(TextureDescriptor{Size: Extent3D{Width: 8, Height: 8}, Format: "bc1-rgba-unorm", Usage: TextureUsageCopyDst})
	if err := checkTextureCopy("w", TexelCopyTextureInfo{Texture: bc}, TexelCopyBufferLayout{BytesPerRow: 16}, 32, Extent3D{8, 8, 1}, true); err != nil {
		t.Fatal(err)
	}
	bad(TexelCopyTextureInfo{Texture: bc, Origin: Origin3D{X: 2}}, TexelCopyBufferLayout{BytesPerRow: 16}, 32, Extent3D{4, 4, 1}, true, "multiples of the 4x4")
}

func TestCheckLinearOverflowOnLastRows(t *testing.T) {
	// imgs fits, adding the last rows overflows.
	l := TexelCopyBufferLayout{BytesPerRow: math.MaxUint32, RowsPerImage: math.MaxUint32}
	e := Extent3D{Width: 1, Height: math.MaxUint32, DepthOrArrayLayers: 2}
	wantBounds(t, checkLinear("w", l, math.MaxUint64, e, 1, 1, 1), "")
	// Depth 0 copies nothing.
	if err := checkLinear("w", TexelCopyBufferLayout{}, 0, Extent3D{Width: 1, Height: 1}, 1, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := checkLinear("w", TexelCopyBufferLayout{}, 0, Extent3D{}, 1, 1, 1); err != nil {
		t.Fatal(err)
	}
}

func TestCopyFootprintDepthStencil(t *testing.T) {
	mk := func(f TextureFormat) *Texture { return tex(TextureDescriptor{Format: f, Size: Extent3D{Width: 1}}) }
	for _, c := range []struct {
		f      TextureFormat
		aspect TextureAspect
		dst    bool
		want   uint32
		err    string
	}{
		{"stencil8", "", true, 1, ""},
		{"stencil8", TextureAspectStencilOnly, true, 1, ""},
		{"stencil8", TextureAspectDepthOnly, false, 0, "no depth-only aspect"},
		{"depth16unorm", TextureAspectAll, true, 2, ""},
		{"depth24plus", "", false, 0, "cannot be copied"},
		{"depth32float", "", false, 4, ""},
		{"depth32float", "", true, 0, "cannot be a copy destination"},
		{"depth24plus-stencil8", "", true, 0, "name the aspect"},
		{"depth24plus-stencil8", TextureAspectStencilOnly, true, 1, ""},
		{"depth24plus-stencil8", TextureAspectDepthOnly, false, 0, "cannot be copied"},
		{"depth32float-stencil8", TextureAspectDepthOnly, false, 4, ""},
		{"rgba8unorm", TextureAspectAll, true, 4, ""},
	} {
		n, _, _, err := copyFootprint("op", mk(c.f), c.aspect, c.dst)
		if c.err != "" {
			wantBounds(t, err, c.err)
			continue
		}
		if err != nil || n != c.want {
			t.Errorf("%s %q: %d, %v; want %d", c.f, c.aspect, n, err, c.want)
		}
	}
}

func TestFormatInfo(t *testing.T) {
	for f, fi := range formats {
		if fi.BlockWidth == 0 || fi.BlockHeight == 0 || fi.MemoryBytes == 0 {
			t.Errorf("%s: %+v", f, fi)
		}
		if !fi.Depth && !fi.Stencil && (fi.CopyBytes == 0 || !fi.CopyDst) {
			t.Errorf("%s: a color format must be copyable: %+v", f, fi)
		}
	}
	if fi, err := TextureFormatRGBA8Unorm.Info(); err != nil || fi.CopyBytes != 4 {
		t.Fatalf("rgba8unorm: %+v %v", fi, err)
	}
	if _, err := TextureFormat("rgba8").Info(); err == nil {
		t.Fatal("unknown format accepted")
	}
}
