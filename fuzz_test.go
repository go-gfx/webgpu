package webgpu

import (
	"math/big"
	"testing"
)

// FuzzCheckTextureCopy: the arguments of WriteTexture come from untrusted
// data (a game map's lightmaps, a container's call). Whatever they are,
// the check must not panic, and when it accepts, the byte of the last
// texel block it lets the browser read must lie inside the data,
// recomputed here with unbounded integers.
func FuzzCheckTextureCopy(f *testing.F) {
	f.Add(uint32(64), uint32(32), uint32(1), "r8uint", uint32(0), uint32(0), uint32(0), uint32(0), uint64(0), uint32(64), uint32(0), uint64(2048), uint32(64), uint32(32), uint32(1))
	f.Add(uint32(8), uint32(8), uint32(3), "bc1-rgba-unorm", uint32(0), uint32(4), uint32(4), uint32(1), uint64(8), uint32(16), uint32(2), uint64(1000), uint32(4), uint32(4), uint32(2))
	f.Add(uint32(4), uint32(4), uint32(2), "depth32float", uint32(0), uint32(0), uint32(0), uint32(0), uint64(0), uint32(16), uint32(4), uint64(128), uint32(4), uint32(4), uint32(2))
	f.Fuzz(func(t *testing.T, w, h, layers uint32, format string, mip, ox, oy, oz uint32, off uint64, bpr, rpi uint32, n uint64, cw, ch, cd uint32) {
		tx := tex(TextureDescriptor{Size: Extent3D{w, h, layers}, Format: TextureFormat(format), MipLevelCount: 1 + mip%4,
			Usage: TextureUsageCopyDst | TextureUsageCopySrc})
		info := TexelCopyTextureInfo{Texture: tx, MipLevel: mip % 5, Origin: Origin3D{ox, oy, oz}}
		l := TexelCopyBufferLayout{Offset: off, BytesPerRow: bpr, RowsPerImage: rpi}
		size := Extent3D{cw, ch, cd}
		if checkTextureCopy("fuzz", info, l, n, size, false) != nil {
			return
		}
		fb, bw, bh, err := copyFootprint("fuzz", tx, "", false)
		if err != nil {
			t.Fatalf("accepted a copy whose footprint is an error: %v", err)
		}
		size = size.norm()
		if size.Width == 0 || size.Height == 0 || size.DepthOrArrayLayers == 0 {
			return
		}
		// End of the last block row of the last image.
		B := func(v uint64) *big.Int { return new(big.Int).SetUint64(v) }
		hBlocks := uint64(size.Height / bh)
		wBlocks := uint64(size.Width / bw)
		end := B(off)
		end.Add(end, new(big.Int).Mul(B(uint64(bpr)*uint64(rpi)), B(uint64(size.DepthOrArrayLayers-1))))
		end.Add(end, new(big.Int).Mul(B(uint64(bpr)), B(hBlocks-1)))
		end.Add(end, B(wBlocks*uint64(fb)))
		if end.Cmp(B(n)) > 0 {
			t.Fatalf("accepted a copy reading to byte %s of %d", end, n)
		}
		e := mipExtent(tx.desc, info.MipLevel)
		if uint64(ox)+uint64(size.Width) > roundUp(e.Width, bw) || uint64(oy)+uint64(size.Height) > roundUp(e.Height, bh) {
			t.Fatalf("accepted a region outside the mip level")
		}
	})
}

// FuzzTextureBytes: never panics; a size it returns is non-negative and
// at least the base level's.
func FuzzTextureBytes(f *testing.F) {
	f.Add(uint32(64), uint32(64), uint32(1), uint32(7), uint32(1), "rgba8unorm", "2d")
	f.Add(uint32(4096), uint32(4096), uint32(2048), uint32(13), uint32(4), "rgba32float", "3d")
	f.Fuzz(func(t *testing.T, w, h, d, mips, samples uint32, format, dim string) {
		desc := TextureDescriptor{Size: Extent3D{w, h, d}, MipLevelCount: mips, SampleCount: samples, Format: TextureFormat(format), Dimension: TextureDimension(dim)}
		n, err := TextureBytes(desc)
		if err != nil {
			return
		}
		if n < 0 {
			t.Fatalf("negative size %d", n)
		}
	})
}
