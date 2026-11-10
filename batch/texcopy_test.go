//go:build !js

package batch

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
)

// texSim is a Backend and Reader that executes the memory commands of a
// stream on byte slices: WriteTexture, CopyTextureToBuffer, CopyBufferToBuffer
// and WriteBuffer, for RGBA8 textures of one mip level. It stands in for the
// GPU so that a guest's write, copy and read-back can be compared byte for
// byte; the range rules are the Context's, not its own.
type texSim struct {
	next  uint32
	texs  map[uint32]texMem
	bufs  map[uint32][]byte
	plays int
}

type texMem struct {
	w, h uint32
	px   []byte
}

func newTexSim() *texSim { return &texSim{texs: map[uint32]texMem{}, bufs: map[uint32][]byte{}} }

func (s *texSim) Create(kind uint32, desc []byte) (uint32, error) {
	s.next++
	switch wire.Kind(kind) {
	case wire.KindBuffer:
		d, err := webgpu.DecodeDescriptor(wire.KindBuffer, desc, nil)
		if err != nil {
			return 0, err
		}
		s.bufs[s.next] = make([]byte, d.(webgpu.BufferDescriptor).Size)
	case wire.KindTexture:
		d, err := webgpu.DecodeDescriptor(wire.KindTexture, desc, nil)
		if err != nil {
			return 0, err
		}
		sz := d.(webgpu.TextureDescriptor).Size
		s.texs[s.next] = texMem{sz.Width, sz.Height, make([]byte, 4*sz.Width*sz.Height)}
	}
	return s.next, nil
}

func (s *texSim) Destroy(id uint32) { delete(s.texs, id); delete(s.bufs, id) }
func (s *texSim) Close()            {}

func (s *texSim) Replay(words []uint32, data []byte) error {
	s.plays++
	cmds, err := wire.Decode(words, uint32(len(data)), func(uint32) bool { return true })
	if err != nil {
		return err
	}
	for _, c := range cmds {
		a := c.Args
		switch c.Op {
		case wire.WriteTexture:
			t := s.texs[a[0]]
			src := data[a[6] : a[6]+a[7]]
			off, bpr := uint64(a[8])|uint64(a[9])<<32, uint64(a[10])
			for y := uint32(0); y < a[13]; y++ {
				row := src[off+uint64(y)*bpr:]
				copy(t.px[4*((a[3]+y)*t.w+a[2]):], row[:4*a[12]])
			}
		case wire.CopyTextureToBuffer:
			tc, _ := wire.TextureCopyOf(c)
			t, b := s.texs[tc.Texture], s.bufs[tc.Buffer]
			for y := uint32(0); y < tc.Height; y++ {
				dst := tc.Offset + uint64(y)*uint64(tc.BytesPerRow)
				copy(b[dst:], t.px[4*((tc.OriginY+y)*t.w+tc.OriginX):][:4*tc.Width])
			}
		}
	}
	return nil
}

func (s *texSim) Read(id uint32, offset uint64, dst []byte) error {
	copy(dst, s.bufs[id][offset:])
	return nil
}

// TestTextureRoundTrip: a guest writes a texture with WriteTexture, copies
// it back into a MAP_READ buffer with CopyTextureToBuffer (rows padded to 256
// bytes) and reads the buffer through the stream; the bytes equal what it
// wrote, in one submit, with the region and an offset honoured.
func TestTextureRoundTrip(t *testing.T) {
	sim := newTexSim()
	d, c := guest(t, Config{}, sim)
	const w, h = 8, 4
	tex, err := d.CreateTexture(webgpu.TextureDescriptor{Size: webgpu.Extent3D{Width: w, Height: h},
		Format: webgpu.TextureFormatRGBA8Unorm, Usage: webgpu.TextureUsageCopyDst | webgpu.TextureUsageCopySrc})
	if err != nil {
		t.Fatal(err)
	}
	px := make([]byte, 4*w*h)
	for i := range px {
		px[i] = byte(i*7 + 3)
	}
	out := buffer(t, d, 2048, webgpu.BufferUsageMapRead|webgpu.BufferUsageCopyDst)

	r := d.NewRecorder()
	r.WriteTexture(webgpu.TexelCopyTextureInfo{Texture: tex}, px, webgpu.TexelCopyBufferLayout{BytesPerRow: 4 * w}, webgpu.Extent3D{Width: w, Height: h})
	r.CopyTextureToBuffer(webgpu.TexelCopyTextureInfo{Texture: tex}, out, webgpu.TexelCopyBufferLayout{BytesPerRow: 256}, webgpu.Extent3D{Width: w, Height: h})
	// A sub-rectangle (2x2 at (3,1)) to offset 1024.
	r.CopyTextureToBuffer(webgpu.TexelCopyTextureInfo{Texture: tex, Origin: webgpu.Origin3D{X: 3, Y: 1}}, out,
		webgpu.TexelCopyBufferLayout{Offset: 1024, BytesPerRow: 256}, webgpu.Extent3D{Width: 2, Height: 2})
	whole := r.ReadBuffer(out, 0, 4*256)
	part := r.ReadBuffer(out, 1024, 264)
	if _, err := r.Submit(); err != nil {
		t.Fatal(err)
	}
	got, err := whole.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	for y := 0; y < h; y++ {
		if !bytes.Equal(got[256*y:256*y+4*w], px[4*w*y:4*w*(y+1)]) {
			t.Errorf("row %d: %v, want %v", y, got[256*y:256*y+4*w], px[4*w*y:4*w*(y+1)])
		}
		if !bytes.Equal(got[256*y+4*w:256*(y+1)], make([]byte, 256-4*w)) {
			t.Errorf("row %d: padding was written", y)
		}
	}
	pg, err := part.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 2; y++ {
		want := px[4*(w*(1+y)+3) : 4*(w*(1+y)+3)+8]
		if !bytes.Equal(pg[256*y:256*y+8], want) {
			t.Errorf("sub-rectangle row %d: %v, want %v", y, pg[256*y:256*y+8], want)
		}
	}
	if c.Lost() != nil || sim.plays != 1 {
		t.Fatalf("lost %v, %d replays", c.Lost(), sim.plays)
	}
}

// TestTextureCopyRefusals: each out-of-range copy is an error at Stream that
// wraps wire.ErrRange (or the decoder's alignment check), loses the context
// and never reaches the backend.
func TestTextureCopyRefusals(t *testing.T) {
	mk := func(c *Context, id uint32, d any) {
		t.Helper()
		kind, desc, err := webgpu.EncodeDescriptor(d)
		if err != nil {
			t.Fatal(err)
		}
		if err := c.Create(id, uint32(kind), desc); err != nil {
			t.Fatal(err)
		}
	}
	setup := func() (*Context, *texSim) {
		sim := newTexSim()
		c := New(sim, Config{})
		mk(c, 1, webgpu.TextureDescriptor{Size: webgpu.Extent3D{Width: 8, Height: 4}, MipLevelCount: 2, Format: webgpu.TextureFormatRGBA8Unorm, Usage: webgpu.TextureUsageCopySrc})
		mk(c, 2, webgpu.BufferDescriptor{Size: 1024, Usage: webgpu.BufferUsageCopyDst})
		mk(c, 3, webgpu.BufferDescriptor{Size: 1024, Usage: webgpu.BufferUsageMapRead}) // no COPY_DST
		mk(c, 4, webgpu.SamplerDescriptor{})
		mk(c, 5, webgpu.TextureDescriptor{Size: webgpu.Extent3D{Width: 8, Height: 4}, Format: webgpu.TextureFormatRGBA8Unorm, Usage: webgpu.TextureUsageCopyDst}) // no COPY_SRC
		mk(c, 6, webgpu.TextureDescriptor{Size: webgpu.Extent3D{Width: 8, Height: 4, DepthOrArrayLayers: 3}, Format: webgpu.TextureFormatRGBA8Unorm, Usage: webgpu.TextureUsageCopySrc})
		return c, sim
	}
	const max32 = 1<<32 - 1
	// texture mip x y z aspect buffer offLo offHi bytesPerRow rowsPerImage w h d
	cp := func(a ...uint32) []uint32 {
		return append([]uint32{uint32(wire.CopyTextureToBuffer)}, append(a, uint32(wire.Submit))...)
	}
	for _, tc := range []struct {
		name  string
		words []uint32
		want  string
	}{
		{"texture is a sampler", cp(4, 0, 0, 0, 0, 0, 2, 0, 0, 256, 0, 8, 4, 1), "not a texture"},
		{"buffer is a texture", cp(1, 0, 0, 0, 0, 0, 5, 0, 0, 256, 0, 8, 4, 1), "not a buffer"},
		{"mip 1 is smaller than the whole texture", cp(1, 1, 0, 0, 0, 0, 2, 0, 0, 256, 0, 8, 4, 1), "outside mip level 1"},
		{"mip level out of range", cp(1, 2, 0, 0, 0, 0, 2, 0, 0, 256, 0, 1, 1, 1), "mip level 2"},
		{"region past the width", cp(1, 0, 4, 0, 0, 0, 2, 0, 0, 256, 0, 8, 4, 1), "outside mip level 0"},
		{"origin plus size overflows 32 bits", cp(1, 0, max32, 0, 0, 0, 2, 0, 0, 256, 0, 8, 4, 1), "outside mip level 0"},
		{"size overflows 32 bits", cp(1, 0, 1, 0, 0, 0, 2, 0, 0, 256, 0, max32, 4, 1), "outside mip level 0"},
		{"layers past the texture", cp(1, 0, 0, 0, 0, 0, 2, 0, 0, 256, 4, 8, 4, 2), "outside mip level 0"},
		{"layout past the buffer", cp(1, 0, 0, 0, 0, 0, 2, 512, 0, 256, 0, 8, 4, 1), "needs"},
		{"offset past 4 GiB", cp(1, 0, 0, 0, 0, 0, 2, 0, 1, 256, 0, 8, 4, 1), "needs"},
		{"offset near 2^64", cp(1, 0, 0, 0, 0, 0, 2, max32, max32, 256, 0, 8, 4, 1), "needs"},
		{"layout overflows 64 bits", cp(6, 0, 0, 0, 0, 0, 2, 0, 0, max32-255, max32, 8, 4, 3), "overflows"},
		{"bytesPerRow 0 with four rows", cp(1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 8, 4, 1), "bytesPerRow is required"},
		{"bytesPerRow 128", cp(1, 0, 0, 0, 0, 0, 2, 0, 0, 128, 0, 8, 4, 1), "multiple of 256"},
		{"buffer without COPY_DST", cp(1, 0, 0, 0, 0, 0, 3, 0, 0, 256, 0, 8, 4, 1), "COPY_DST"},
		{"texture without COPY_SRC", cp(5, 0, 0, 0, 0, 0, 2, 0, 0, 256, 0, 8, 4, 1), "COPY_SRC"},
	} {
		c, sim := setup()
		err := c.Stream(tc.words, nil, nil)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want %q", tc.name, err, tc.want)
			continue
		}
		if !errors.Is(err, wire.ErrRange) && !strings.Contains(tc.want, "multiple of 256") {
			t.Errorf("%s: %v does not wrap wire.ErrRange", tc.name, err)
		}
		if c.Lost() == nil || sim.plays != 0 {
			t.Errorf("%s: lost %v, %d replays", tc.name, c.Lost(), sim.plays)
		}
	}
	// The same table's accepted neighbours reach the backend.
	for name, words := range map[string][]uint32{
		"ending at the buffer's end": cp(1, 0, 0, 0, 0, 0, 2, 224, 0, 256, 0, 8, 4, 1),
		"mip 1":                      cp(1, 1, 0, 0, 0, 0, 2, 0, 0, 256, 0, 4, 2, 1),
	} {
		c, sim := setup()
		if err := c.Stream(words, nil, nil); err != nil || sim.plays != 1 {
			t.Errorf("%s: %v, %d replays", name, err, sim.plays)
		}
	}
}
