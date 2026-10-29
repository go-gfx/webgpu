package webgpu

import "fmt"

// FormatInfo describes a texture format's texel blocks, after the WebGPU
// specification's texture format capability tables
// (https://www.w3.org/TR/webgpu/#texture-format-caps).
type FormatInfo struct {
	// BlockWidth and BlockHeight are the texel block's size in texels: 1x1
	// for every pixel-based format.
	BlockWidth, BlockHeight uint32
	// CopyBytes is the texel block copy footprint of the format's only
	// copyable aspect, 0 if no aspect can be copied from or to a buffer
	// without naming one (depth24plus, and the combined depth-stencil
	// formats, whose aspects are copied separately).
	CopyBytes uint32
	// MemoryBytes bounds what one texel block costs in GPU memory. The
	// specification calls this informative and gives ranges for some
	// depth-stencil formats; the upper bound is used, so quotas never
	// under-count.
	MemoryBytes uint32
	// Depth and Stencil say which aspects the format has.
	Depth, Stencil bool
	// CopyDst is false for formats whose depth aspect may not be the
	// destination of a texel copy (depth32float and friends).
	CopyDst bool
}

func plain(n uint32) FormatInfo {
	return FormatInfo{BlockWidth: 1, BlockHeight: 1, CopyBytes: n, MemoryBytes: n, CopyDst: true}
}

func block(w, h, n uint32) FormatInfo {
	return FormatInfo{BlockWidth: w, BlockHeight: h, CopyBytes: n, MemoryBytes: n, CopyDst: true}
}

var formats = map[TextureFormat]FormatInfo{
	"r8unorm": plain(1), "r8snorm": plain(1), "r8uint": plain(1), "r8sint": plain(1),

	"r16unorm": plain(2), "r16snorm": plain(2), "r16uint": plain(2), "r16sint": plain(2), "r16float": plain(2),
	"rg8unorm": plain(2), "rg8snorm": plain(2), "rg8uint": plain(2), "rg8sint": plain(2),

	"r32uint": plain(4), "r32sint": plain(4), "r32float": plain(4),
	"rg16unorm": plain(4), "rg16snorm": plain(4), "rg16uint": plain(4), "rg16sint": plain(4), "rg16float": plain(4),
	"rgba8unorm": plain(4), "rgba8unorm-srgb": plain(4), "rgba8snorm": plain(4), "rgba8uint": plain(4), "rgba8sint": plain(4),
	"bgra8unorm": plain(4), "bgra8unorm-srgb": plain(4),
	"rgb9e5ufloat": plain(4), "rgb10a2uint": plain(4), "rgb10a2unorm": plain(4), "rg11b10ufloat": plain(4),

	"rg32uint": plain(8), "rg32sint": plain(8), "rg32float": plain(8),
	"rgba16unorm": plain(8), "rgba16snorm": plain(8), "rgba16uint": plain(8), "rgba16sint": plain(8), "rgba16float": plain(8),

	"rgba32uint": plain(16), "rgba32sint": plain(16), "rgba32float": plain(16),

	// Depth and stencil (#depth-formats): memory cost is the upper bound of
	// the spec's range.
	"stencil8":              {BlockWidth: 1, BlockHeight: 1, CopyBytes: 1, MemoryBytes: 4, Stencil: true, CopyDst: true},
	"depth16unorm":          {BlockWidth: 1, BlockHeight: 1, CopyBytes: 2, MemoryBytes: 2, Depth: true, CopyDst: true},
	"depth24plus":           {BlockWidth: 1, BlockHeight: 1, CopyBytes: 0, MemoryBytes: 4, Depth: true},
	"depth24plus-stencil8":  {BlockWidth: 1, BlockHeight: 1, CopyBytes: 0, MemoryBytes: 8, Depth: true, Stencil: true},
	"depth32float":          {BlockWidth: 1, BlockHeight: 1, CopyBytes: 4, MemoryBytes: 4, Depth: true},
	"depth32float-stencil8": {BlockWidth: 1, BlockHeight: 1, CopyBytes: 0, MemoryBytes: 8, Depth: true, Stencil: true},

	"bc1-rgba-unorm": block(4, 4, 8), "bc1-rgba-unorm-srgb": block(4, 4, 8),
	"bc2-rgba-unorm": block(4, 4, 16), "bc2-rgba-unorm-srgb": block(4, 4, 16),
	"bc3-rgba-unorm": block(4, 4, 16), "bc3-rgba-unorm-srgb": block(4, 4, 16),
	"bc4-r-unorm": block(4, 4, 8), "bc4-r-snorm": block(4, 4, 8),
	"bc5-rg-unorm": block(4, 4, 16), "bc5-rg-snorm": block(4, 4, 16),
	"bc6h-rgb-ufloat": block(4, 4, 16), "bc6h-rgb-float": block(4, 4, 16),
	"bc7-rgba-unorm": block(4, 4, 16), "bc7-rgba-unorm-srgb": block(4, 4, 16),

	"etc2-rgb8unorm": block(4, 4, 8), "etc2-rgb8unorm-srgb": block(4, 4, 8),
	"etc2-rgb8a1unorm": block(4, 4, 8), "etc2-rgb8a1unorm-srgb": block(4, 4, 8),
	"etc2-rgba8unorm": block(4, 4, 16), "etc2-rgba8unorm-srgb": block(4, 4, 16),
	"eac-r11unorm": block(4, 4, 8), "eac-r11snorm": block(4, 4, 8),
	"eac-rg11unorm": block(4, 4, 16), "eac-rg11snorm": block(4, 4, 16),

	"astc-4x4-unorm": block(4, 4, 16), "astc-4x4-unorm-srgb": block(4, 4, 16),
	"astc-5x4-unorm": block(5, 4, 16), "astc-5x4-unorm-srgb": block(5, 4, 16),
	"astc-5x5-unorm": block(5, 5, 16), "astc-5x5-unorm-srgb": block(5, 5, 16),
	"astc-6x5-unorm": block(6, 5, 16), "astc-6x5-unorm-srgb": block(6, 5, 16),
	"astc-6x6-unorm": block(6, 6, 16), "astc-6x6-unorm-srgb": block(6, 6, 16),
	"astc-8x5-unorm": block(8, 5, 16), "astc-8x5-unorm-srgb": block(8, 5, 16),
	"astc-8x6-unorm": block(8, 6, 16), "astc-8x6-unorm-srgb": block(8, 6, 16),
	"astc-8x8-unorm": block(8, 8, 16), "astc-8x8-unorm-srgb": block(8, 8, 16),
	"astc-10x5-unorm": block(10, 5, 16), "astc-10x5-unorm-srgb": block(10, 5, 16),
	"astc-10x6-unorm": block(10, 6, 16), "astc-10x6-unorm-srgb": block(10, 6, 16),
	"astc-10x8-unorm": block(10, 8, 16), "astc-10x8-unorm-srgb": block(10, 8, 16),
	"astc-10x10-unorm": block(10, 10, 16), "astc-10x10-unorm-srgb": block(10, 10, 16),
	"astc-12x10-unorm": block(12, 10, 16), "astc-12x10-unorm-srgb": block(12, 10, 16),
	"astc-12x12-unorm": block(12, 12, 16), "astc-12x12-unorm-srgb": block(12, 12, 16),
}

// Info returns what the package knows of a format. An unknown format is an
// error: sizes are never guessed, since quotas and copy bounds depend on
// them.
func (f TextureFormat) Info() (FormatInfo, error) {
	fi, ok := formats[f]
	if !ok {
		return FormatInfo{}, fmt.Errorf("webgpu: unknown texture format %q", string(f))
	}
	return fi, nil
}
