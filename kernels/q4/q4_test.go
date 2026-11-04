package q4

import (
	"strings"
	"testing"

	"github.com/go-gfx/wgsl"
)

func specs() map[string]Spec {
	base := Spec{WG: 64, TPR: 8, MaxShared: 16384, MaxGroups: 65535, N: 576, Rows: 576, Eps: 1e-5}
	norm := base
	norm.Norm, norm.NW = true, 576*18
	q41 := base
	q41.N, q41.Q41, q41.Mode = 1536, true, Acc
	acc := base
	acc.Mode = Acc
	big := base
	big.N, big.MaxShared = 8192, 8192 // tiled
	custom := base
	custom.Mode = Custom
	custom.Hooks = Hooks{NoInput: true, Load: "vec4<f32>(0.0)", RowMap: RowPlain, Epi: EpiStore, Bindings: []Binding{{"y", "array<f32>", true}}}
	return map[string]Spec{"store": base, "norm": norm, "q41acc": q41, "acc": acc, "tiled": big, "custom": custom}
}

// Every placeholder is substituted and the WGSL parses, at the workgroup
// sizes an adapter can give.
func TestSourceParses(t *testing.T) {
	for _, wg := range []int{64, 32, 16} {
		for name, s := range specs() {
			s.WG = wg
			if s.TPR > wg {
				s.TPR = wg
			}
			src, access, gx, gy := Source(s, 0)
			if i := strings.Index(src, "{{"); i >= 0 {
				t.Fatalf("%s wg %d: unsubstituted %q", name, wg, src[i:min(len(src), i+16)])
			}
			if _, err := wgsl.Parse(src); err != nil {
				t.Errorf("%s wg %d: %v", name, wg, err)
			}
			if gx*gy < s.Workgroups() {
				t.Errorf("%s: grid %dx%d too small for %d workgroups", name, gx, gy, s.Workgroups())
			}
			if got := strings.Count(src, "@binding("); got != len(access) {
				t.Errorf("%s: %d bindings, access %q", name, got, access)
			}
		}
	}
}

// The staged tile and the reduction array never exceed the workgroup memory
// and the tiles cover the input.
func TestTile(t *testing.T) {
	for _, shared := range []int{16384, 32768, 8192} {
		for _, n := range []int{576, 1536, 4864, 8192} {
			s := Spec{WG: 64, TPR: 8, MaxShared: shared, N: n, Rows: 64}
			t4, nt, tb := s.Tile()
			if used := t4*16 + 64*4; used > shared {
				t.Errorf("shared %d n %d: tile uses %d bytes", shared, n, used)
			}
			if nt*tb*32 < n {
				t.Errorf("shared %d n %d: %d tiles of %d blocks do not cover", shared, n, nt, tb)
			}
		}
	}
}

// A 2-D grid when the workgroups exceed the per-dimension limit.
func TestGrid2D(t *testing.T) {
	s := Spec{WG: 64, TPR: 8, MaxShared: 16384, MaxGroups: 100, N: 64, Rows: 8 * 250}
	_, _, gx, gy := Source(s, 0)
	if gx != 100 || gy != 3 {
		t.Fatalf("grid %dx%d, want 100x3", gx, gy)
	}
}

// With workgroup memory below one block the tile is one block (the caller's
// limits then make the kernel invalid on the device, never out of range here).
func TestTileMinimum(t *testing.T) {
	s := Spec{WG: 64, TPR: 8, MaxShared: 512, N: 1024, Rows: 8}
	if t4, nt, tb := s.Tile(); t4 != 8 || tb != 1 || nt != 32 {
		t.Fatalf("tile %d vec4, %d tiles of %d blocks", t4, nt, tb)
	}
}
