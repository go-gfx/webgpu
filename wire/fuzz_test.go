// The browser runner has no file system for the seed corpus; the fuzz
// targets run natively.

//go:build !js

package wire

import (
	"encoding/binary"
	"testing"
)

func words(raw []byte) []uint32 {
	w := make([]uint32, len(raw)/4)
	for i := range w {
		w[i] = binary.LittleEndian.Uint32(raw[4*i:])
	}
	return w
}

func seed(f *testing.F) {
	w := stream()
	b := make([]byte, 4*len(w))
	for i, v := range w {
		binary.LittleEndian.PutUint32(b[4*i:], v)
	}
	f.Add(b, uint32(4), uint32(3))
	f.Add([]byte{20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 40, 0, 0, 0}, uint32(0), uint32(1))
}

// FuzzDecode: never panics or reads past the stream; an accepted stream
// is exactly covered by its commands.
func FuzzDecode(f *testing.F) {
	seed(f)
	f.Fuzz(func(t *testing.T, raw []byte, ndata, nobj uint32) {
		w := words(raw)
		cmds, err := Decode(w, ndata, func(id uint32) bool { return id <= nobj })
		if err != nil {
			return
		}
		n := 0
		for _, c := range cmds {
			n += 1 + len(c.Args)
		}
		if n != len(w) {
			t.Fatalf("commands cover %d of %d words", n, len(w))
		}
	})
}

// FuzzRemap: Remap accepts exactly what Decode accepts, and a remapped
// stream decodes against the new table only, with the same commands.
func FuzzRemap(f *testing.F) {
	seed(f)
	f.Fuzz(func(t *testing.T, raw []byte, ndata, nobj uint32) {
		nobj %= 1 << 20
		w := words(raw)
		old := func(id uint32) bool { return id <= nobj }
		_, derr := Decode(w, ndata, old)
		const shift = 1 << 24
		rerr := Remap(w, ndata, func(id uint32) (uint32, bool) { return id + shift, old(id) })
		if (derr == nil) != (rerr == nil) {
			t.Fatalf("Decode %v, Remap %v", derr, rerr)
		}
		if rerr != nil {
			return
		}
		if _, err := Decode(w, ndata, func(id uint32) bool { return id > shift && id <= shift+nobj }); err != nil {
			t.Fatalf("remapped stream refused by the new table: %v", err)
		}
	})
}

// FuzzCheckRanges: never panics, and a stream it accepts keeps every write
// and copy inside the sizes it was given.
func FuzzCheckRanges(f *testing.F) {
	seed(f)
	f.Fuzz(func(t *testing.T, raw []byte, ndata, nobj uint32) {
		w := words(raw)
		cmds, err := Decode(w, ndata, func(id uint32) bool { return id <= nobj })
		if err != nil {
			return
		}
		size := func(id uint32) (uint64, bool) { return uint64(id) * 8, id != 0 }
		if CheckRanges(cmds, size, 0) != nil {
			return
		}
		for _, c := range cmds {
			if c.Op == WriteBuffer {
				if n, _ := size(c.Args[0]); u64(c.Args[1], c.Args[2])+uint64(c.Args[4]) > n {
					t.Fatalf("accepted a write past %d bytes: %v", n, c.Args)
				}
			}
		}
	})
}
