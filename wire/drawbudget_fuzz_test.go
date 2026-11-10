//go:build !js

package wire

import (
	"encoding/binary"
	"errors"
	"testing"
)

// FuzzCheckDraws: never panics; a stream it accepts stays within both
// bounds, counted independently in floating point.
func FuzzCheckDraws(f *testing.F) {
	w := stream()
	b := make([]byte, 4*len(w))
	for i, v := range w {
		binary.LittleEndian.PutUint32(b[4*i:], v)
	}
	f.Add(b, uint32(4), uint64(1<<20), uint64(1<<24))
	f.Fuzz(func(t *testing.T, raw []byte, ndata uint32, perDraw, perStream uint64) {
		cmds, err := Decode(words(raw), ndata, func(id uint32) bool { return true })
		if err != nil {
			return
		}
		total, _, err := CheckDraws(cmds, perDraw, perStream)
		if err != nil {
			if !errors.Is(err, ErrDrawBudget) || !errors.Is(err, ErrRange) {
				t.Fatalf("refusal without ErrDrawBudget: %v", err)
			}
			return
		}
		var sum float64
		for _, c := range cmds {
			if c.Op != Draw && c.Op != DrawIndexed {
				continue
			}
			n := float64(c.Args[0]) * float64(c.Args[1])
			if perDraw != 0 && n > float64(perDraw)*1.0000001 {
				t.Fatalf("accepted a draw of %g above %d", n, perDraw)
			}
			sum += n
		}
		if perStream != 0 && sum > float64(perStream)*1.0000001 {
			t.Fatalf("accepted a stream of %g above %d (total %d)", sum, perStream, total)
		}
	})
}
