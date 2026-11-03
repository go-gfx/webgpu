//go:build !js

package wire

import (
	"encoding/binary"
	"errors"
	"testing"
)

// FuzzCheckInvocations: never panics; a stream it accepts stays within both
// bounds, counted independently in floating point.
func FuzzCheckInvocations(f *testing.F) {
	w := stream()
	b := make([]byte, 4*len(w))
	for i, v := range w {
		binary.LittleEndian.PutUint32(b[4*i:], v)
	}
	f.Add(b, uint32(4), uint32(64), uint64(1<<20), uint64(1<<24))
	f.Fuzz(func(t *testing.T, raw []byte, ndata, wg uint32, perDisp, perStream uint64) {
		cmds, err := Decode(words(raw), ndata, func(id uint32) bool { return true })
		if err != nil {
			return
		}
		total, err := CheckInvocations(cmds, wg, perDisp, perStream)
		if err != nil {
			if !errors.Is(err, ErrBudget) || !errors.Is(err, ErrRange) {
				t.Fatalf("refusal without ErrBudget: %v", err)
			}
			return
		}
		var sum float64
		for _, c := range cmds {
			if c.Op != Dispatch {
				continue
			}
			n := float64(c.Args[0]) * float64(c.Args[1]) * float64(c.Args[2]) * float64(wg)
			if perDisp != 0 && n > float64(perDisp)*1.0000001 {
				t.Fatalf("accepted a dispatch of %g above %d", n, perDisp)
			}
			sum += n
		}
		if perStream != 0 && sum > float64(perStream)*1.0000001 {
			t.Fatalf("accepted a stream of %g above %d (total %d)", sum, perStream, total)
		}
	})
}
