package wire

import (
	"errors"
	"math"
	"testing"
)

func TestCheckDraws(t *testing.T) {
	d := func(v, i uint32) Cmd { return Cmd{Draw, []uint32{v, i, 0, 0}} }
	di := func(n, i uint32) Cmd { return Cmd{DrawIndexed, []uint32{n, i, 0, 0, 0}} }
	ind := []Cmd{{DrawIndirect, []uint32{1, 0, 0}}, {DrawIndexedIndirect, []uint32{1, 0, 0}}}
	huge := d(math.MaxUint32, math.MaxUint32)
	for _, tc := range []struct {
		name            string
		cmds            []Cmd
		perDraw, perStr uint64
		total           uint64 // checked when the stream is accepted
		bad             bool
	}{
		{"empty", nil, 10, 10, 0, false},
		{"exact per draw", []Cmd{d(30, 4)}, 120, 0, 120, false},
		{"one over per draw", []Cmd{d(30, 4)}, 119, 0, 0, true},
		{"indexed is counted by its index count", []Cmd{di(6, 100)}, 600, 0, 600, false},
		{"indexed one over", []Cmd{di(6, 100)}, 599, 0, 0, true},
		{"stream sum", []Cmd{d(10, 1), di(10, 1)}, 0, 20, 20, false},
		{"stream sum over", []Cmd{d(10, 1), d(10, 1), di(1, 1)}, 0, 20, 0, true},
		{"each draw within, together over", []Cmd{d(60, 1), d(60, 1)}, 100, 100, 0, true},
		{"the total saturates across draws", []Cmd{huge, huge}, 0, 0, math.MaxUint64, false},
		{"a saturated total is refused, not wrapped", []Cmd{huge, huge}, 0, math.MaxUint64 - 1, 0, true},
		{"one huge draw is above any real quota", []Cmd{huge}, 1 << 40, 0, 0, true},
		{"zero instances is no work", []Cmd{d(math.MaxUint32, 0)}, 1, 1, 0, false},
		{"zero vertices is no work", []Cmd{di(0, math.MaxUint32)}, 1, 1, 0, false},
		{"indirect is not counted", ind, 1, 1, 0, false},
		{"other commands are not counted", []Cmd{{Dispatch, []uint32{math.MaxUint32, math.MaxUint32, math.MaxUint32}}}, 1, 1, 0, false},
	} {
		total, largest, err := CheckDraws(tc.cmds, tc.perDraw, tc.perStr)
		if (err != nil) != tc.bad {
			t.Errorf("%s: error %v, want refusal %v", tc.name, err, tc.bad)
		}
		if err != nil && (!errors.Is(err, ErrDrawBudget) || !errors.Is(err, ErrRange)) {
			t.Errorf("%s: %v does not wrap ErrDrawBudget and ErrRange", tc.name, err)
		}
		if err == nil && total != tc.total {
			t.Errorf("%s: total %d, want %d", tc.name, total, tc.total)
		}
		if err == nil && largest > total {
			t.Errorf("%s: largest %d above total %d", tc.name, largest, total)
		}
	}
	if got := DrawVertices(math.MaxUint32, math.MaxUint32); got != uint64(math.MaxUint32)*uint64(math.MaxUint32) {
		t.Errorf("DrawVertices = %d", got)
	}
}
