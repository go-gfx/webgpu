package wire

import (
	"errors"
	"math"
	"testing"
)

func TestCheckInvocations(t *testing.T) {
	d := func(x, y, z uint32) Cmd { return Cmd{Dispatch, []uint32{x, y, z}} }
	ind := Cmd{DispatchIndirect, []uint32{1, 0, 0}}
	for _, tc := range []struct {
		name            string
		cmds            []Cmd
		perDisp, perStr uint64
		total           uint64
		bad             bool
	}{
		{"empty", nil, 10, 10, 0, false},
		{"exact per dispatch", []Cmd{d(2, 5, 1)}, 640, 0, 640, false},
		{"one over per dispatch", []Cmd{d(2, 5, 1)}, 639, 0, 0, true},
		{"stream sum", []Cmd{d(1, 1, 1), d(1, 1, 1)}, 0, 128, 128, false},
		{"stream sum over", []Cmd{d(1, 1, 1), d(1, 1, 1), d(1, 1, 1)}, 0, 128, 128 * 2, true},
		{"no bound", []Cmd{d(65535, 65535, 65535)}, 0, 0, math.MaxUint64, false},
		{"the total saturates across dispatches", []Cmd{d(math.MaxUint32, math.MaxUint32, math.MaxUint32), d(math.MaxUint32, math.MaxUint32, math.MaxUint32)}, 0, 0, math.MaxUint64, false},
		{"saturates, never wraps", []Cmd{d(math.MaxUint32, math.MaxUint32, math.MaxUint32)}, 1 << 40, 0, 0, true},
		{"zero count is no work", []Cmd{d(0, math.MaxUint32, math.MaxUint32)}, 1, 1, 0, false},
		{"indirect is not counted", []Cmd{ind}, 1, 1, 0, false},
	} {
		total, err := CheckInvocations(tc.cmds, 64, tc.perDisp, tc.perStr)
		if (err != nil) != tc.bad {
			t.Errorf("%s: error %v, want refusal %v", tc.name, err, tc.bad)
		}
		if err != nil && (!errors.Is(err, ErrBudget) || !errors.Is(err, ErrRange)) {
			t.Errorf("%s: %v does not wrap ErrBudget and ErrRange", tc.name, err)
		}
		if err == nil && tc.name != "no bound" && total != tc.total {
			t.Errorf("%s: total %d, want %d", tc.name, total, tc.total)
		}
	}
	if got := DispatchInvocations(65535, 65535, 65535, 256); got != math.MaxUint64 && got != 65535*65535*65535*256 {
		t.Errorf("DispatchInvocations = %d", got)
	}
}
