package wire

import (
	"errors"
	"testing"
)

func sizes(m map[uint32]uint64) func(uint32) (uint64, bool) {
	return func(id uint32) (uint64, bool) { n, ok := m[id]; return n, ok }
}

func TestCheckRanges(t *testing.T) {
	sz := sizes(map[uint32]uint64{1: 64, 2: 32})
	const max32 = ^uint32(0)
	tests := []struct {
		name string
		cmd  Cmd
		ok   bool
	}{
		{"write inside", Cmd{WriteBuffer, []uint32{1, 0, 0, 0, 64}}, true},
		{"write past end", Cmd{WriteBuffer, []uint32{1, 4, 0, 0, 64}}, false},
		{"write offset past 4 GiB", Cmd{WriteBuffer, []uint32{1, 0, 1, 0, 4}}, false},
		{"write offset overflow", Cmd{WriteBuffer, []uint32{1, max32, max32, 0, max32}}, false},
		{"write to a texture", Cmd{WriteBuffer, []uint32{9, 0, 0, 0, 4}}, false},
		{"copy inside", Cmd{CopyBufferToBuffer, []uint32{1, 0, 0, 2, 0, 0, 32, 0}}, true},
		{"copy dst past end", Cmd{CopyBufferToBuffer, []uint32{1, 0, 0, 2, 4, 0, 32, 0}}, false},
		{"copy src past end", Cmd{CopyBufferToBuffer, []uint32{2, 0, 0, 1, 0, 0, 33, 0}}, false},
		{"copy size above 4 GiB", Cmd{CopyBufferToBuffer, []uint32{1, 0, 0, 2, 0, 0, 0, 1}}, false},
		{"copy of a non-buffer", Cmd{CopyBufferToBuffer, []uint32{9, 0, 0, 2, 0, 0, 4, 0}}, false},
		{"dispatch at the limit", Cmd{Dispatch, []uint32{65535, 1, 1}}, true},
		{"dispatch x over", Cmd{Dispatch, []uint32{65536, 1, 1}}, false},
		{"dispatch z over", Cmd{Dispatch, []uint32{1, 1, 1 << 31}}, false},
		{"indirect dispatch inside", Cmd{DispatchIndirect, []uint32{1, 52, 0}}, true},
		{"indirect dispatch past end", Cmd{DispatchIndirect, []uint32{1, 53, 0}}, false},
		{"indirect draw past end", Cmd{DrawIndirect, []uint32{2, 20, 0}}, false},
		{"indirect indexed draw inside", Cmd{DrawIndexedIndirect, []uint32{2, 12, 0}}, true},
		{"indirect indexed draw past end", Cmd{DrawIndexedIndirect, []uint32{2, 16, 0}}, false},
	}
	for _, tc := range tests {
		err := CheckRanges([]Cmd{tc.cmd}, sz, 0)
		if tc.ok && err != nil {
			t.Errorf("%s: %v", tc.name, err)
		}
		if !tc.ok && !errors.Is(err, ErrRange) {
			t.Errorf("%s: got %v, want ErrRange", tc.name, err)
		}
	}
	if err := CheckRanges([]Cmd{{Dispatch, []uint32{10, 1, 1}}}, sz, 8); !errors.Is(err, ErrRange) {
		t.Errorf("a granted limit of 8: %v", err)
	}
}

// RemapCmds must give back exactly the commands Decode does, with the ids
// translated.
func TestRemapCmds(t *testing.T) {
	w := []uint32{uint32(WriteBuffer), 7, 0, 0, 0, 4, uint32(ReadBuffer), 7, 0, 0, 4, 0}
	cmds, err := RemapCmds(w, 4, func(id uint32) (uint32, bool) { return id + 100, id == 7 })
	if err != nil || len(cmds) != 2 || cmds[0].Args[0] != 107 || cmds[1].Args[0] != 107 {
		t.Fatalf("%v %+v", err, cmds)
	}
	if _, ok := ReadOf(cmds[1]); !ok {
		t.Fatal("ReadOf")
	}
}
