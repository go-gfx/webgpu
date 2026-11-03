package wire

import (
	"errors"
	"fmt"
)

// DefaultMaxWorkgroups is WebGPU's default maxComputeWorkgroupsPerDimension,
// the dispatch bound of a device that was not asked for a raised limit.
const DefaultMaxWorkgroups = 65535

// ErrRange is wrapped by every CheckRanges refusal.
var ErrRange = errors.New("replay: range refused")

func rangeErr(format string, a ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrRange}, a...)...)
}

func u64(lo, hi uint32) uint64 { return uint64(lo) | uint64(hi)<<32 }

// inside reports whether [off, off+n) lies within size, without overflow.
func inside(off, n, size uint64) bool { return off <= size && n <= size-off }

// CheckRanges checks, before any replay, what Decode does not know: the
// buffer sizes. WriteBuffer and CopyBufferToBuffer ranges must lie inside
// their buffers, an indirect dispatch or draw must find its arguments
// inside the buffer, and a Dispatch may not exceed maxWorkgroups per
// dimension (0 is DefaultMaxWorkgroups). size returns the byte size of the
// buffer a replayer id names, and false for an id that is not a buffer.
// cmds are the commands of a stream already checked by Decode or Remap, in
// the replayer's ids. WebGPU validates the same rules later and loses the
// device; a host refuses them first, as a Go error.
func CheckRanges(cmds []Cmd, size func(id uint32) (uint64, bool), maxWorkgroups uint32) error {
	if maxWorkgroups == 0 {
		maxWorkgroups = DefaultMaxWorkgroups
	}
	buf := func(op Op, id uint32) (uint64, error) {
		n, ok := size(id)
		if !ok {
			return 0, rangeErr("opcode %d: object %d is not a buffer", op, id)
		}
		return n, nil
	}
	indirect := func(c Cmd, args uint64) error {
		n, err := buf(c.Op, c.Args[0])
		if err != nil {
			return err
		}
		if off := u64(c.Args[1], c.Args[2]); !inside(off, args, n) {
			return rangeErr("opcode %d: arguments [%d, +%d) outside buffer %d of %d bytes", c.Op, off, args, c.Args[0], n)
		}
		return nil
	}
	for _, c := range cmds {
		a := c.Args
		switch c.Op {
		case WriteBuffer: // id offset(2) dataOffset length
			n, err := buf(c.Op, a[0])
			if err != nil {
				return err
			}
			if off := u64(a[1], a[2]); !inside(off, uint64(a[4]), n) {
				return rangeErr("WriteBuffer [%d, +%d) outside buffer %d of %d bytes", off, a[4], a[0], n)
			}
		case CopyBufferToBuffer: // src srcOffset(2) dst dstOffset(2) size(2)
			sz := u64(a[6], a[7])
			for _, r := range [2][3]uint32{{a[0], a[1], a[2]}, {a[3], a[4], a[5]}} {
				n, err := buf(c.Op, r[0])
				if err != nil {
					return err
				}
				if off := u64(r[1], r[2]); !inside(off, sz, n) {
					return rangeErr("CopyBufferToBuffer [%d, +%d) outside buffer %d of %d bytes", off, sz, r[0], n)
				}
			}
		case Dispatch:
			for _, g := range a[:3] {
				if g > maxWorkgroups {
					return rangeErr("Dispatch %d,%d,%d above %d workgroups per dimension", a[0], a[1], a[2], maxWorkgroups)
				}
			}
		case DispatchIndirect:
			if err := indirect(c, 12); err != nil {
				return err
			}
		case DrawIndirect:
			if err := indirect(c, 16); err != nil {
				return err
			}
		case DrawIndexedIndirect:
			if err := indirect(c, 20); err != nil {
				return err
			}
		}
	}
	return nil
}
