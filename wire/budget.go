package wire

import (
	"fmt"
	"math/bits"
)

// ErrBudget is wrapped by every CheckInvocations refusal. It also wraps
// ErrRange, so a caller that maps ErrRange to a refusal maps this one too.
var ErrBudget = fmt.Errorf("%w: compute quota", ErrRange)

// DispatchInvocations returns the upper bound of the invocations of one
// Dispatch whose workgroup holds at most wgInvocations invocations: the
// product of its three workgroup counts and wgInvocations, saturating at
// MaxUint64. WebGPU validation guarantees a pipeline's workgroup size is at
// most the device's maxComputeInvocationsPerWorkgroup, so passing that limit
// is a sound bound without reading the WGSL.
func DispatchInvocations(x, y, z, wgInvocations uint32) uint64 {
	n := uint64(1)
	for _, f := range [4]uint32{x, y, z, wgInvocations} {
		hi, lo := bits.Mul64(n, uint64(f))
		if hi != 0 {
			return ^uint64(0)
		}
		n = lo
	}
	return n
}

// CheckInvocations bounds the compute work a stream asks for, before any
// replay: no Dispatch may exceed perDispatch invocations and the stream as a
// whole may not exceed perStream (a bound of 0 is no bound), counting every
// Dispatch at DispatchInvocations(.., wgInvocations). It returns the stream's
// total and largest Dispatch (both saturating) so a host can record the
// peaks. A DispatchIndirect takes its counts from GPU memory and cannot be
// bounded here: it counts as zero, and the host's per-submit time budget is
// what bounds it.
func CheckInvocations(cmds []Cmd, wgInvocations uint32, perDispatch, perStream uint64) (total, largest uint64, err error) {
	for _, c := range cmds {
		if c.Op != Dispatch {
			continue
		}
		n := DispatchInvocations(c.Args[0], c.Args[1], c.Args[2], wgInvocations)
		largest = max(largest, n)
		if perDispatch != 0 && n > perDispatch {
			return total, largest, fmt.Errorf("%w: Dispatch of %d invocations above the quota of %d", ErrBudget, n, perDispatch)
		}
		if total += n; total < n {
			total = ^uint64(0)
		}
		if perStream != 0 && total > perStream {
			return total, largest, fmt.Errorf("%w: stream of %d invocations above the quota of %d", ErrBudget, total, perStream)
		}
	}
	return total, largest, nil
}
