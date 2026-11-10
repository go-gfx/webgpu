package wire

import "fmt"

// ErrDrawBudget is wrapped by every CheckDraws refusal. It also wraps
// ErrRange, so a caller that maps ErrRange to a refusal maps this one too.
var ErrDrawBudget = fmt.Errorf("%w: draw quota", ErrRange)

// DrawVertices returns the vertex invocations of one Draw or DrawIndexed:
// vertexCount (or indexCount) times instanceCount. Two uint32 factors always
// fit in a uint64, so the product is exact; it is the sum over draws that
// saturates (CheckDraws).
func DrawVertices(count, instances uint32) uint64 {
	return uint64(count) * uint64(instances)
}

// CheckDraws bounds the render work a stream asks for, before any replay,
// the render counterpart of CheckInvocations: no Draw or DrawIndexed may
// exceed perDraw vertex invocations (vertexCount, or indexCount, times
// instanceCount) and the stream's draws together may not exceed perStream
// (a bound of 0 is no bound). It returns the stream's total (saturating at
// MaxUint64) and its largest draw so a host can record the peaks. A
// DrawIndirect or DrawIndexedIndirect takes its counts from GPU memory and
// cannot be bounded here, and the draws inside an executed render bundle are
// not in the stream: they count as zero, and the host's per-submit time
// budget is what bounds them.
func CheckDraws(cmds []Cmd, perDraw, perStream uint64) (total, largest uint64, err error) {
	for _, c := range cmds {
		if c.Op != Draw && c.Op != DrawIndexed {
			continue
		}
		n := DrawVertices(c.Args[0], c.Args[1])
		largest = max(largest, n)
		if perDraw != 0 && n > perDraw {
			return total, largest, fmt.Errorf("%w: draw of %d vertices above the quota of %d", ErrDrawBudget, n, perDraw)
		}
		if total += n; total < n {
			total = ^uint64(0)
		}
		if perStream != 0 && total > perStream {
			return total, largest, fmt.Errorf("%w: stream of %d vertices above the quota of %d", ErrDrawBudget, total, perStream)
		}
	}
	return total, largest, nil
}
