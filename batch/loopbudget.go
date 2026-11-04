package batch

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"strconv"
	"strings"

	"github.com/go-gfx/webgpu/wire"
	"github.com/go-gfx/wgsl"
	"github.com/go-gfx/wgsl/transform"
)

// LoopBudget bounds the GPU time one dispatch can burn inside a loop, which
// WebGPU cannot cancel once submitted (layer 4 of go-wasmcontainers/gpu's
// scheduling). A DeviceBackend with a LoopBudget rewrites every guest WGSL
// module that has a compute entry point (go-gfx/wgsl's transform.LoopBudget):
// each outermost loop nest of each function activation runs at most B
// iterations, then the invocation leaves the loop and stores 1 into a
// host-owned flag. After every stream that ran such a pipeline the backend
// reads the flag; a set flag fails the stream with ErrTruncated, so a
// partial result is never mistaken for a complete one.
//
// The budget B of a dispatch is Work divided by its invocations (its
// workgroup counts times WorkgroupInvocations), rounded down to a power of
// two and at least MinIterations: a dispatch's loops cost at most about
// Work invocation-iterations whatever its size, so a big dispatch gets short
// loops and a small one long loops. A DispatchIndirect, whose counts are in
// GPU memory, is budgeted as a dispatch of IndirectInvocations.
//
// Reserved slots. In bind group Group, binding 0 is the flag (`var<storage,
// read_write> atomic<u32>`) and binding 1 the budget (`var<uniform> u32`,
// bound with a dynamic offset into a host buffer holding 2^0 ... 2^32-1, one
// per 256 bytes). Group is reserved on the whole device: a guest pipeline
// layout with more than Group bind group layouts, a WGSL module that
// declares a binding in @group(Group), and a SetBindGroup of index Group or
// more are refused with ErrReserved. Group is normally the device's
// maxBindGroups-1 (3 by default).
type LoopBudget struct {
	Work                 uint64 // invocation-iterations per dispatch (> 0)
	MinIterations        uint32 // the smallest B (default 1024)
	WorkgroupInvocations uint32 // invocations one workgroup counts as (default 256)
	IndirectInvocations  uint64 // a DispatchIndirect counts as this many (default 2^26)
	Group                uint32 // the reserved bind group
}

// Errors of the loop budget.
var (
	// ErrReserved: the guest used the bind group the host reserved for its
	// loop-budget flag.
	ErrReserved = errors.New("batch: the guest uses the bind group reserved by the host")
	// ErrTruncated: at least one shader invocation hit the loop budget and
	// was cut; the stream's results are partial and are not returned.
	ErrTruncated = errors.New("batch: a shader exceeded its loop budget and was truncated")
)

// budgetSlots is the number of budget values in the host's budget buffer:
// 2^0 ... 2^31 and the largest uint32.
const (
	budgetSlots  = 33
	budgetStride = 256 // minUniformBufferOffsetAlignment's default
)

// Validate checks the configuration.
func (lb LoopBudget) Validate() error {
	if lb.Work == 0 {
		return errors.New("batch: LoopBudget.Work is zero")
	}
	return nil
}

func (lb LoopBudget) withDefaults() LoopBudget {
	if lb.MinIterations == 0 {
		lb.MinIterations = 1024
	}
	if lb.WorkgroupInvocations == 0 {
		lb.WorkgroupInvocations = 256
	}
	if lb.IndirectInvocations == 0 {
		lb.IndirectInvocations = 1 << 26
	}
	return lb
}

// Iterations returns the budget B of a dispatch of inv invocations: the
// per-nest iteration bound its loops get. lb must have its defaults filled.
func (lb LoopBudget) Iterations(inv uint64) uint32 {
	slot := lb.slot(inv)
	if slot >= 32 {
		return math.MaxUint32
	}
	return 1 << slot
}

// slot is the index of the budget value of a dispatch of inv invocations.
func (lb LoopBudget) slot(inv uint64) uint32 {
	b := uint64(math.MaxUint32)
	if inv > 0 {
		b = min(b, lb.Work/inv)
	}
	b = max(b, uint64(lb.MinIterations))
	if b >= math.MaxUint32 {
		return 32
	}
	return uint32(bits.Len64(b) - 1)
}

// Instrument returns the module text to hand to the browser, and whether it
// was rewritten. It parses only: a module that does not parse is returned
// unchanged and un-instrumented (the browser stays the validation
// authority and reports its own error), as is a module without a compute
// entry point or one that also has a vertex entry (a vertex stage cannot
// write storage). A module that declares a binding in the reserved group is
// refused with ErrReserved.
func (lb LoopBudget) Instrument(code string) (string, bool, error) {
	m, err := wgsl.Parse(code)
	if err != nil {
		return code, false, nil
	}
	var compute, vertex bool
	for _, d := range m.Decls {
		switch d := d.(type) {
		case *wgsl.Func:
			for _, a := range d.Attrs {
				switch a.Name {
				case "compute":
					compute = true
				case "vertex":
					vertex = true
				}
			}
		case *wgsl.Var:
			for _, a := range d.Attrs {
				if a.Name == "group" && len(a.Args) == 1 {
					if lit, ok := a.Args[0].(*wgsl.IntLit); ok && litValue(lit.Text) == int64(lb.Group) {
						return "", false, fmt.Errorf("%w: @group(%d) in a WGSL module", ErrReserved, lb.Group)
					}
				}
			}
		}
	}
	if !compute || vertex {
		return code, false, nil
	}
	out, rep, err := transform.LoopBudget(m, transform.Config{
		Budget:    transform.BudgetSource{Group: int(lb.Group), Binding: 1},
		FlagGroup: int(lb.Group), FlagBinding: 0,
	})
	if err != nil || !rep.Instrumented {
		return code, false, nil
	}
	return wgsl.Print(out), true, nil
}

// litValue is the value of an integer literal's text ("3", "3u", "0x3"), -1
// if it is not one.
func litValue(s string) int64 {
	s = strings.TrimRight(s, "ui")
	v, err := strconv.ParseInt(s, 0, 64)
	if err != nil {
		return -1
	}
	return v
}

// binding is one bind group a backend sets after a SetPipeline.
type binding struct {
	index uint32
	id    uint32 // replay id of the bind group
}

// pipePlan is what a backend adds around an instrumented compute pipeline:
// the empty groups its padded layout needs, and the flag group, set (with
// the dispatch's budget as dynamic offset) before every dispatch.
type pipePlan struct {
	empties []binding
	flag    uint32 // replay id of the bind group of the flag and the budgets
}

// inject returns words with the plan's groups added around each use of an
// instrumented pipeline: the empty groups after its SetPipeline, the flag
// group before each of its dispatches. It refuses a SetBindGroup of index
// Group or more. The words come back unchanged (the same slice, false) when
// nothing is added.
func (lb LoopBudget) inject(words []uint32, ndata uint32, plan map[uint32]*pipePlan) ([]uint32, bool, error) {
	cmds, err := wire.Decode(words, ndata, func(uint32) bool { return true })
	if err != nil {
		return nil, false, err
	}
	lb = lb.withDefaults()
	var out []uint32
	var cur *pipePlan
	emitted := 0 // cmds[:emitted] are in out
	flush := func(upTo int) {
		if out == nil {
			out = make([]uint32, 0, len(words)+len(words)/4+16)
		}
		for _, c := range cmds[emitted:upTo] {
			out = append(out, uint32(c.Op))
			out = append(out, c.Args...)
		}
		emitted = upTo
	}
	setFlag := func(slot uint32) {
		out = append(out, uint32(wire.SetBindGroup), lb.Group, cur.flag, 1, slot*budgetStride)
	}
	for i, c := range cmds {
		switch c.Op {
		case wire.BeginComputePass, wire.EndPass, wire.BeginRenderPass:
			cur = nil
		case wire.SetBindGroup:
			if c.Args[0] >= lb.Group {
				return nil, false, fmt.Errorf("%w: SetBindGroup index %d", ErrReserved, c.Args[0])
			}
		case wire.SetPipeline:
			cur = plan[c.Args[0]]
			if cur != nil {
				flush(i + 1)
				for _, b := range cur.empties {
					out = append(out, uint32(wire.SetBindGroup), b.index, b.id, 0)
				}
			}
		case wire.Dispatch:
			if cur != nil {
				flush(i)
				setFlag(lb.slot(uint64(c.Args[0]) * uint64(c.Args[1]) * uint64(c.Args[2]) * uint64(lb.WorkgroupInvocations)))
			}
		case wire.DispatchIndirect:
			if cur != nil {
				flush(i)
				setFlag(lb.slot(lb.IndirectInvocations))
			}
		}
	}
	if out == nil {
		return words, false, nil
	}
	flush(len(cmds))
	return out, true, nil
}
