package batch

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/go-gfx/webgpu/wire"
	"github.com/go-gfx/wgsl"
	"github.com/go-gfx/wgsl/transform"
)

var lb = LoopBudget{Work: 1 << 36, Group: 3}

const loopy = `@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1) fn main() { var s = 0u; for (var i = 0u; i < 4000000000u; i++) { s += i; } o[0] = s; }`

func TestInstrumentRewritesLoops(t *testing.T) {
	out, ok, err := lb.Instrument(loopy)
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	for _, want := range []string{
		"@group(3) @binding(1) var<uniform> wgsl_loop_budget: u32",
		"@group(3) @binding(0) var<storage, read_write> wgsl_truncated: atomic<u32>", "atomicStore"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in\n%s", want, out)
		}
	}
}

func TestInstrumentLeavesLoopFreeAlone(t *testing.T) {
	for name, src := range map[string]string{
		"no loop":         `@compute @workgroup_size(1) fn main() {}`,
		"render, no loop": `@vertex fn v() -> @builtin(position) vec4f { return vec4f(); }`,
		"loop, no entry":  `fn f() { loop { break; } }`,
	} {
		out, ok, err := lb.Instrument(src)
		if err != nil || ok || out != src {
			t.Errorf("%s: ok=%v err=%v changed=%v", name, ok, err, out != src)
		}
	}
}

// What cannot be bounded is refused, never passed through.
func TestInstrumentRefusesWhatItCannotBound(t *testing.T) {
	for name, c := range map[string]struct {
		src  string
		want error
	}{
		"unparsable":     {`@compute fn main( { loop`, ErrUnparsable},
		"vertex loop":    {`@vertex fn v(@builtin(vertex_index) n: u32) -> @builtin(position) vec4f { for (var i = 0u; i < n; i++) {} return vec4f(); }`, ErrUnbounded},
		"fragment loop":  {`@fragment fn f() -> @location(0) vec4f { while (true) { break; } return vec4f(); }`, ErrUnbounded},
		"vertex+compute": {`@vertex fn v() -> @builtin(position) vec4f { return vec4f(); } @compute @workgroup_size(1) fn c() { loop { break; } }`, ErrUnbounded},
	} {
		out, ok, err := lb.Instrument(c.src)
		if !errors.Is(err, c.want) || ok || out != "" {
			t.Errorf("%s: out=%q ok=%v err=%v", name, out, ok, err)
		}
	}
}

func TestInstrumentRefusesReservedGroup(t *testing.T) {
	for _, g := range []string{"@group(3) @binding(0)", "@group(3u) @binding(7)"} {
		src := g + ` var<uniform> u: u32; @compute @workgroup_size(1) fn main() {}`
		if _, _, err := lb.Instrument(src); !errors.Is(err, ErrReserved) {
			t.Errorf("%s: %v", g, err)
		}
	}
	if _, _, err := lb.Instrument(`@group(2) @binding(0) var<uniform> u: u32; @compute @workgroup_size(1) fn main() {}`); err != nil {
		t.Error(err)
	}
}

func TestIterations(t *testing.T) {
	l := LoopBudget{Work: 1 << 36}.withDefaults()
	for _, c := range []struct {
		inv  uint64
		want uint32
	}{
		{1, 1 << 24},      // tiny dispatch: the largest
		{1 << 4, 1 << 24}, // 2^32 > the cap
		{1 << 5, 1 << 24},
		{1 << 26, 1 << 10}, // the dispatch quota: the floor
		{1 << 30, 1 << 10}, // below the floor: the floor
		{3 << 20, 1 << 14}, // 2^36/3M = 21845 -> 16384
		{0, 1 << 24},
	} {
		if got := l.Iterations(c.inv); got != c.want {
			t.Errorf("%d invocations: %d, want %d", c.inv, got, c.want)
		}
	}
}

func TestSmallPieces(t *testing.T) {
	if (LoopBudget{}).Validate() == nil || (LoopBudget{Work: 1}).Validate() != nil {
		t.Error("Validate")
	}
	// A cap of the whole uint32 range is the last budget slot.
	l := LoopBudget{Work: 1 << 62, MaxIterations: 1<<32 - 1}.withDefaults()
	if got := l.Iterations(1); got != 1<<32-1 || l.slot(1) != 32 {
		t.Errorf("last slot: %d, slot %d", got, l.slot(1))
	}
	if litValue("abc") != -1 || litValue("3u") != 3 || litValue("0x10") != 16 {
		t.Error("litValue")
	}
	if _, _, err := lb.inject([]uint32{9999}, 0, nil, nil); err == nil {
		t.Error("a bad opcode was accepted")
	}
	// An unbind (id 0) is tracked, so the host's set after it is emitted.
	plan := map[uint32]*pipePlan{5: {empties: []binding{{1, 90}}, flag: 91}}
	w := []uint32{
		uint32(wire.BeginComputePass),
		uint32(wire.SetPipeline), 5,
		uint32(wire.SetBindGroup), 1, 0, 0,
		uint32(wire.Dispatch), 1, 1, 1,
		uint32(wire.EndPass),
	}
	if got, inj, err := (LoopBudget{Work: 1 << 36, Group: 3}).inject(w, 0, plan, nil); err != nil || !inj || len(got) <= len(w) {
		t.Errorf("unbind: %v %v %v", got, inj, err)
	}
}

func TestInject(t *testing.T) {
	const pipe, other = 5, 6
	l := LoopBudget{Work: 1 << 36, Group: 3, WorkgroupInvocations: 64}
	plan := map[uint32]*pipePlan{pipe: {empties: []binding{{1, 90}, {2, 92}}, flag: 91}}
	w := []uint32{
		uint32(wire.BeginComputePass),
		uint32(wire.SetPipeline), pipe,
		uint32(wire.SetBindGroup), 0, 70, 0,
		uint32(wire.Dispatch), 1024, 1, 1, // 65536 invocations: 2^36/2^16 = 2^20
		uint32(wire.DispatchIndirect), 70, 0, 0, // 2^26: the floor 2^10
		uint32(wire.SetPipeline), other,
		uint32(wire.Dispatch), 1, 1, 1,
		uint32(wire.EndPass),
	}
	got, inj, err := l.inject(slices.Clone(w), 0, plan, nil)
	if err != nil || !inj {
		t.Fatal(inj, err)
	}
	want := []uint32{
		uint32(wire.BeginComputePass),
		uint32(wire.SetPipeline), pipe,
		uint32(wire.SetBindGroup), 1, 90, 0,
		uint32(wire.SetBindGroup), 2, 92, 0,
		uint32(wire.SetBindGroup), 0, 70, 0,
		uint32(wire.SetBindGroup), 3, 91, 1, 20 * budgetStride,
		uint32(wire.Dispatch), 1024, 1, 1,
		uint32(wire.SetBindGroup), 3, 91, 1, 10 * budgetStride,
		uint32(wire.DispatchIndirect), 70, 0, 0,
		uint32(wire.SetPipeline), other,
		uint32(wire.Dispatch), 1, 1, 1,
		uint32(wire.EndPass),
		uint32(wire.Submit),
	}
	if !slices.Equal(got, want) {
		t.Errorf("got %v\nwant %v", got, want)
	}
	// Repeats are not set again: the same pipeline and size twice.
	rep := []uint32{
		uint32(wire.BeginComputePass),
		uint32(wire.SetPipeline), pipe,
		uint32(wire.Dispatch), 1024, 1, 1,
		uint32(wire.SetPipeline), pipe,
		uint32(wire.Dispatch), 1024, 1, 1,
		uint32(wire.EndPass),
	}
	wantRep := []uint32{
		uint32(wire.BeginComputePass),
		uint32(wire.SetPipeline), pipe,
		uint32(wire.SetBindGroup), 1, 90, 0,
		uint32(wire.SetBindGroup), 2, 92, 0,
		uint32(wire.SetBindGroup), 3, 91, 1, 20 * budgetStride,
		uint32(wire.Dispatch), 1024, 1, 1,
		uint32(wire.SetPipeline), pipe,
		uint32(wire.Dispatch), 1024, 1, 1,
		uint32(wire.EndPass),
		uint32(wire.Submit),
	}
	if g2, _, err := l.inject(rep, 0, plan, nil); err != nil || !slices.Equal(g2, wantRep) {
		t.Errorf("repeat: got %v (%v)\nwant %v", g2, err, wantRep)
	}
	// The tail goes before a final Submit, not after it.
	withSubmit := []uint32{
		uint32(wire.BeginComputePass),
		uint32(wire.SetPipeline), pipe,
		uint32(wire.Dispatch), 1024, 1, 1,
		uint32(wire.EndPass),
		uint32(wire.Submit),
	}
	g3, _, err := l.inject(withSubmit, 0, plan, []uint32{777, 888})
	if err != nil || g3[len(g3)-1] != uint32(wire.Submit) || g3[len(g3)-3] != 777 || g3[len(g3)-2] != 888 {
		t.Errorf("tail: %v (%v)", g3, err)
	}
	same, inj, err := l.inject(w, 0, map[uint32]*pipePlan{99: {flag: 1}}, nil)
	if err != nil || inj || &same[0] != &w[0] {
		t.Errorf("unchanged expected: %v", err)
	}
	bad := []uint32{uint32(wire.BeginComputePass), uint32(wire.SetBindGroup), 3, 70, 0, uint32(wire.EndPass)}
	if _, _, err := l.inject(bad, 0, plan, nil); !errors.Is(err, ErrReserved) {
		t.Errorf("reserved index: %v", err)
	}
}

// A rewrite that fails is a module the host cannot bound.
func TestInstrumentRewriteFailure(t *testing.T) {
	old := rewrite
	defer func() { rewrite = old }()
	rewrite = func(*wgsl.Module, transform.Config) (*wgsl.Module, *transform.Report, error) {
		return nil, nil, errors.New("boom")
	}
	if _, _, err := lb.Instrument(loopy); !errors.Is(err, ErrUnbounded) {
		t.Errorf("err %v", err)
	}
}
