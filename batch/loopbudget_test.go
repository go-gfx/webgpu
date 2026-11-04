package batch

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/go-gfx/webgpu/wire"
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

func TestInstrumentLeavesOthersAlone(t *testing.T) {
	for name, src := range map[string]string{
		"no loop":      `@compute @workgroup_size(1) fn main() {}`,
		"unparsable":   `@compute fn main( { loop`,
		"render only":  `@vertex fn v() -> @builtin(position) vec4f { for (var i = 0; i < 3; i++) {} return vec4f(); }`,
		"vertex+comp.": `@vertex fn v() -> @builtin(position) vec4f { return vec4f(); } @compute @workgroup_size(1) fn c() { loop { break; } }`,
	} {
		out, ok, err := lb.Instrument(src)
		if err != nil || ok || out != src {
			t.Errorf("%s: ok=%v err=%v changed=%v", name, ok, err, out != src)
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
		{1, math.MaxUint32},      // tiny dispatch: the largest
		{1 << 4, math.MaxUint32}, // 2^32 > u32
		{1 << 5, 1 << 31},
		{1 << 26, 1 << 10}, // the dispatch quota: the floor
		{1 << 30, 1 << 10}, // below the floor: the floor
		{3 << 20, 1 << 14}, // 2^36/3M = 21845 -> 16384
		{0, math.MaxUint32},
	} {
		if got := l.Iterations(c.inv); got != c.want {
			t.Errorf("%d invocations: %d, want %d", c.inv, got, c.want)
		}
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
	got, inj, err := l.inject(slices.Clone(w), 0, plan)
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
	}
	if !slices.Equal(got, want) {
		t.Errorf("got %v\nwant %v", got, want)
	}
	same, inj, err := l.inject(w, 0, map[uint32]*pipePlan{99: {flag: 1}})
	if err != nil || inj || &same[0] != &w[0] {
		t.Errorf("unchanged expected: %v", err)
	}
	bad := []uint32{uint32(wire.BeginComputePass), uint32(wire.SetBindGroup), 3, 70, 0, uint32(wire.EndPass)}
	if _, _, err := l.inject(bad, 0, plan); !errors.Is(err, ErrReserved) {
		t.Errorf("reserved index: %v", err)
	}
}
