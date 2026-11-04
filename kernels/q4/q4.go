// Package q4 generates the WGSL of a matrix-vector product for ggml Q4_0 and
// Q4_1 weights, f32 activations, with no subgroups and no f16.
//
// The weights are repacked by the caller: the 16 quant bytes of a block as one
// aligned vec4<u32> (array<vec4<u32>>, one element per block, row-major), the
// block scale (and for Q4_1 the minimum) as f32 arrays. A workgroup of WG
// invocations computes RPW = WG/TPR output rows; TPR invocations share a row
// and each takes the blocks b = lane, lane+TPR, ... of it; the input vector is
// staged once per workgroup in workgroup memory (in tiles when it does not fit
// MaxShared) and the TPR partial sums of a row are added by a tree reduction
// in workgroup memory. Sizes come from the adapter's limits, never from a
// vendor's values; the summation order depends on WG and TPR only.
//
// The z axis of the dispatch is a batch of independent input vectors (tokens):
// vector tk is read at tk*N floats and row r of vector tk written at
// tk*Rows+r, so a chunk of vectors reads every weight once.
//
// Callers add fused prologues and epilogues through Hooks (an RMSNorm in
// front, a SiLU, RoPE or attention merge behind); Store and Acc are the plain
// ones. The shaders are generated per model: dimensions are WGSL constants.
package q4

import (
	"strconv"
	"strings"
)

// Prelude is the workgroup reduction helpers {{WG}} wide: rsum and rmax add
// or take the maximum of one value per invocation by a tree reduction and
// return it to all of them. Replace {{WG}} before use.
const Prelude = `
var<workgroup> red: array<f32, {{WG}}>;

fn rsum(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  var s = {{WG}}u / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) { red[li] = red[li] + red[li + s]; }
    workgroupBarrier();
    s = s >> 1u;
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

fn rmax(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  var s = {{WG}}u / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
    s = s >> 1u;
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
`

// Mode selects the epilogue.
type Mode int

const (
	// Store: y[tk*Rows+row] = result (binding y, read_write f32).
	Store Mode = iota
	// Acc: y[tk*Rows+row] += result (a residual add).
	Acc
	// Custom: Hooks.RowMap and Hooks.Epi (and Hooks.Bindings) do the work.
	Custom
)

// Binding is an extra storage binding, numbered after the standard ones.
type Binding struct {
	Name, Type string // WGSL name and type, e.g. "pv", "array<vec4<f32>>"
	RW         bool
}

// Hooks are the fused parts of a kernel, as WGSL text. In scope: wg (workgroup
// index in the 2-D grid), tk (z, the vector), lr (row in the workgroup), ln
// (lane in the row), li, the constants WG TPR RPW N NB ROWS NX EPS, red.
type Hooks struct {
	Bindings []Binding // after qs, scale, [mm], [x4]; y is added by Store and Acc
	Decl     string    // module-scope declarations (constants)
	Func     string    // module-scope functions
	NoInput  bool      // no x4 binding: Load computes the input
	Load     string    // expression of type vec4<f32>: input element gi (vec4 index) of vector tk
	RowMap   string    // statements defining `row` (u32) and `valid` (bool)
	Epi      string    // statements after the reduction: red[lr*TPR] is row lr's sum
}

// Spec describes one kernel.
type Spec struct {
	WG, TPR   int // workgroup size and invocations per row (powers of two, TPR | WG)
	MaxShared int // the adapter's workgroup memory, bytes
	MaxGroups int // the adapter's workgroups per dimension
	N, Rows   int // input length (a multiple of 32) and output rows
	Eps       float32
	Q41       bool // Q4_1 (scale d and minimum m) instead of Q4_0
	Q80       bool // Q8_0 (32 signed bytes and a scale per block, quants as two vec4<u32> per block)
	// Norm puts an RMSNorm with weights in front: the scales binding is "aux"
	// holding the block scales followed, at float NW, by the N weights.
	Norm  bool
	NW    int
	Mode  Mode
	Hooks Hooks
}

func itoa(n int) string { return strconv.Itoa(n) }

// F32 formats a float as a WGSL literal.
func F32(x float32) string { return strconv.FormatFloat(float64(x), 'e', -1, 32) }

const kernel = `
const WG: u32 = {{WG}}u;
const TPR: u32 = {{TPR}}u;
const RPW: u32 = {{RPW}}u;
const N: u32 = {{N}}u;
const NB: u32 = {{NB}}u;
const ROWS: u32 = {{ROWS}}u;
const NX: u32 = {{NX}}u;
const TILE4: u32 = {{TILE4}}u;
const NTILES: u32 = {{NTILES}}u;
const TB: u32 = {{TB}}u;
const EPS: f32 = {{EPS}};
{{DECL}}
var<workgroup> xs: array<vec4<f32>, {{TILE4}}>;

fn nib(w: u32) -> vec4<f32> { return vec4<f32>(f32(w & 15u), f32((w >> 8u) & 15u), f32((w >> 16u) & 15u), f32((w >> 24u) & 15u)); }
fn nibh(w: u32) -> vec4<f32> { return nib(w >> 4u); }
{{FUNC}}

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let wg = wid.y * NX + wid.x;
  let tk = wid.z;
  let lr = li / TPR;
  let ln = li % TPR;
  {{ROWMAP}}
  let r = select(0u, row, valid);
  {{PRE}}
  var acc = 0.0;
  for (var t = 0u; t < NTILES; t = t + 1u) {
    let base4 = t * TILE4;
    var i = li;
    loop {
      if (i >= TILE4) { break; }
      let gi = base4 + i;
      var v = vec4<f32>(0.0);
      if (gi < N / 4u) { v = {{LOAD}}; }
      xs[i] = v;
      i = i + WG;
    }
    workgroupBarrier();
    var b = ln;
    loop {
      if (b >= TB) { break; }
      let gb = t * TB + b;
      if (gb < NB) {
        let xb = b * 8u;
        {{BLOCK}}
      }
      b = b + TPR;
    }
    workgroupBarrier();
  }
  red[li] = acc;
  workgroupBarrier();
  var s = TPR / 2u;
  loop {
    if (s == 0u) { break; }
    if (ln < s) { red[li] = red[li] + red[li + s]; }
    workgroupBarrier();
    s = s >> 1u;
  }
  {{EPI}}
}
`

// Q4_0 block: byte i holds element i (low nibble) and element i+16 (high
// nibble) of the 32 of the block, as in ggml.
const blockQ40 = `
        let q = qs[r * NB + gb];
        let sc = {{SCALE}}[r * NB + gb];
        var s4 = vec4<f32>(0.0);
        s4 = s4 + (nib(q.x) - 8.0) * xs[xb] + (nibh(q.x) - 8.0) * xs[xb + 4u];
        s4 = s4 + (nib(q.y) - 8.0) * xs[xb + 1u] + (nibh(q.y) - 8.0) * xs[xb + 5u];
        s4 = s4 + (nib(q.z) - 8.0) * xs[xb + 2u] + (nibh(q.z) - 8.0) * xs[xb + 6u];
        s4 = s4 + (nib(q.w) - 8.0) * xs[xb + 3u] + (nibh(q.w) - 8.0) * xs[xb + 7u];
        acc = acc + sc * (s4.x + s4.y + s4.z + s4.w);
`

// Q8_0: 32 signed bytes per block (two vec4<u32>, qs[2*block], qs[2*block+1]),
// element j is byte j; word k of the block covers elements 4k..4k+3.
const blockQ80 = `
        let q0 = qs[(r * NB + gb) * 2u];
        let q1 = qs[(r * NB + gb) * 2u + 1u];
        var s4 = i8x4(q0.x) * xs[xb] + i8x4(q0.y) * xs[xb + 1u] + i8x4(q0.z) * xs[xb + 2u] + i8x4(q0.w) * xs[xb + 3u];
        s4 = s4 + i8x4(q1.x) * xs[xb + 4u] + i8x4(q1.y) * xs[xb + 5u] + i8x4(q1.z) * xs[xb + 6u] + i8x4(q1.w) * xs[xb + 7u];
        acc = acc + {{SCALE}}[r * NB + gb] * (s4.x + s4.y + s4.z + s4.w);
`

const fnQ80 = `
fn i8x4(w: u32) -> vec4<f32> {
  let v = bitcast<i32>(w);
  return vec4<f32>(f32(extractBits(v, 0u, 8u)), f32(extractBits(v, 8u, 8u)), f32(extractBits(v, 16u, 8u)), f32(extractBits(v, 24u, 8u)));
}
`

// Q4_1: value = d*q + m, so a block's sum is d*sum(q*x) + m*sum(x).
const blockQ41 = `
        let q = qs[r * NB + gb];
        var s4 = vec4<f32>(0.0);
        s4 = s4 + nib(q.x) * xs[xb] + nibh(q.x) * xs[xb + 4u];
        s4 = s4 + nib(q.y) * xs[xb + 1u] + nibh(q.y) * xs[xb + 5u];
        s4 = s4 + nib(q.z) * xs[xb + 2u] + nibh(q.z) * xs[xb + 6u];
        s4 = s4 + nib(q.w) * xs[xb + 3u] + nibh(q.w) * xs[xb + 7u];
        let sx = xs[xb] + xs[xb + 1u] + xs[xb + 2u] + xs[xb + 3u] + xs[xb + 4u] + xs[xb + 5u] + xs[xb + 6u] + xs[xb + 7u];
        acc = acc + dd[r * NB + gb] * (s4.x + s4.y + s4.z + s4.w) + mm[r * NB + gb] * (sx.x + sx.y + sx.z + sx.w);
`

const preRMS = `
  var ss = 0.0;
  var si = li;
  loop {
    if (si >= N / 4u) { break; }
    let v = x4[tk * (N / 4u) + si];
    ss = ss + dot(v, v);
    si = si + WG;
  }
  let rs = 1.0 / sqrt(rsum(li, ss) / f32(N) + EPS);
`

const loadRMS = `x4[tk * (N / 4u) + gi] * rs * vec4<f32>(aux[NW + 4u * gi], aux[NW + 4u * gi + 1u], aux[NW + 4u * gi + 2u], aux[NW + 4u * gi + 3u])`

// RowPlain maps workgroup, row in the workgroup to row = wg*RPW + lr.
const RowPlain = `let row = wg * RPW + lr; let valid = row < ROWS;`

// EpiAcc and EpiStore are the plain epilogues (the y binding is last).
const (
	EpiStore = `
  if (ln == 0u && valid) { y[tk * ROWS + row] = red[li]; }`
	EpiAcc = `
  if (ln == 0u && valid) { y[tk * ROWS + row] = y[tk * ROWS + row] + red[li]; }`
)

// Tile returns the staged tile of the input: in vec4s, the number of tiles
// and the blocks per tile. The tile and the reduction array fit MaxShared.
func (s Spec) Tile() (tile4, ntiles, tb int) {
	avail := s.MaxShared - s.WG*4 - s.WG*16 - 512
	t := s.N
	if t*4 > avail {
		t = avail / 4 / 32 * 32
	}
	if t < 32 {
		t = 32
	}
	return t / 4, (s.N + t - 1) / t, t / 32
}

// Workgroups is the number of workgroups for a vector (before the z axis) in
// the 1-D count; a Custom mode whose workgroups do not map RPW rows each
// passes its own count to Source.
func (s Spec) Workgroups() int { return (s.Rows + s.WG/s.TPR - 1) / (s.WG / s.TPR) }

// Source returns the WGSL, the binding access string ('r' read-only, 'w'
// read-write, one letter per binding) and the grid: dispatch (X, Y, vectors).
// groups is the workgroup count (0: Workgroups()).
func Source(s Spec, groups int) (src, access string, gx, gy int) {
	rpw := s.WG / s.TPR
	if groups == 0 {
		groups = s.Workgroups()
	}
	gx = min(groups, s.MaxGroups)
	gy = (groups + gx - 1) / gx
	t4, nt, tb := s.Tile()
	nb := s.N / 32
	var decl, pre, load, rowmap, epi, block string
	b := 0
	bind := func(name, typ string, rw bool) {
		a := "read"
		access += "r"
		if rw {
			a = "read_write"
			access = access[:len(access)-1] + "w"
		}
		decl += "@group(0) @binding(" + itoa(b) + ") var<storage, " + a + "> " + name + ": " + typ + ";\n"
		b++
	}
	bind("qs", "array<vec4<u32>>", false)
	scale := "dd"
	if s.Norm {
		bind("aux", "array<f32>", false)
		scale = "aux"
		decl += "const NW: u32 = " + itoa(s.NW) + "u;\n"
	} else {
		bind("dd", "array<f32>", false)
	}
	block = blockQ40
	h := s.Hooks
	switch {
	case s.Q41:
		bind("mm", "array<f32>", false)
		block = blockQ41
	case s.Q80:
		block = blockQ80
		h.Func += fnQ80
	}
	switch {
	case h.NoInput:
		load = h.Load
	case s.Norm:
		bind("x4", "array<vec4<f32>>", false)
		pre, load = preRMS, loadRMS
	default:
		bind("x4", "array<vec4<f32>>", false)
		load = "x4[tk * (N / 4u) + gi]"
	}
	for _, e := range h.Bindings {
		bind(e.Name, e.Type, e.RW)
	}
	switch s.Mode {
	case Store:
		bind("y", "array<f32>", true)
		rowmap, epi = RowPlain, EpiStore
	case Acc:
		bind("y", "array<f32>", true)
		rowmap, epi = RowPlain, EpiAcc
	default:
		rowmap, epi = h.RowMap, h.Epi
	}
	decl += h.Decl
	block = strings.ReplaceAll(block, "{{SCALE}}", scale)
	r := strings.NewReplacer(
		"{{DECL}}", decl, "{{FUNC}}", h.Func, "{{PRE}}", pre, "{{LOAD}}", load, "{{ROWMAP}}", rowmap, "{{EPI}}", epi, "{{BLOCK}}", block,
		"{{RPW}}", itoa(rpw), "{{TPR}}", itoa(s.TPR), "{{N}}", itoa(s.N), "{{NB}}", itoa(nb), "{{ROWS}}", itoa(s.Rows),
		"{{NX}}", itoa(gx), "{{TILE4}}", itoa(t4), "{{NTILES}}", itoa(nt), "{{TB}}", itoa(tb), "{{EPS}}", F32(s.Eps), "{{WG}}", itoa(s.WG))
	src = r.Replace(Prelude + kernel)
	return src, access, gx, gy
}
