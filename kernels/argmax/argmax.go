// Package argmax generates the WGSL of a greedy sampler in two stages, so that
// a large vocabulary is not walked by one workgroup: Groups workgroups each
// reduce a slice of the logits to its (largest, lowest index among equals),
// then one workgroup reduces the partials and writes the token. Ties go to
// the lowest index, like llama.cpp's greedy sampler.
//
// Stage 1 bindings: 0 logits (read f32), 1 partial values (read_write f32,
// Groups), 2 partial indices (read_write u32, Groups); dispatch Groups x 1.
// Stage 2 bindings: 0 values, 1 indices (read), 2 tokens (read_write u32),
// 3 out (read_write u32), 4 state (read u32); dispatch 1 x 1. It writes the
// token to tokens[state[1]+1] and to out[0].
package argmax

import (
	"strconv"
	"strings"
)

// Groups is the number of workgroups of stage 1.
const Groups = 64

// Access strings of the two stages, one letter per binding.
const (
	Access1 = "rww"
	Access2 = "rrwwr"
)

const stage1 = `
const VOCAB: u32 = {{VOCAB}}u;
const SL: u32 = {{SL}}u;
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> pbv: array<f32>;
@group(0) @binding(2) var<storage, read_write> pbi: array<u32>;
var<workgroup> bv: array<f32, {{WG}}>;
var<workgroup> bi: array<u32, {{WG}}>;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let lo = wid.x * SL;
  let hi = min(lo + SL, VOCAB);
  var best = -3.0e38;
  var bidx = lo;
  var i = lo + li;
  loop {
    if (i >= hi) { break; }
    let v = logits[i];
    if (v > best) { best = v; bidx = i; }
    i = i + {{WG}}u;
  }
  bv[li] = best;
  bi[li] = bidx;
  workgroupBarrier();
  var s = {{WG}}u / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) {
      let ov = bv[li + s];
      let oi = bi[li + s];
      if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; }
    }
    workgroupBarrier();
    s = s >> 1u;
  }
  if (li == 0u) { pbv[wid.x] = bv[0]; pbi[wid.x] = bi[0]; }
}
`

const stage2 = `
const NWG: u32 = {{NWG}}u;
@group(0) @binding(0) var<storage, read> pbv: array<f32>;
@group(0) @binding(1) var<storage, read> pbi: array<u32>;
@group(0) @binding(2) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(3) var<storage, read_write> outv: array<u32>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
var<workgroup> bv: array<f32, {{WG}}>;
var<workgroup> bi: array<u32, {{WG}}>;

@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_index) li: u32) {
  var best = -3.0e38;
  var bidx = 0xffffffffu;
  var i = li;
  loop {
    if (i >= NWG) { break; }
    let v = pbv[i];
    if (v > best) { best = v; bidx = pbi[i]; }
    i = i + {{WG}}u;
  }
  bv[li] = best;
  bi[li] = bidx;
  workgroupBarrier();
  var s = {{WG}}u / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) {
      let ov = bv[li + s];
      let oi = bi[li + s];
      if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; }
    }
    workgroupBarrier();
    s = s >> 1u;
  }
  if (li == 0u) {
    tokens[state[1] + 1u] = bi[0];
    outv[0] = bi[0];
  }
}
`

// Sources returns the WGSL of the two stages for workgroup size wg (a power
// of two) and a vocabulary of vocab logits.
func Sources(wg, vocab int) (s1, s2 string) {
	sl := (vocab + Groups - 1) / Groups
	r := strings.NewReplacer("{{VOCAB}}", strconv.Itoa(vocab), "{{SL}}", strconv.Itoa(sl), "{{WG}}", strconv.Itoa(wg), "{{NWG}}", strconv.Itoa(Groups))
	return r.Replace(stage1), r.Replace(stage2)
}
