//go:build js

package batch

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
)

// Guard is the loop budget (LoopBudget) of one browser device: the host's
// flag, budget and empty bind groups, the table of rewritten modules and
// pipelines, and the checks every path that reaches createShaderModule must
// go through. DeviceBackend (the batched stream) uses it by replay id;
// objects-API hosts (the wasi:webgpu module, wasi-nn) use Shader, Layout,
// Pipeline, Pass, Finish and Submit. A module the Guard cannot bound is
// refused with ErrUnbounded: nothing unbounded runs.
type Guard struct {
	dev *webgpu.Device
	lb  LoopBudget

	flag, read, budget *webgpu.Buffer // 4 bytes each: the flag and its MAP_READ copy; the budgets
	flagBGL, emptyBGL  *webgpu.BindGroupLayout
	flagBG, emptyBG    *webgpu.BindGroup

	mu      sync.Mutex
	instr   map[*webgpu.ShaderModule]bool
	groups  map[*webgpu.ShaderModule][]uint32                    // the groups a module declares
	layouts map[*webgpu.PipelineLayout][]*webgpu.BindGroupLayout // a guest layout's own groups
	derived map[*webgpu.PipelineLayout]*webgpu.PipelineLayout    // its padded twin
	plan    map[uint32]*pipePlan                                 // batch: compute pipeline replay id
	oplan   map[*webgpu.ComputePipeline]objPlan                  // objects API
	armedEn map[*webgpu.CommandEncoder]bool
	armedCB map[*webgpu.CommandBuffer]*webgpu.Buffer // the flag's copy

	modules, rewritten, refused int           // WGSL modules seen, rewritten, refused as unboundable
	spent                       time.Duration // in LoopBudget.Instrument
}

// objPlan is how the objects API sets the host's groups for a pipeline:
// explicit layouts own groups below first; an auto layout (auto) takes the
// guest's bound groups as they are.
//
// The bind group layouts of an auto layout belong to their pipeline: a group
// is compatible with them only if it was made from them, so an auto
// pipeline has its own flag group and its own empty groups (flag, empty).
type objPlan struct {
	first uint32
	auto  bool
	// An auto layout's budget binding has no dynamic offset, so each budget
	// slot has a group of its own, made when a dispatch first needs it.
	flagLayout *webgpu.BindGroupLayout
	flagSlots  map[uint32]*webgpu.BindGroup
	empty      map[uint32]*webgpu.BindGroup
}

// NewGuard makes the guard of a device (see LoopBudget).
func NewGuard(dev *webgpu.Device, lb LoopBudget) (*Guard, error) {
	if err := lb.Validate(); err != nil {
		return nil, err
	}
	g := &Guard{dev: dev, lb: lb.withDefaults(),
		instr:   map[*webgpu.ShaderModule]bool{},
		groups:  map[*webgpu.ShaderModule][]uint32{},
		layouts: map[*webgpu.PipelineLayout][]*webgpu.BindGroupLayout{},
		derived: map[*webgpu.PipelineLayout]*webgpu.PipelineLayout{},
		plan:    map[uint32]*pipePlan{},
		oplan:   map[*webgpu.ComputePipeline]objPlan{},
		armedEn: map[*webgpu.CommandEncoder]bool{},
		armedCB: map[*webgpu.CommandBuffer]*webgpu.Buffer{}}
	var err error
	if g.flag, err = dev.CreateBuffer(webgpu.BufferDescriptor{Size: 4, Usage: webgpu.BufferUsageStorage | webgpu.BufferUsageCopySrc | webgpu.BufferUsageCopyDst}); err != nil {
		return nil, err
	}
	if g.read, err = dev.CreateBuffer(webgpu.BufferDescriptor{Size: 4, Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst}); err != nil {
		return nil, err
	}
	if g.budget, err = dev.CreateBuffer(webgpu.BufferDescriptor{Size: budgetSlots * budgetStride, Usage: webgpu.BufferUsageUniform | webgpu.BufferUsageCopyDst}); err != nil {
		return nil, err
	}
	vals := make([]byte, budgetSlots*budgetStride)
	for k := range budgetSlots {
		v := uint32(math.MaxUint32)
		if k < 32 {
			v = 1 << k
		}
		binary.LittleEndian.PutUint32(vals[k*budgetStride:], v)
	}
	if err := dev.Queue().WriteBuffer(g.budget, 0, vals); err != nil {
		return nil, err
	}
	if g.flagBGL, err = dev.CreateBindGroupLayout(webgpu.BindGroupLayoutDescriptor{Entries: []webgpu.BindGroupLayoutEntry{
		{Binding: 0, Visibility: webgpu.ShaderStageCompute, Buffer: &webgpu.BufferBindingLayout{Type: webgpu.BufferBindingTypeStorage}},
		{Binding: 1, Visibility: webgpu.ShaderStageCompute, Buffer: &webgpu.BufferBindingLayout{Type: webgpu.BufferBindingTypeUniform, HasDynamicOffset: true, MinBindingSize: 4}}}}); err != nil {
		return nil, err
	}
	if g.emptyBGL, err = dev.CreateBindGroupLayout(webgpu.BindGroupLayoutDescriptor{}); err != nil {
		return nil, err
	}
	if g.flagBG, err = dev.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: g.flagBGL, Entries: []webgpu.BindGroupEntry{
		{Binding: 0, Buffer: g.flag},
		{Binding: 1, Buffer: g.budget, Size: 4}}}); err != nil {
		return nil, err
	}
	if g.emptyBG, err = dev.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: g.emptyBGL}); err != nil {
		return nil, err
	}
	return g, nil
}

// Close destroys the guard's buffers.
func (g *Guard) Close() {
	g.flag.Destroy()
	g.read.Destroy()
	g.budget.Destroy()
}

// Stats reports the WGSL modules seen, rewritten and refused as unboundable,
// and the time spent in the parser and the rewriter.
func (g *Guard) Stats() (modules, rewritten, refused int, spent time.Duration) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.modules, g.rewritten, g.refused, g.spent
}

// compile is CreateShaderModule inside a validation scope: a module the
// browser refuses is ErrInvalid with the compiler's message.
func (g *Guard) compile(d webgpu.ShaderModuleDescriptor) (*webgpu.ShaderModule, error) {
	g.dev.PushErrorScope(webgpu.ErrorFilterValidation)
	m, err := g.dev.CreateShaderModule(d)
	serr := g.dev.PopErrorScope(context.Background())
	var ge *webgpu.GPUError
	if errors.As(serr, &ge) {
		msg := ge.Message
		if m != nil && err == nil {
			// The compiler's own text: some browsers keep it out of the error.
			if cms, cerr := m.CompilationMessages(context.Background()); cerr == nil {
				for _, cm := range cms {
					if cm.Type == "error" {
						msg += "\n" + cm.String()
					}
				}
			}
		}
		return nil, fmt.Errorf("%w: %s", ErrInvalid, msg)
	}
	if err == nil && serr != nil {
		err = serr
	}
	return m, err
}

// Shader makes a guest's shader module, rewritten if it has loops. The
// browser's verdict on the guest's own text comes first, so an invalid
// module is reported as the browser reports it (the guest's own line and
// column). A valid module the host cannot bound (it does not parse, it has a
// loop in a module with a vertex or fragment entry, or the browser refuses
// the rewrite) is refused with ErrUnbounded: it never runs unbounded.
func (g *Guard) Shader(d webgpu.ShaderModuleDescriptor) (*webgpu.ShaderModule, error) {
	t0 := time.Now()
	an, err := g.lb.Analyze(d.Code)
	code, inst := an.Text, an.Rewritten
	g.mu.Lock()
	g.spent += time.Since(t0)
	g.modules++
	g.mu.Unlock()
	switch {
	case errors.Is(err, ErrUnparsable) || errors.Is(err, ErrUnbounded):
		if _, cerr := g.compile(d); cerr != nil {
			return nil, cerr
		}
		g.count(func() { g.refused++ })
		return nil, err
	case err != nil: // ErrReserved
		return nil, err
	case !inst:
		m, err := g.compile(d)
		g.noteGroups(m, an.Groups)
		return m, err
	}
	rd := d
	rd.Code = code
	m, cerr := g.compile(rd)
	if cerr == nil {
		g.mu.Lock()
		g.instr[m] = true
		g.rewritten++
		g.mu.Unlock()
		g.noteGroups(m, an.Groups)
		return m, nil
	}
	if !errors.Is(cerr, ErrInvalid) {
		return nil, cerr
	}
	if _, oerr := g.compile(d); oerr != nil {
		return nil, oerr
	}
	g.count(func() { g.refused++ })
	return nil, fmt.Errorf("%w: the browser refused the loop-budget rewrite of a module it accepts: %v", ErrUnbounded, cerr)
}

func (g *Guard) noteGroups(m *webgpu.ShaderModule, groups []uint32) {
	if m != nil {
		g.mu.Lock()
		g.groups[m] = groups
		g.mu.Unlock()
	}
}

// ShaderNoWait is Shader for a host that cannot wait on the browser (a
// synchronous call): the same checks and rewrite, but the browser's verdict
// is not asked for. A module the host cannot bound is refused at once; a
// module the browser refuses stays invalid (its pipeline fails), and a
// rewritten text the browser refuses cannot run either.
func (g *Guard) ShaderNoWait(d webgpu.ShaderModuleDescriptor) (*webgpu.ShaderModule, error) {
	t0 := time.Now()
	an, err := g.lb.Analyze(d.Code)
	g.mu.Lock()
	g.spent += time.Since(t0)
	g.modules++
	if err != nil && !errors.Is(err, ErrReserved) {
		g.refused++
	}
	g.mu.Unlock()
	if err != nil {
		return nil, err
	}
	rd := d
	rd.Code = an.Text
	m, err := g.dev.CreateShaderModule(rd)
	if err != nil {
		return nil, err
	}
	g.mu.Lock()
	if an.Rewritten {
		g.instr[m] = true
		g.rewritten++
	}
	g.groups[m] = an.Groups
	g.mu.Unlock()
	return m, nil
}

func (g *Guard) count(f func()) { g.mu.Lock(); f(); g.mu.Unlock() }

// CheckLayout refuses a guest layout that reaches the reserved group.
func (g *Guard) CheckLayout(bgls []*webgpu.BindGroupLayout) error {
	if uint32(len(bgls)) > g.lb.Group {
		return fmt.Errorf("%w: a pipeline layout with %d bind group layouts (at most %d)", ErrReserved, len(bgls), g.lb.Group)
	}
	return nil
}

// NoteLayout remembers a guest layout's own groups (after CheckLayout).
func (g *Guard) NoteLayout(pl *webgpu.PipelineLayout, bgls []*webgpu.BindGroupLayout) {
	g.mu.Lock()
	g.layouts[pl] = bgls
	g.mu.Unlock()
}

// prepare gives a compute pipeline descriptor its padded layout if its
// module was rewritten; first is the first group the host sets.
func (g *Guard) prepare(d webgpu.ComputePipelineDescriptor) (nd webgpu.ComputePipelineDescriptor, inst bool, first uint32, err error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.instr[d.Compute.Module] {
		return d, false, 0, nil
	}
	if d.Layout != nil {
		own, ok := g.layouts[d.Layout]
		if !ok { // a layout the guard did not see: it cannot carry the flag group
			return d, false, 0, fmt.Errorf("%w: a pipeline layout the host did not see", ErrUnbounded)
		}
		first = uint32(len(own))
		pl, ok := g.derived[d.Layout]
		if !ok {
			bgls := append([]*webgpu.BindGroupLayout(nil), own...)
			for i := first; i < g.lb.Group; i++ {
				bgls = append(bgls, g.emptyBGL)
			}
			bgls = append(bgls, g.flagBGL)
			if pl, err = g.dev.CreatePipelineLayout(webgpu.PipelineLayoutDescriptor{BindGroupLayouts: bgls}); err != nil {
				return d, false, 0, err
			}
			g.derived[d.Layout] = pl
		}
		d.Layout = pl
	}
	return d, true, first, nil
}

// planFor is the wire-stream plan of a rewritten pipeline, by replay id.
func (g *Guard) planFor(id, first uint32) error {
	eid, err := webgpu.ReplayID(g.emptyBG)
	if err != nil {
		return err
	}
	fid, err := webgpu.ReplayID(g.flagBG)
	if err != nil {
		return err
	}
	p := &pipePlan{flag: fid}
	for i := first; i < g.lb.Group; i++ {
		p.empties = append(p.empties, binding{i, eid})
	}
	g.mu.Lock()
	g.plan[id] = p
	g.mu.Unlock()
	return nil
}

// Pipeline makes a compute pipeline (objects API). A pipeline of a rewritten
// module gets the flag group in its layout; one of an unrewritten module has
// no loop to bound.
func (g *Guard) Pipeline(d webgpu.ComputePipelineDescriptor) (*webgpu.ComputePipeline, error) {
	nd, inst, first, err := g.prepare(d)
	if err != nil {
		return nil, err
	}
	p, err := g.dev.CreateComputePipeline(nd)
	if err != nil {
		return nil, err
	}
	return p, g.notePipeline(p, d, inst, first)
}

// notePipeline records a rewritten pipeline's plan. An auto layout's groups
// are made from the pipeline's own layouts.
func (g *Guard) notePipeline(p *webgpu.ComputePipeline, d webgpu.ComputePipelineDescriptor, inst bool, first uint32) error {
	if !inst {
		return nil
	}
	plan := objPlan{first: first}
	if d.Layout == nil {
		plan.auto = true
		l, err := p.GetBindGroupLayout(g.lb.Group)
		if err != nil {
			return err
		}
		plan.flagLayout, plan.flagSlots = l, map[uint32]*webgpu.BindGroup{}
		plan.empty = map[uint32]*webgpu.BindGroup{}
		g.mu.Lock()
		declared := g.groups[d.Compute.Module]
		g.mu.Unlock()
		for i := range g.lb.Group {
			if slices.Contains(declared, i) {
				continue // the guest binds it
			}
			l, err := p.GetBindGroupLayout(i) // no binding is declared at i: its layout is empty
			if err != nil {
				continue
			}
			if plan.empty[i], err = g.dev.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: l}); err != nil {
				delete(plan.empty, i)
			}
		}
	}
	g.mu.Lock()
	g.oplan[p] = plan
	g.mu.Unlock()
	return nil
}

// PipelineAsync is Pipeline with CreateComputePipelineAsync.
func (g *Guard) PipelineAsync(ctx context.Context, d webgpu.ComputePipelineDescriptor) (*webgpu.ComputePipeline, error) {
	nd, inst, first, err := g.prepare(d)
	if err != nil {
		return nil, err
	}
	p, err := g.dev.CreateComputePipelineAsync(ctx, nd)
	if err != nil {
		return nil, err
	}
	return p, g.notePipeline(p, d, inst, first)
}

// PassGuard wraps a compute pass encoder: before each dispatch of a
// rewritten pipeline it sets the host's groups (an empty group wherever the
// layout needs one, the flag group with the dispatch's budget).
type PassGuard struct {
	g     *Guard
	pe    *webgpu.ComputePassEncoder
	enc   *webgpu.CommandEncoder
	cur   objPlan
	use   bool
	armed bool
	bound map[uint32]heldObj
}

type heldObj struct {
	id   any
	off  uint32
	host bool // set by the host, not the guest
}

// Pass wraps pe, a pass of enc.
func (g *Guard) Pass(enc *webgpu.CommandEncoder, pe *webgpu.ComputePassEncoder) *PassGuard {
	return &PassGuard{g: g, pe: pe, enc: enc, bound: map[uint32]heldObj{}}
}

// SetPipeline is pe.SetPipeline.
func (p *PassGuard) SetPipeline(pl *webgpu.ComputePipeline) {
	p.pe.SetPipeline(pl)
	p.g.mu.Lock()
	p.cur, p.use = p.g.oplan[pl]
	p.g.mu.Unlock()
}

// SetBindGroup is pe.SetBindGroup; the reserved group is refused.
func (p *PassGuard) SetBindGroup(index uint32, bg *webgpu.BindGroup, offsets ...uint32) error {
	if index >= p.g.lb.Group {
		return fmt.Errorf("%w: SetBindGroup index %d", ErrReserved, index)
	}
	p.pe.SetBindGroup(index, bg, offsets...)
	if bg == nil {
		delete(p.bound, index)
	} else {
		p.bound[index] = heldObj{bg, math.MaxUint32, false}
	}
	return nil
}

func (p *PassGuard) set(index uint32, bg *webgpu.BindGroup, off uint32, dynamic bool) {
	h := heldObj{bg, 0, true}
	if dynamic {
		h.off = off + 1
	}
	if p.bound[index] == h {
		return
	}
	p.bound[index] = h
	if dynamic {
		p.pe.SetBindGroup(index, bg, off)
	} else {
		p.pe.SetBindGroup(index, bg)
	}
}

// host sets the host's groups for a dispatch of inv invocations.
func (p *PassGuard) host(inv uint64) {
	if !p.use {
		return
	}
	p.armed = true
	slot := p.g.lb.slot(inv)
	for i := range p.g.lb.Group {
		switch h, taken := p.bound[i]; {
		case p.cur.auto && taken && !h.host: // the guest's group, from this pipeline's layout
		case p.cur.auto:
			if e := p.cur.empty[i]; e != nil {
				p.set(i, e, 0, false)
			}
		case i >= p.cur.first:
			p.set(i, p.g.emptyBG, 0, false)
		}
	}
	if !p.cur.auto {
		p.set(p.g.lb.Group, p.g.flagBG, slot*budgetStride, true)
		return
	}
	bg := p.cur.flagSlots[slot]
	if bg == nil {
		var err error
		if bg, err = p.g.dev.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: p.cur.flagLayout, Entries: []webgpu.BindGroupEntry{
			{Binding: 0, Buffer: p.g.flag}, {Binding: 1, Buffer: p.g.budget, Offset: uint64(slot) * budgetStride, Size: 4}}}); err != nil {
			return // the dispatch then fails WebGPU validation: nothing runs unbounded
		}
		p.cur.flagSlots[slot] = bg
	}
	p.set(p.g.lb.Group, bg, 0, false)
}

// DispatchWorkgroups is pe.DispatchWorkgroups.
func (p *PassGuard) DispatchWorkgroups(x, y, z uint32) {
	p.host(uint64(x) * uint64(y) * uint64(z) * uint64(p.g.lb.WorkgroupInvocations))
	p.pe.DispatchWorkgroups(x, y, z)
}

// DispatchWorkgroupsIndirect is pe.DispatchWorkgroupsIndirect, budgeted as a
// dispatch of LoopBudget.IndirectInvocations.
func (p *PassGuard) DispatchWorkgroupsIndirect(b *webgpu.Buffer, offset uint64) {
	p.host(p.g.lb.IndirectInvocations)
	p.pe.DispatchWorkgroupsIndirect(b, offset)
}

// End is pe.End; an encoder whose pass used a rewritten pipeline is armed:
// Finish adds the flag's copy.
func (p *PassGuard) End() error {
	if p.armed {
		p.g.mu.Lock()
		p.g.armedEn[p.enc] = true
		p.g.mu.Unlock()
	}
	return p.pe.End()
}

// Finish is enc.Finish; the command buffer of an armed encoder carries a copy
// of the flag to a MAP_READ buffer of its own (submits may overlap, so each
// reads its own copy). The flag is never cleared between submits: once a loop
// is cut the work is condemned, and Submit re-arms it when it reports it.
func (g *Guard) Finish(enc *webgpu.CommandEncoder) (*webgpu.CommandBuffer, error) {
	g.mu.Lock()
	armed := g.armedEn[enc]
	delete(g.armedEn, enc)
	g.mu.Unlock()
	var rb *webgpu.Buffer
	if armed {
		var err error
		if rb, err = g.dev.CreateBuffer(webgpu.BufferDescriptor{Size: 4, Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst}); err != nil {
			return nil, err
		}
		if err := enc.CopyBufferToBuffer(g.flag, 0, rb, 0, 4); err != nil {
			rb.Destroy()
			return nil, err
		}
	}
	cb, err := enc.Finish()
	if err != nil && rb != nil {
		rb.Destroy()
	}
	if err == nil && armed {
		g.mu.Lock()
		g.armedCB[cb] = rb
		g.mu.Unlock()
	}
	return cb, err
}

// take removes and returns the flag copies of the armed command buffers.
func (g *Guard) take(cbs []*webgpu.CommandBuffer) []*webgpu.Buffer {
	g.mu.Lock()
	defer g.mu.Unlock()
	var rbs []*webgpu.Buffer
	for _, cb := range cbs {
		if rb := g.armedCB[cb]; rb != nil {
			rbs = append(rbs, rb)
			delete(g.armedCB, cb)
		}
	}
	return rbs
}

// Submit is q.Submit; if a command buffer is armed, the flag copies are read
// after the work (which waits for it): ErrTruncated if a loop was cut.
func (g *Guard) Submit(q *webgpu.Queue, cbs ...*webgpu.CommandBuffer) error {
	rbs := g.take(cbs)
	if err := q.Submit(cbs...); err != nil {
		for _, rb := range rbs {
			rb.Destroy()
		}
		return err
	}
	return g.check(q, rbs)
}

// SubmitNoWait is Submit for a host that cannot wait in the call (a
// synchronous import): it submits and returns a function that waits for the
// work and reads the flag copies (ErrTruncated if a loop was cut). The
// function must be called exactly once; it is nil if nothing was armed.
func (g *Guard) SubmitNoWait(q *webgpu.Queue, cbs ...*webgpu.CommandBuffer) (func() error, error) {
	rbs := g.take(cbs)
	if err := q.Submit(cbs...); err != nil {
		for _, rb := range rbs {
			rb.Destroy()
		}
		return nil, err
	}
	if len(rbs) == 0 {
		return nil, nil
	}
	return func() error { return g.check(q, rbs) }, nil
}

// check reads the flag copies, which completes after their work.
func (g *Guard) check(q *webgpu.Queue, rbs []*webgpu.Buffer) error {
	var out error
	for _, rb := range rbs {
		v, err := readFlag(rb)
		rb.Destroy()
		switch {
		case err != nil && out == nil:
			out = err
		case v != [4]byte{} && out == nil:
			out = ErrTruncated
		}
	}
	if errors.Is(out, ErrTruncated) {
		var zero [4]byte
		q.WriteBuffer(g.flag, 0, zero[:]) // re-armed: a host that keeps the device goes on
	}
	return out
}

// readFlag maps a flag copy, which waits for the work that fills it.
func readFlag(rb *webgpu.Buffer) ([4]byte, error) {
	var v [4]byte
	if err := rb.MapAsync(context.Background(), webgpu.MapModeRead, 0, 4); err != nil {
		return v, err
	}
	err := rb.ReadMapped(0, v[:])
	if uerr := rb.Unmap(); err == nil {
		err = uerr
	}
	return v, err
}

// checkFlag reads the flag after a batched stream (its copy is g.read).
func (g *Guard) checkFlag() error {
	v, err := readFlag(g.read)
	if err != nil {
		return err
	}
	if v != [4]byte{} {
		return ErrTruncated
	}
	return nil
}

// armedStream rewrites a wire stream (DeviceBackend): the host's groups are
// set around each use of a rewritten pipeline and the flag's copy rides the
// stream's own submit. The flag is cleared first when anything was added.
func (g *Guard) armedStream(words []uint32, data []byte) ([]uint32, bool, error) {
	fid, err := webgpu.ReplayID(g.flag)
	if err != nil {
		return nil, false, err
	}
	rid, err := webgpu.ReplayID(g.read)
	if err != nil {
		return nil, false, err
	}
	copyFlag := []uint32{uint32(wire.CopyBufferToBuffer), fid, 0, 0, rid, 0, 0, 4, 0}
	g.mu.Lock()
	plan := g.plan
	g.mu.Unlock()
	words, inj, err := g.lb.inject(words, uint32(len(data)), plan, copyFlag)
	if err != nil || !inj {
		return words, false, err
	}
	var zero [4]byte
	return words, true, g.dev.Queue().WriteBuffer(g.flag, 0, zero[:])
}

// hostRefs are the objects a rewritten stream names besides the guest's.
func (g *Guard) hostRefs() []any { return []any{g.flagBG, g.emptyBG, g.flag, g.read} }
