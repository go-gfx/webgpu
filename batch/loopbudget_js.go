//go:build js

package batch

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
)

// loopState is a DeviceBackend's loop-budget machinery (see LoopBudget).
type loopState struct {
	lb                LoopBudget
	flag, read        *webgpu.Buffer // 4 bytes each: the host's flag, and its MAP_READ copy
	budget            *webgpu.Buffer // the budget values 2^0 ... 2^32-1, one per budgetStride
	flagBGL, emptyBGL *webgpu.BindGroupLayout
	flagBG, emptyBG   *webgpu.BindGroup

	instr   map[*webgpu.ShaderModule]bool                        // modules whose code was rewritten
	layouts map[*webgpu.PipelineLayout][]*webgpu.BindGroupLayout // a guest layout's own groups
	derived map[*webgpu.PipelineLayout]*webgpu.PipelineLayout    // its padded twin
	plan    map[uint32]*pipePlan                                 // compute pipeline replay id -> groups to set
	// Counters, for the host's report.
	instrumented, fallbacks int
	modules                 int           // modules seen (instrumented or not)
	spent                   time.Duration // time in LoopBudget.Instrument (parse, rewrite, print)
}

// SetLoopBudget turns the loop budget on for the objects this backend
// creates from now on (call it before the first Create). See LoopBudget.
func (b *DeviceBackend) SetLoopBudget(lb LoopBudget) error {
	if err := lb.Validate(); err != nil {
		return err
	}
	s := &loopState{lb: lb.withDefaults(),
		instr:   map[*webgpu.ShaderModule]bool{},
		layouts: map[*webgpu.PipelineLayout][]*webgpu.BindGroupLayout{},
		derived: map[*webgpu.PipelineLayout]*webgpu.PipelineLayout{},
		plan:    map[uint32]*pipePlan{}}
	var err error
	if s.flag, err = b.dev.CreateBuffer(webgpu.BufferDescriptor{Size: 4, Usage: webgpu.BufferUsageStorage | webgpu.BufferUsageCopySrc | webgpu.BufferUsageCopyDst}); err != nil {
		return err
	}
	if s.read, err = b.dev.CreateBuffer(webgpu.BufferDescriptor{Size: 4, Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst}); err != nil {
		return err
	}
	if s.budget, err = b.dev.CreateBuffer(webgpu.BufferDescriptor{Size: budgetSlots * budgetStride, Usage: webgpu.BufferUsageUniform | webgpu.BufferUsageCopyDst}); err != nil {
		return err
	}
	vals := make([]byte, budgetSlots*budgetStride)
	for k := range budgetSlots {
		v := uint32(math.MaxUint32)
		if k < 32 {
			v = 1 << k
		}
		binary.LittleEndian.PutUint32(vals[k*budgetStride:], v)
	}
	if err := b.dev.Queue().WriteBuffer(s.budget, 0, vals); err != nil {
		return err
	}
	if s.flagBGL, err = b.dev.CreateBindGroupLayout(webgpu.BindGroupLayoutDescriptor{Entries: []webgpu.BindGroupLayoutEntry{
		{Binding: 0, Visibility: webgpu.ShaderStageCompute, Buffer: &webgpu.BufferBindingLayout{Type: webgpu.BufferBindingTypeStorage}},
		{Binding: 1, Visibility: webgpu.ShaderStageCompute, Buffer: &webgpu.BufferBindingLayout{Type: webgpu.BufferBindingTypeUniform, HasDynamicOffset: true, MinBindingSize: 4}}}}); err != nil {
		return err
	}
	if s.emptyBGL, err = b.dev.CreateBindGroupLayout(webgpu.BindGroupLayoutDescriptor{}); err != nil {
		return err
	}
	if s.flagBG, err = b.dev.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: s.flagBGL, Entries: []webgpu.BindGroupEntry{
		{Binding: 0, Buffer: s.flag},
		{Binding: 1, Buffer: s.budget, Size: 4}}}); err != nil {
		return err
	}
	if s.emptyBG, err = b.dev.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: s.emptyBGL}); err != nil {
		return err
	}
	b.ls = s
	return nil
}

// LoopBudgetStats reports how many shader modules were seen, how many were
// rewritten, how many the browser refused after the rewrite (they were kept
// as the guest wrote them, bounded by the host's time budget only), and the
// time spent parsing, rewriting and printing them.
func (b *DeviceBackend) LoopBudgetStats() (modules, instrumented, fallbacks int, spent time.Duration) {
	if b.ls == nil {
		return 0, 0, 0, 0
	}
	return b.ls.modules, b.ls.instrumented, b.ls.fallbacks, b.ls.spent
}

// createShader makes a guest's shader module, rewritten when the loop budget
// is on. If the browser refuses the rewritten text and accepts the original,
// the original is kept (a defect of the rewrite must not refuse a legitimate
// shader); if it refuses both, the error is the original's, so its line and
// column are the guest's own.
func (b *DeviceBackend) createShader(d webgpu.ShaderModuleDescriptor) (uint32, error) {
	t0 := time.Now()
	code, inst, err := b.ls.lb.Instrument(d.Code)
	b.ls.spent += time.Since(t0)
	b.ls.modules++
	if err != nil {
		return 0, err
	}
	if !inst {
		return b.create(d)
	}
	rd := d
	rd.Code = code
	id, err := b.create(rd)
	if err == nil {
		b.ls.instr[b.objs[id].(*webgpu.ShaderModule)] = true
		b.ls.instrumented++
		return id, nil
	}
	if !errors.Is(err, ErrInvalid) {
		return 0, err
	}
	b.ls.fallbacks++
	return b.create(d)
}

// createLayout refuses a guest layout that reaches the reserved group and
// remembers the layout's groups.
func (b *DeviceBackend) createLayout(d webgpu.PipelineLayoutDescriptor) (uint32, error) {
	if uint32(len(d.BindGroupLayouts)) > b.ls.lb.Group {
		return 0, fmt.Errorf("%w: a pipeline layout with %d bind group layouts (at most %d)", ErrReserved, len(d.BindGroupLayouts), b.ls.lb.Group)
	}
	id, err := b.create(d)
	if err == nil {
		b.ls.layouts[b.objs[id].(*webgpu.PipelineLayout)] = d.BindGroupLayouts
	}
	return id, err
}

// createCompute makes a compute pipeline. If its module was rewritten, its
// layout gains the host's flag group (and empty groups up to it), and the
// groups are recorded to be set after every SetPipeline of it.
func (b *DeviceBackend) createCompute(d webgpu.ComputePipelineDescriptor) (uint32, error) {
	s := b.ls
	if !s.instr[d.Compute.Module] {
		return b.create(d)
	}
	g := s.lb.Group
	first := uint32(0) // first group the host sets
	if d.Layout != nil {
		own, ok := s.layouts[d.Layout]
		if !ok {
			return b.create(d)
		}
		first = uint32(len(own))
		pl, ok := s.derived[d.Layout]
		if !ok {
			bgls := append([]*webgpu.BindGroupLayout(nil), own...)
			for i := first; i < g; i++ {
				bgls = append(bgls, s.emptyBGL)
			}
			bgls = append(bgls, s.flagBGL)
			var err error
			if pl, err = b.dev.CreatePipelineLayout(webgpu.PipelineLayoutDescriptor{BindGroupLayouts: bgls}); err != nil {
				return 0, err
			}
			s.derived[d.Layout] = pl
		}
		d.Layout = pl
	}
	id, err := b.create(d)
	if err != nil {
		return 0, err
	}
	eid, err := webgpu.ReplayID(s.emptyBG)
	if err != nil {
		return 0, err
	}
	fid, err := webgpu.ReplayID(s.flagBG)
	if err != nil {
		return 0, err
	}
	p := &pipePlan{flag: fid}
	for i := first; i < g; i++ {
		p.empties = append(p.empties, binding{i, eid})
	}
	s.plan[id] = p
	return id, nil
}

// armed rewrites a stream: the host's groups are set after every
// SetPipeline of an instrumented pipeline. If it set any, the flag is cleared
// first and the stream must be checked with checkFlag after it ran.
func (b *DeviceBackend) armed(words []uint32, data []byte) ([]uint32, bool, error) {
	// The flag's copy to its MAP_READ twin rides the stream's own submit.
	fid, err := webgpu.ReplayID(b.ls.flag)
	if err != nil {
		return nil, false, err
	}
	rid, err := webgpu.ReplayID(b.ls.read)
	if err != nil {
		return nil, false, err
	}
	copyFlag := []uint32{uint32(wire.CopyBufferToBuffer), fid, 0, 0, rid, 0, 0, 4, 0}
	words, inj, err := b.ls.lb.inject(words, uint32(len(data)), b.ls.plan, copyFlag)
	if err != nil || !inj {
		return words, false, err
	}
	var zero [4]byte
	return words, true, b.dev.Queue().WriteBuffer(b.ls.flag, 0, zero[:])
}

// checkFlag reads the flag after a stream: ErrTruncated if a loop was cut.
func (b *DeviceBackend) checkFlag() error {
	s := b.ls
	ctx := context.Background()
	if err := s.read.MapAsync(ctx, webgpu.MapModeRead, 0, 4); err != nil { // after the stream's work
		return err
	}
	var v [4]byte
	err := s.read.ReadMapped(0, v[:])
	if uerr := s.read.Unmap(); err == nil {
		err = uerr
	}
	if err != nil {
		return err
	}
	if v != [4]byte{} {
		return ErrTruncated
	}
	return nil
}

// hostRefs are the objects a rewritten stream names besides the guest's.
func (b *DeviceBackend) hostRefs() []any {
	if b.ls == nil {
		return nil
	}
	return []any{b.ls.flagBG, b.ls.emptyBG, b.ls.flag, b.ls.read}
}

func (b *DeviceBackend) closeLoop() {
	if b.ls == nil {
		return
	}
	b.ls.flag.Destroy()
	b.ls.read.Destroy()
	b.ls.budget.Destroy()
}
