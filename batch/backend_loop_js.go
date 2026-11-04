//go:build js

package batch

import (
	"errors"
	"fmt"
	"time"

	"github.com/go-gfx/webgpu"
)

// SetLoopBudget turns the loop budget on for the objects this backend creates
// from now on (call it before the first Create). See LoopBudget and Guard.
func (b *DeviceBackend) SetLoopBudget(lb LoopBudget) error {
	g, err := NewGuard(b.dev, lb)
	if err != nil {
		return err
	}
	b.g = g
	return nil
}

// LoopBudgetStats reports the WGSL modules seen, rewritten and refused as
// unboundable, and the time spent in the parser and the rewriter.
func (b *DeviceBackend) LoopBudgetStats() (modules, rewritten, refused int, spent time.Duration) {
	if b.g == nil {
		return 0, 0, 0, 0
	}
	return b.g.Stats()
}

func (b *DeviceBackend) createShader(d webgpu.ShaderModuleDescriptor) (uint32, error) {
	m, err := b.g.Shader(d)
	if err != nil {
		return 0, err
	}
	return b.register(m)
}

func (b *DeviceBackend) register(o any) (uint32, error) {
	id, err := webgpu.ReplayID(o)
	if err != nil {
		return 0, err
	}
	b.objs[id] = o
	return id, nil
}

func (b *DeviceBackend) createLayout(d webgpu.PipelineLayoutDescriptor) (uint32, error) {
	if err := b.g.CheckLayout(d.BindGroupLayouts); err != nil {
		return 0, err
	}
	id, err := b.create(d)
	if err == nil {
		b.g.NoteLayout(b.objs[id].(*webgpu.PipelineLayout), d.BindGroupLayouts)
	}
	return id, err
}

// createCompute makes a compute pipeline: a rewritten module's pipeline gets
// the flag group in its layout and a plan to set the host's groups.
func (b *DeviceBackend) createCompute(d webgpu.ComputePipelineDescriptor) (uint32, error) {
	nd, inst, first, err := b.g.prepare(d)
	if err != nil {
		return 0, err
	}
	if inst && d.Layout == nil {
		// An auto layout's groups are the pipeline's own, which a stream
		// cannot name: the flag group could not be set. Give the pipeline an
		// explicit layout.
		b.g.count(func() { b.g.refused++ })
		return 0, fmt.Errorf("%w: a pipeline of a rewritten module needs an explicit layout (not \"auto\")", ErrUnbounded)
	}
	id, err := b.create(nd)
	if err != nil && inst && errors.Is(err, ErrInvalid) {
		// The rewrite or the padded layout is what the browser refuses (a
		// shader that already binds as many storage buffers as the device
		// allows has no room for the flag), or the guest's pipeline was
		// invalid anyway; the guest's own module is gone (it was rewritten),
		// so the host cannot tell which and refuses the pipeline as
		// unboundable, with the browser's message.
		b.g.count(func() { b.g.refused++ })
		return 0, fmt.Errorf("%w: the browser refused the loop-budget layout of a pipeline it accepts (%v)", ErrUnbounded, err)
	}
	if err != nil || !inst {
		return id, err
	}
	return id, b.g.planFor(id, first)
}

func (b *DeviceBackend) closeLoop() {
	if b.g != nil {
		b.g.Close()
	}
}
