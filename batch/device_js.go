//go:build js

package batch

import (
	"context"
	"fmt"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
)

// DeviceBackend is a Reader on a browser device: objects are made from wire
// descriptors (webgpu.DecodeDescriptor), streams run with Device.Replay and
// are waited for, and reads are mapAsync, getMappedRange, copy, unmap. It
// is the one implementation both transports (go-virtio's gpudev, the
// WebAssembly host module) share. The device's Quota bounds the memory and
// pipelines its contexts create; create it with the grant of the tenant.
type DeviceBackend struct {
	dev  *webgpu.Device
	objs map[uint32]any // replay id -> handle
}

// NewDeviceBackend returns a backend whose objects live on dev. webgpu's
// replay.js must be installed (webgpu.InstallReplay).
func NewDeviceBackend(dev *webgpu.Device) *DeviceBackend {
	return &DeviceBackend{dev: dev, objs: map[uint32]any{}}
}

func (b *DeviceBackend) resolve(id uint32) (any, error) {
	if o, ok := b.objs[id]; ok {
		return o, nil
	}
	return nil, fmt.Errorf("batch: object %d", id)
}

// Create implements Backend.
func (b *DeviceBackend) Create(kind uint32, desc []byte) (uint32, error) {
	d, err := webgpu.DecodeDescriptor(wire.Kind(kind), desc, b.resolve)
	if err != nil {
		return 0, err
	}
	var o any
	switch d := d.(type) {
	case webgpu.BufferDescriptor:
		o, err = b.dev.CreateBuffer(d)
	case webgpu.TextureDescriptor:
		o, err = b.dev.CreateTexture(d)
	case webgpu.TextureViewCreation:
		o, err = d.Texture.CreateView(d.Descriptor)
	case webgpu.SamplerDescriptor:
		o, err = b.dev.CreateSampler(&d)
	case webgpu.ShaderModuleDescriptor:
		o, err = b.dev.CreateShaderModule(d)
	case webgpu.BindGroupLayoutDescriptor:
		o, err = b.dev.CreateBindGroupLayout(d)
	case webgpu.PipelineLayoutDescriptor:
		o, err = b.dev.CreatePipelineLayout(d)
	case webgpu.BindGroupDescriptor:
		o, err = b.dev.CreateBindGroup(d)
	case webgpu.RenderPipelineDescriptor:
		o, err = b.dev.CreateRenderPipeline(d)
	case webgpu.ComputePipelineDescriptor:
		o, err = b.dev.CreateComputePipeline(d)
	default:
		err = fmt.Errorf("batch: descriptor %T", d)
	}
	if err != nil {
		return 0, err
	}
	id, err := webgpu.ReplayID(o)
	if err != nil {
		return 0, err
	}
	b.objs[id] = o
	return id, nil
}

// Destroy implements Backend.
func (b *DeviceBackend) Destroy(id uint32) {
	switch o := b.objs[id].(type) {
	case *webgpu.Buffer:
		o.Destroy()
	case *webgpu.Texture:
		o.Destroy()
	case *webgpu.RenderPipeline:
		o.Release()
	case *webgpu.ComputePipeline:
		o.Release()
	}
	delete(b.objs, id)
}

// Replay implements Backend: the stream may name this backend's objects
// only, and Replay returns once the GPU has executed it.
func (b *DeviceBackend) Replay(words []uint32, data []byte) error {
	refs := make([]any, 0, len(b.objs))
	for _, o := range b.objs {
		switch o.(type) {
		case *webgpu.Buffer, *webgpu.Texture, *webgpu.TextureView, *webgpu.BindGroup, *webgpu.RenderPipeline, *webgpu.ComputePipeline:
			refs = append(refs, o)
		}
	}
	seq, err := b.dev.Replay(words, data, refs...)
	if err != nil {
		return err
	}
	return b.dev.WaitSubmitted(context.Background(), seq)
}

// Read implements Reader: GPUBuffer.mapAsync(READ), getMappedRange, copy,
// unmap, on a buffer of this backend.
func (b *DeviceBackend) Read(id uint32, offset uint64, dst []byte) error {
	buf, ok := b.objs[id].(*webgpu.Buffer)
	if !ok {
		return fmt.Errorf("batch: read of object %d, not a buffer", id)
	}
	ctx := context.Background()
	if err := buf.MapAsync(ctx, webgpu.MapModeRead, offset, uint64(len(dst))); err != nil {
		return err
	}
	err := buf.ReadMapped(offset, dst)
	if uerr := buf.Unmap(); err == nil {
		err = uerr
	}
	return err
}

// Objects returns the handles the backend holds (for inspection by tests).
func (b *DeviceBackend) Objects() []any {
	out := make([]any, 0, len(b.objs))
	for _, o := range b.objs {
		out = append(out, o)
	}
	return out
}

// Close implements Backend: it destroys what is left.
func (b *DeviceBackend) Close() {
	for id := range b.objs {
		b.Destroy(id)
	}
}
