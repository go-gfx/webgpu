//go:build js

package webgpu

import (
	"context"
	"syscall/js"
)

// Queue returns the device's queue.
func (d *Device) Queue() *Queue { return d.queue }

// Destroy destroys the device and everything created from it, and refunds
// its quota. Lost is closed once the browser confirms.
func (d *Device) Destroy() {
	_ = try("destroy", func() { d.v.Call("destroy") })
	d.releaseAll()
}

// OnUncapturedError calls fn, on a new goroutine, for every WebGPU error
// no error scope caught. It returns a function that removes the handler.
func (d *Device) OnUncapturedError(fn func(error)) (remove func()) {
	h := js.FuncOf(func(_ js.Value, a []js.Value) any {
		e := gpuError(a[0].Get("error"))
		go fn(e)
		return nil
	})
	d.v.Call("addEventListener", "uncapturederror", h)
	return func() {
		d.v.Call("removeEventListener", "uncapturederror", h)
		h.Release()
	}
}

func gpuError(e js.Value) *GPUError {
	kind := "validation"
	switch str(e.Get("constructor").Get("name")) {
	case "GPUOutOfMemoryError":
		kind = "out-of-memory"
	case "GPUInternalError":
		kind = "internal"
	}
	return &GPUError{Kind: kind, Message: str(e.Get("message"))}
}

// PushErrorScope is device.pushErrorScope(filter).
func (d *Device) PushErrorScope(filter ErrorFilter) {
	d.v.Call("pushErrorScope", string(filter))
}

// PopErrorScope is device.popErrorScope(): nil if the scope caught
// nothing, a *GPUError if it did, another error if popping failed.
func (d *Device) PopErrorScope(ctx context.Context) error {
	var p js.Value
	if err := try("popErrorScope", func() { p = d.v.Call("popErrorScope") }); err != nil {
		return err
	}
	v, err := await(ctx, "popErrorScope", p)
	if err != nil {
		return err
	}
	if nullish(v) {
		return nil
	}
	return gpuError(v)
}

// CreateBuffer is device.createBuffer(). The size is charged to the quota
// first.
func (d *Device) CreateBuffer(desc BufferDescriptor) (*Buffer, error) {
	if desc.Size > 1<<53 {
		return nil, bounds("createBuffer", "size %d", desc.Size)
	}
	if err := d.charge(int64(desc.Size), 0); err != nil {
		return nil, err
	}
	var v js.Value
	if err := try("createBuffer", func() { v = d.v.Call("createBuffer", js.ValueOf(desc.toJS())) }); err != nil {
		d.refund(int64(desc.Size), 0)
		return nil, err
	}
	b := &Buffer{size: desc.Size, usage: desc.Usage}
	b.v, b.label, b.dev = v, desc.Label, d
	return b, nil
}

// CreateTexture is device.createTexture(). Its size (TextureBytes) is
// charged to the quota first.
func (d *Device) CreateTexture(desc TextureDescriptor) (*Texture, error) {
	desc = desc.norm()
	n, err := TextureBytes(desc)
	if err != nil {
		return nil, err
	}
	if err := d.charge(n, 0); err != nil {
		return nil, err
	}
	var v js.Value
	if err := try("createTexture", func() { v = d.v.Call("createTexture", js.ValueOf(desc.toJS())) }); err != nil {
		d.refund(n, 0)
		return nil, err
	}
	t := &Texture{desc: desc, bytes: n}
	t.v, t.label, t.dev = v, desc.Label, d
	return t, nil
}

// CreateSampler is device.createSampler().
func (d *Device) CreateSampler(desc *SamplerDescriptor) (*Sampler, error) {
	var v js.Value
	if err := try("createSampler", func() { v = d.v.Call("createSampler", js.ValueOf(desc.toJS())) }); err != nil {
		return nil, err
	}
	s := &Sampler{}
	s.v = v
	if desc != nil {
		s.label = desc.Label
	}
	return s, nil
}

// CreateShaderModule is device.createShaderModule() with WGSL source,
// bounded by the quota's ShaderBytes.
func (d *Device) CreateShaderModule(desc ShaderModuleDescriptor) (*ShaderModule, error) {
	if err := d.quota.checkShader(len(desc.Code)); err != nil {
		return nil, err
	}
	var v js.Value
	if err := try("createShaderModule", func() { v = d.v.Call("createShaderModule", js.ValueOf(desc.toJS())) }); err != nil {
		return nil, err
	}
	return &ShaderModule{object{v: v, label: desc.Label}}, nil
}

// CreateBindGroupLayout is device.createBindGroupLayout().
func (d *Device) CreateBindGroupLayout(desc BindGroupLayoutDescriptor) (*BindGroupLayout, error) {
	m, err := desc.toJS()
	if err != nil {
		return nil, err
	}
	var v js.Value
	if err := try("createBindGroupLayout", func() { v = d.v.Call("createBindGroupLayout", js.ValueOf(m)) }); err != nil {
		return nil, err
	}
	return &BindGroupLayout{object{v: v, label: desc.Label}}, nil
}

// CreatePipelineLayout is device.createPipelineLayout().
func (d *Device) CreatePipelineLayout(desc PipelineLayoutDescriptor) (*PipelineLayout, error) {
	m, err := desc.toJS()
	if err != nil {
		return nil, err
	}
	var v js.Value
	if err := try("createPipelineLayout", func() { v = d.v.Call("createPipelineLayout", js.ValueOf(m)) }); err != nil {
		return nil, err
	}
	return &PipelineLayout{object{v: v, label: desc.Label}}, nil
}

// CreateBindGroup is device.createBindGroup(). Buffer ranges are checked
// against the buffers' sizes first.
func (d *Device) CreateBindGroup(desc BindGroupDescriptor) (*BindGroup, error) {
	m, err := desc.toJS()
	if err != nil {
		return nil, err
	}
	var v js.Value
	if err := try("createBindGroup", func() { v = d.v.Call("createBindGroup", js.ValueOf(m)) }); err != nil {
		return nil, err
	}
	return &BindGroup{object{v: v, label: desc.Label}}, nil
}

// CreateRenderPipeline is device.createRenderPipeline(); it counts against
// the quota's Pipelines until Release.
func (d *Device) CreateRenderPipeline(desc RenderPipelineDescriptor) (*RenderPipeline, error) {
	m, err := desc.toJS()
	if err != nil {
		return nil, err
	}
	if err := d.charge(0, 1); err != nil {
		return nil, err
	}
	var v js.Value
	if err := try("createRenderPipeline", func() { v = d.v.Call("createRenderPipeline", js.ValueOf(m)) }); err != nil {
		d.refund(0, 1)
		return nil, err
	}
	p := &RenderPipeline{}
	p.v, p.label, p.dev = v, desc.Label, d
	return p, nil
}

// CreateRenderPipelineAsync is device.createRenderPipelineAsync(): the
// pipeline is compiled off the caller's critical path, and a shader
// error is an error here rather than on the device.
func (d *Device) CreateRenderPipelineAsync(ctx context.Context, desc RenderPipelineDescriptor) (*RenderPipeline, error) {
	m, err := desc.toJS()
	if err != nil {
		return nil, err
	}
	if err := d.charge(0, 1); err != nil {
		return nil, err
	}
	var pr js.Value
	if err := try("createRenderPipelineAsync", func() { pr = d.v.Call("createRenderPipelineAsync", js.ValueOf(m)) }); err != nil {
		d.refund(0, 1)
		return nil, err
	}
	v, err := await(ctx, "createRenderPipelineAsync", pr)
	if err != nil {
		d.refund(0, 1)
		return nil, err
	}
	p := &RenderPipeline{}
	p.v, p.label, p.dev = v, desc.Label, d
	return p, nil
}

// CreateComputePipeline is device.createComputePipeline(); it counts
// against the quota's Pipelines until Release.
func (d *Device) CreateComputePipeline(desc ComputePipelineDescriptor) (*ComputePipeline, error) {
	m, err := desc.toJS()
	if err != nil {
		return nil, err
	}
	if err := d.charge(0, 1); err != nil {
		return nil, err
	}
	var v js.Value
	if err := try("createComputePipeline", func() { v = d.v.Call("createComputePipeline", js.ValueOf(m)) }); err != nil {
		d.refund(0, 1)
		return nil, err
	}
	p := &ComputePipeline{}
	p.v, p.label, p.dev = v, desc.Label, d
	return p, nil
}

// CreateComputePipelineAsync is device.createComputePipelineAsync().
func (d *Device) CreateComputePipelineAsync(ctx context.Context, desc ComputePipelineDescriptor) (*ComputePipeline, error) {
	m, err := desc.toJS()
	if err != nil {
		return nil, err
	}
	if err := d.charge(0, 1); err != nil {
		return nil, err
	}
	var pr js.Value
	if err := try("createComputePipelineAsync", func() { pr = d.v.Call("createComputePipelineAsync", js.ValueOf(m)) }); err != nil {
		d.refund(0, 1)
		return nil, err
	}
	v, err := await(ctx, "createComputePipelineAsync", pr)
	if err != nil {
		d.refund(0, 1)
		return nil, err
	}
	p := &ComputePipeline{}
	p.v, p.label, p.dev = v, desc.Label, d
	return p, nil
}

// CreateCommandEncoder is device.createCommandEncoder().
func (d *Device) CreateCommandEncoder(label string) (*CommandEncoder, error) {
	if d.isLost() {
		return nil, ErrDestroyed
	}
	m := map[string]any{}
	setStr(m, "label", label)
	var v js.Value
	if err := try("createCommandEncoder", func() { v = d.v.Call("createCommandEncoder", js.ValueOf(m)) }); err != nil {
		return nil, err
	}
	return &CommandEncoder{object: object{v: v, label: label}, dev: d}, nil
}

// CreateRenderBundleEncoder is device.createRenderBundleEncoder(): record
// draws once, replay them every frame with ExecuteBundles.
func (d *Device) CreateRenderBundleEncoder(desc RenderBundleEncoderDescriptor) (*RenderBundleEncoder, error) {
	var v js.Value
	if err := try("createRenderBundleEncoder", func() { v = d.v.Call("createRenderBundleEncoder", js.ValueOf(desc.toJS())) }); err != nil {
		return nil, err
	}
	e := &RenderBundleEncoder{}
	e.v, e.label = v, desc.Label
	return e, nil
}

// Destroy is buffer.destroy(); it refunds the buffer's quota once.
func (b *Buffer) Destroy() {
	if !b.markDead() {
		return
	}
	_ = try("destroy", func() { b.v.Call("destroy") })
	b.dev.refund(int64(b.size), 0)
}

// MapAsync is buffer.mapAsync(mode, offset, size); size 0 maps to the end.
func (b *Buffer) MapAsync(ctx context.Context, mode MapMode, offset, size uint64) error {
	if b.isDead() {
		return ErrDestroyed
	}
	if size == 0 {
		if offset > b.size {
			return bounds("mapAsync", "offset %d past the buffer's %d bytes", offset, b.size)
		}
		size = b.size - offset
	}
	if err := checkRange("mapAsync", b, offset, size); err != nil {
		return err
	}
	var p js.Value
	if err := try("mapAsync", func() { p = b.v.Call("mapAsync", uint32(mode), offset, size) }); err != nil {
		return err
	}
	_, err := await(ctx, "mapAsync", p)
	return err
}

// ReadMapped copies len(dst) bytes of the mapped range starting at offset
// into dst.
func (b *Buffer) ReadMapped(offset uint64, dst []byte) error {
	if err := checkRange("getMappedRange", b, offset, uint64(len(dst))); err != nil {
		return err
	}
	return try("getMappedRange", func() {
		u := jsUint8Array.New(b.v.Call("getMappedRange", offset, len(dst)))
		js.CopyBytesToGo(dst, u)
	})
}

// WriteMapped copies src into the mapped range starting at offset.
func (b *Buffer) WriteMapped(offset uint64, src []byte) error {
	if err := checkRange("getMappedRange", b, offset, uint64(len(src))); err != nil {
		return err
	}
	return try("getMappedRange", func() {
		u := jsUint8Array.New(b.v.Call("getMappedRange", offset, len(src)))
		js.CopyBytesToJS(u, src)
	})
}

// Unmap is buffer.unmap().
func (b *Buffer) Unmap() error { return try("unmap", func() { b.v.Call("unmap") }) }

// Destroy is texture.destroy(); it refunds the texture's quota once. A
// canvas texture is not the caller's to destroy and is left alone.
func (t *Texture) Destroy() {
	if t.canvas || !t.markDead() {
		return
	}
	_ = try("destroy", func() { t.v.Call("destroy") })
	t.dev.refund(t.bytes, 0)
}

// CreateView is texture.createView(); nil desc is the default view.
func (t *Texture) CreateView(desc *TextureViewDescriptor) (*TextureView, error) {
	var v js.Value
	if err := try("createView", func() { v = t.v.Call("createView", js.ValueOf(desc.toJS())) }); err != nil {
		return nil, err
	}
	tv := &TextureView{tex: t}
	tv.v = v
	if desc != nil {
		tv.label = desc.Label
	}
	return tv, nil
}

// GetBindGroupLayout is pipeline.getBindGroupLayout(index), for "auto"
// layouts.
func (p *RenderPipeline) GetBindGroupLayout(index uint32) (*BindGroupLayout, error) {
	var v js.Value
	if err := try("getBindGroupLayout", func() { v = p.v.Call("getBindGroupLayout", index) }); err != nil {
		return nil, err
	}
	return &BindGroupLayout{object{v: v}}, nil
}

// Release gives the pipeline's quota back; the pipeline must not be used
// after. (WebGPU pipelines have no destroy; the browser frees them when
// unreferenced.)
func (p *RenderPipeline) Release() {
	if p.markDead() {
		p.dev.refund(0, 1)
	}
}

// GetBindGroupLayout is pipeline.getBindGroupLayout(index).
func (p *ComputePipeline) GetBindGroupLayout(index uint32) (*BindGroupLayout, error) {
	var v js.Value
	if err := try("getBindGroupLayout", func() { v = p.v.Call("getBindGroupLayout", index) }); err != nil {
		return nil, err
	}
	return &BindGroupLayout{object{v: v}}, nil
}

// Release gives the pipeline's quota back.
func (p *ComputePipeline) Release() {
	if p.markDead() {
		p.dev.refund(0, 1)
	}
}
