//go:build !js

package webgpu

import (
	"context"
	"errors"
)

// The remote device: outside the browser (a VM guest, a native program)
// a Device created by NewRemoteDevice implements the creation calls and
// Recorder.Submit by sending wire descriptors and streams to a
// RemoteSink. Checks and quotas are the browser path's: a descriptor is
// validated (toJS) and charged before it is sent, so a remote device
// refuses at the call what a browser device refuses.
//
// It covers object creation, destruction, Queue.WriteBuffer and the
// Recorder. Mapping buffers (readback), Queue.WriteTexture, command
// encoders and render bundle encoders return ErrRemoteUnsupported.

// ErrNotRemote: a GPU call outside the browser on a Device that is not a
// remote device.
var ErrNotRemote = errors.New("webgpu: outside the browser, only a remote device (NewRemoteDevice) runs GPU work")

// ErrRemoteUnsupported: a call a remote device does not carry (yet).
var ErrRemoteUnsupported = errors.New("webgpu: not supported by a remote device")

// NewRemoteDevice returns a Device whose work goes to sink, bounded by q
// (nil: no quota).
func NewRemoteDevice(sink RemoteSink, label string, q *Quota) *Device {
	d := newDevice(jsValue{}, label, q)
	d.remote = sink
	d.queue = &Queue{dev: d}
	return d
}

// Queue returns the device's queue.
func (d *Device) Queue() *Queue { return d.queue }

func (d *Device) send(o *object, desc any) error {
	if d.remote == nil {
		return ErrNotRemote
	}
	if d.isLost() {
		return ErrDestroyed
	}
	kind, b, err := EncodeDescriptor(desc)
	if err != nil {
		return err
	}
	return d.remote.Create(idOf(o), kind, b)
}

// lose records that the remote device failed: everything is refunded and
// waiters are released with ErrDestroyed.
func (d *Device) lose(err error) {
	d.setLost(DeviceLostInfo{Reason: "remote", Message: err.Error()})
}

// CreateBuffer sends a buffer's creation; its size is charged first.
func (d *Device) CreateBuffer(desc BufferDescriptor) (*Buffer, error) {
	if desc.Size > 1<<53 {
		return nil, bounds("createBuffer", "size %d", desc.Size)
	}
	if err := d.charge(int64(desc.Size), 0); err != nil {
		return nil, err
	}
	b := &Buffer{size: desc.Size, usage: desc.Usage}
	b.label, b.dev = desc.Label, d
	if err := d.send(&b.object, desc); err != nil {
		d.refund(int64(desc.Size), 0)
		return nil, err
	}
	return b, nil
}

// CreateTexture sends a texture's creation; TextureBytes is charged first.
func (d *Device) CreateTexture(desc TextureDescriptor) (*Texture, error) {
	desc = desc.norm()
	n, err := TextureBytes(desc)
	if err != nil {
		return nil, err
	}
	if err := d.charge(n, 0); err != nil {
		return nil, err
	}
	t := &Texture{desc: desc, bytes: n}
	t.label, t.dev = desc.Label, d
	if err := d.send(&t.object, desc); err != nil {
		d.refund(n, 0)
		return nil, err
	}
	return t, nil
}

// CreateView sends a view's creation; nil desc is the default view.
func (t *Texture) CreateView(desc *TextureViewDescriptor) (*TextureView, error) {
	tv := &TextureView{tex: t}
	if desc != nil {
		tv.label = desc.Label
	}
	if t.dev == nil {
		return nil, ErrNotRemote
	}
	if err := t.dev.send(&tv.object, TextureViewCreation{Texture: t, Descriptor: desc}); err != nil {
		return nil, err
	}
	return tv, nil
}

// CreateSampler sends a sampler's creation.
func (d *Device) CreateSampler(desc *SamplerDescriptor) (*Sampler, error) {
	s := &Sampler{}
	var sd SamplerDescriptor
	if desc != nil {
		sd, s.label = *desc, desc.Label
	}
	if err := d.send(&s.object, sd); err != nil {
		return nil, err
	}
	return s, nil
}

// CreateShaderModule sends WGSL source, bounded by the quota's ShaderBytes.
func (d *Device) CreateShaderModule(desc ShaderModuleDescriptor) (*ShaderModule, error) {
	if err := d.quota.checkShader(len(desc.Code)); err != nil {
		return nil, err
	}
	m := &ShaderModule{object{label: desc.Label}}
	if err := d.send(&m.object, desc); err != nil {
		return nil, err
	}
	return m, nil
}

// CreateBindGroupLayout sends a bind group layout's creation, checked as
// in the browser.
func (d *Device) CreateBindGroupLayout(desc BindGroupLayoutDescriptor) (*BindGroupLayout, error) {
	if _, err := desc.toJS(); err != nil {
		return nil, err
	}
	l := &BindGroupLayout{object{label: desc.Label}}
	if err := d.send(&l.object, desc); err != nil {
		return nil, err
	}
	return l, nil
}

// CreatePipelineLayout sends a pipeline layout's creation.
func (d *Device) CreatePipelineLayout(desc PipelineLayoutDescriptor) (*PipelineLayout, error) {
	if _, err := desc.toJS(); err != nil {
		return nil, err
	}
	l := &PipelineLayout{object{label: desc.Label}}
	if err := d.send(&l.object, desc); err != nil {
		return nil, err
	}
	return l, nil
}

// CreateBindGroup sends a bind group's creation; buffer ranges are
// checked against the buffers' sizes first, as in the browser.
func (d *Device) CreateBindGroup(desc BindGroupDescriptor) (*BindGroup, error) {
	if _, err := desc.toJS(); err != nil {
		return nil, err
	}
	g := &BindGroup{object{label: desc.Label}}
	if err := d.send(&g.object, desc); err != nil {
		return nil, err
	}
	return g, nil
}

// CreateRenderPipeline sends a render pipeline's creation; it counts
// against the quota's Pipelines until Release.
func (d *Device) CreateRenderPipeline(desc RenderPipelineDescriptor) (*RenderPipeline, error) {
	if _, err := desc.toJS(); err != nil {
		return nil, err
	}
	if err := d.charge(0, 1); err != nil {
		return nil, err
	}
	p := &RenderPipeline{}
	p.label, p.dev = desc.Label, d
	if err := d.send(&p.object, desc); err != nil {
		d.refund(0, 1)
		return nil, err
	}
	return p, nil
}

// CreateComputePipeline sends a compute pipeline's creation; it counts
// against the quota's Pipelines until Release.
func (d *Device) CreateComputePipeline(desc ComputePipelineDescriptor) (*ComputePipeline, error) {
	if _, err := desc.toJS(); err != nil {
		return nil, err
	}
	if err := d.charge(0, 1); err != nil {
		return nil, err
	}
	p := &ComputePipeline{}
	p.label, p.dev = desc.Label, d
	if err := d.send(&p.object, desc); err != nil {
		d.refund(0, 1)
		return nil, err
	}
	return p, nil
}

func (d *Device) destroy(o *object) {
	if d != nil && d.remote != nil {
		_ = d.remote.Destroy(idOf(o))
	}
}

// Destroy releases the buffer remotely and refunds its quota once.
func (b *Buffer) Destroy() {
	if b.markDead() {
		b.dev.destroy(&b.object)
		b.dev.refund(int64(b.size), 0)
	}
}

// Destroy releases the texture remotely and refunds its quota once.
func (t *Texture) Destroy() {
	if t.markDead() {
		t.dev.destroy(&t.object)
		t.dev.refund(t.bytes, 0)
	}
}

// Release releases the pipeline remotely and gives its quota back.
func (p *RenderPipeline) Release() {
	if p.markDead() {
		p.dev.destroy(&p.object)
		p.dev.refund(0, 1)
	}
}

// Release releases the pipeline remotely and gives its quota back.
func (p *ComputePipeline) Release() {
	if p.markDead() {
		p.dev.destroy(&p.object)
		p.dev.refund(0, 1)
	}
}

// MapAsync: readback does not cross to the host yet.
func (b *Buffer) MapAsync(context.Context, MapMode, uint64, uint64) error {
	return ErrRemoteUnsupported
}

// WriteBuffer sends queue.writeBuffer as a one-command submit, checked as
// in the browser.
func (q *Queue) WriteBuffer(b *Buffer, offset uint64, data []byte) error {
	if err := checkWriteBuffer(b, offset, len(data)); err != nil {
		return err
	}
	if b.isDead() {
		return ErrDestroyed
	}
	if len(data) == 0 {
		return nil
	}
	r := q.dev.NewRecorder()
	r.WriteBuffer(b, offset, data)
	_, err := r.Submit()
	return err
}

// WriteTexture: not carried by a remote device yet.
func (q *Queue) WriteTexture(TexelCopyTextureInfo, []byte, TexelCopyBufferLayout, Extent3D) error {
	return ErrRemoteUnsupported
}

// Submit sends everything recorded as one submit and resets the
// Recorder. It returns once the sink reports the submit executed; a sink
// error loses the device.
func (r *Recorder) Submit() (seq uint64, err error) {
	defer r.Reset()
	if err := r.finish(); err != nil {
		return 0, err
	}
	d := r.dev
	if d.remote == nil {
		return 0, ErrNotRemote
	}
	if d.isLost() {
		return 0, ErrDestroyed
	}
	seq = d.tl.Next()
	if err := d.remote.Submit(r.words, r.data); err != nil {
		d.lose(err)
		return 0, err
	}
	d.tl.Signal(seq)
	return seq, r.err
}

// WaitSubmitted waits until submission seq is complete (a remote device's
// submits complete when Submit returns).
func (d *Device) WaitSubmitted(ctx context.Context, seq uint64) error {
	if seq > d.tl.Submitted() {
		return bounds("WaitSubmitted", "submission %d was never made (last %d)", seq, d.tl.Submitted())
	}
	return d.tl.Wait(ctx, seq)
}
