package webgpu

import (
	"sync"
	"sync/atomic"

	"github.com/go-gfx/webgpu/timeline"
)

// object is what every handle holds: the browser object (a js.Value under
// GOOS=js) and its label.
type object struct {
	v     jsValue
	label string
	id    uint32        // replay table slot, 0 until first recorded; guarded by registry.mu
	put   bool          // the slot holds the object; guarded by registry.mu
	ready atomic.Uint32 // id once put: the lock-free path of batchID
}

// Label returns the label the object was created with.
func (o *object) Label() string { return o.label }

// owned is an object a device accounts for: destroying it refunds its
// quota once.
type owned struct {
	object
	dev  *Device
	dead bool
	mu   sync.Mutex
}

func (o *owned) markDead() bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.dead {
		return false
	}
	o.dead = true
	return true
}

func (o *owned) isDead() bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.dead || (o.dev != nil && o.dev.isLost())
}

// Device is a GPUDevice. Its Quota (from DeviceDescriptor) bounds what it
// may hold.
type Device struct {
	object
	quota  *Quota
	queue  *Queue
	limits map[string]uint64
	lostCh chan struct{} // closed when the device is lost

	mu       sync.Mutex
	lost     bool
	lostInfo DeviceLostInfo
	released bool  // the quota was refunded in full
	memory   int64 // bytes this device charged
	pipes    int   // pipelines this device charged
	stage    staging
	asked    uint64 // highest submission an onSubmittedWorkDone is waiting for

	// tl numbers queue submissions and records their completion.
	tl timeline.Timeline
}

// Submitted is the number of the device's last queue submission (Queue
// Submit and Recorder.Submit), from 1.
func (d *Device) Submitted() uint64 { return d.tl.Submitted() }

// Completed is the highest submission known to be complete.
func (d *Device) Completed() uint64 { return d.tl.Completed() }

func newDevice(v jsValue, label string, q *Quota) *Device {
	return &Device{object: object{v: v, label: label}, quota: q, lostCh: make(chan struct{})}
}

func (d *Device) isLost() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.lost
}

// setLost records the loss once, refunds everything the device held (the
// browser freed it) and wakes Lost's receivers.
func (d *Device) setLost(info DeviceLostInfo) {
	d.mu.Lock()
	if d.lost {
		d.mu.Unlock()
		return
	}
	d.lost, d.lostInfo = true, info
	d.mu.Unlock()
	d.releaseAll()
	d.tl.Fail(ErrDestroyed)
	close(d.lostCh)
}

// Lost is closed when the device is lost (destroyed, or by the browser:
// a GPU reset, a watchdog). LostInfo then says why.
func (d *Device) Lost() <-chan struct{} { return d.lostCh }

// LostInfo is why the device was lost; zero while it is not.
func (d *Device) LostInfo() DeviceLostInfo {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.lostInfo
}

// charge charges the quota for memory bytes and pipes pipelines on
// behalf of this device.
func (d *Device) charge(memory int64, pipes int) error {
	if d.isLost() {
		return ErrDestroyed
	}
	if memory != 0 {
		if err := d.quota.chargeMemory(memory); err != nil {
			return err
		}
	}
	for i := 0; i < pipes; i++ {
		if err := d.quota.chargePipeline(); err != nil {
			d.quota.refundMemory(memory)
			for ; i > 0; i-- {
				d.quota.refundPipeline()
			}
			return err
		}
	}
	d.mu.Lock()
	d.memory += memory
	d.pipes += pipes
	d.mu.Unlock()
	return nil
}

// refund gives back what one object charged, unless the device already
// refunded everything.
func (d *Device) refund(memory int64, pipes int) {
	d.mu.Lock()
	if d.released {
		d.mu.Unlock()
		return
	}
	d.memory -= memory
	d.pipes -= pipes
	d.mu.Unlock()
	d.quota.refundMemory(memory)
	for i := 0; i < pipes; i++ {
		d.quota.refundPipeline()
	}
}

// releaseAll refunds everything the device still holds, once.
func (d *Device) releaseAll() {
	d.mu.Lock()
	if d.released {
		d.mu.Unlock()
		return
	}
	d.released = true
	m, p := d.memory, d.pipes
	d.memory, d.pipes = 0, 0
	d.mu.Unlock()
	d.quota.refundMemory(m)
	for i := 0; i < p; i++ {
		d.quota.refundPipeline()
	}
}

// Quota returns the device's quota (nil: none).
func (d *Device) Quota() *Quota { return d.quota }

// Limit returns one of the device's limits (e.g. "maxBufferSize") as the
// browser reported it, and whether it is known.
func (d *Device) Limit(name string) (uint64, bool) {
	v, ok := d.limits[name]
	return v, ok
}

// Queue is a GPUQueue.
type Queue struct {
	object
	dev *Device
}

// Buffer is a GPUBuffer.
type Buffer struct {
	owned
	size  uint64
	usage BufferUsage
}

// Size is the buffer's size in bytes.
func (b *Buffer) Size() uint64 { return b.size }

// Usage is the buffer's usage flags.
func (b *Buffer) Usage() BufferUsage { return b.usage }

// Texture is a GPUTexture.
type Texture struct {
	owned
	desc  TextureDescriptor
	bytes int64
	// canvas is set for a canvas's current texture: not owned, never
	// charged, not destroyable by the caller.
	canvas bool
}

// Descriptor returns the texture's descriptor, defaults filled in.
func (t *Texture) Descriptor() TextureDescriptor { return t.desc }

// Width, Height, DepthOrArrayLayers and Format describe the texture.
func (t *Texture) Width() uint32              { return t.desc.Size.Width }
func (t *Texture) Height() uint32             { return t.desc.Size.Height }
func (t *Texture) DepthOrArrayLayers() uint32 { return t.desc.Size.DepthOrArrayLayers }
func (t *Texture) Format() TextureFormat      { return t.desc.Format }

// TextureView is a GPUTextureView.
type TextureView struct {
	object
	tex *Texture
}

// Texture returns the viewed texture.
func (v *TextureView) Texture() *Texture { return v.tex }

// Sampler is a GPUSampler.
type Sampler struct{ object }

// ShaderModule is a GPUShaderModule.
type ShaderModule struct{ object }

// BindGroupLayout is a GPUBindGroupLayout.
type BindGroupLayout struct{ object }

// PipelineLayout is a GPUPipelineLayout.
type PipelineLayout struct{ object }

// BindGroup is a GPUBindGroup.
type BindGroup struct{ object }

// RenderPipeline is a GPURenderPipeline; Release refunds its quota.
type RenderPipeline struct{ owned }

// ComputePipeline is a GPUComputePipeline; Release refunds its quota.
type ComputePipeline struct{ owned }

// CommandBuffer is a GPUCommandBuffer.
type CommandBuffer struct{ object }

// RenderBundle is a GPURenderBundle.
type RenderBundle struct{ object }
