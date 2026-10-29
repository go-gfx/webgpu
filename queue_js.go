//go:build js

package webgpu

import (
	"context"
	"syscall/js"
)

// WriteBuffer is queue.writeBuffer(buffer, offset, data). The buffer's
// COPY_DST usage, 4-byte alignment and the range are checked first: an
// out-of-range write is an error here, not a silent device error. The
// bytes are copied before WriteBuffer returns, so data may be reused.
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
	d := q.dev
	d.mu.Lock()
	defer d.mu.Unlock()
	u := d.stage.load(data)
	return try("writeBuffer", func() { q.v.Call("writeBuffer", b.v, offset, u, 0, len(data)) })
}

// WriteTexture is queue.writeTexture(destination, data, layout, size).
// The region must lie inside the destination's mip level and the layout
// inside data (the specification's "validating linear texture data"),
// both checked first.
func (q *Queue) WriteTexture(dst TexelCopyTextureInfo, data []byte, layout TexelCopyBufferLayout, size Extent3D) error {
	if err := checkTextureCopy("writeTexture", dst, layout, uint64(len(data)), size, true); err != nil {
		return err
	}
	if dst.Texture.isDead() {
		return ErrDestroyed
	}
	d := q.dev
	d.mu.Lock()
	defer d.mu.Unlock()
	// The staging array may be longer than data; the layout was checked
	// against len(data), so the browser reads no byte past it.
	var u js.Value
	if len(data) > 0 {
		u = d.stage.load(data)
	} else {
		u = jsUint8Array.New(0)
	}
	return try("writeTexture", func() {
		q.v.Call("writeTexture", js.ValueOf(dst.toJS()), u, js.ValueOf(layout.toJS()), js.ValueOf(size.toJS()))
	})
}

// Submit is queue.submit(commandBuffers).
func (q *Queue) Submit(cbs ...*CommandBuffer) error {
	a := make([]any, len(cbs))
	for i, c := range cbs {
		if c == nil {
			return bounds("submit", "command buffer %d is nil", i)
		}
		a[i] = c.v
	}
	return try("submit", func() { q.v.Call("submit", a) })
}

// OnSubmittedWorkDone waits until the GPU finished the work submitted so
// far (queue.onSubmittedWorkDone()).
func (q *Queue) OnSubmittedWorkDone(ctx context.Context) error {
	var p js.Value
	if err := try("onSubmittedWorkDone", func() { p = q.v.Call("onSubmittedWorkDone") }); err != nil {
		return err
	}
	_, err := await(ctx, "onSubmittedWorkDone", p)
	return err
}
