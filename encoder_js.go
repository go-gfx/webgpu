//go:build js

package webgpu

import (
	"syscall/js"
)

// CommandEncoder is a GPUCommandEncoder.
type CommandEncoder struct {
	object
	dev *Device
}

// BeginRenderPass is encoder.beginRenderPass().
func (e *CommandEncoder) BeginRenderPass(desc RenderPassDescriptor) (*RenderPassEncoder, error) {
	m, err := desc.toJS()
	if err != nil {
		return nil, err
	}
	var v js.Value
	if err := try("beginRenderPass", func() { v = e.v.Call("beginRenderPass", js.ValueOf(m)) }); err != nil {
		return nil, err
	}
	p := &RenderPassEncoder{}
	p.v, p.label = v, desc.Label
	return p, nil
}

// BeginComputePass is encoder.beginComputePass().
func (e *CommandEncoder) BeginComputePass(label string) (*ComputePassEncoder, error) {
	m := map[string]any{}
	setStr(m, "label", label)
	var v js.Value
	if err := try("beginComputePass", func() { v = e.v.Call("beginComputePass", js.ValueOf(m)) }); err != nil {
		return nil, err
	}
	p := &ComputePassEncoder{}
	p.v, p.label = v, label
	return p, nil
}

// CopyBufferToBuffer is encoder.copyBufferToBuffer(), ranges checked.
func (e *CommandEncoder) CopyBufferToBuffer(src *Buffer, srcOffset uint64, dst *Buffer, dstOffset, size uint64) error {
	if err := checkCopyBufferToBuffer(src, srcOffset, dst, dstOffset, size); err != nil {
		return err
	}
	return try("copyBufferToBuffer", func() { e.v.Call("copyBufferToBuffer", src.v, srcOffset, dst.v, dstOffset, size) })
}

// CopyBufferToTexture is encoder.copyBufferToTexture(), the region and
// layout checked against the texture and the buffer.
func (e *CommandEncoder) CopyBufferToTexture(src TexelCopyBufferInfo, dst TexelCopyTextureInfo, size Extent3D) error {
	const op = "copyBufferToTexture"
	if src.Buffer == nil {
		return bounds(op, "nil buffer")
	}
	if err := checkUsage(op, src.Buffer, BufferUsageCopySrc, "COPY_SRC"); err != nil {
		return err
	}
	if src.Layout.BytesPerRow%256 != 0 {
		return bounds(op, "bytesPerRow %d is not a multiple of 256", src.Layout.BytesPerRow)
	}
	if err := checkTextureCopy(op, dst, src.Layout, src.Buffer.size, size, true); err != nil {
		return err
	}
	return try(op, func() {
		e.v.Call("copyBufferToTexture", js.ValueOf(src.toJS()), js.ValueOf(dst.toJS()), js.ValueOf(size.toJS()))
	})
}

// CopyTextureToBuffer is encoder.copyTextureToBuffer(), checked likewise.
func (e *CommandEncoder) CopyTextureToBuffer(src TexelCopyTextureInfo, dst TexelCopyBufferInfo, size Extent3D) error {
	const op = "copyTextureToBuffer"
	if dst.Buffer == nil {
		return bounds(op, "nil buffer")
	}
	if err := checkUsage(op, dst.Buffer, BufferUsageCopyDst, "COPY_DST"); err != nil {
		return err
	}
	if dst.Layout.BytesPerRow%256 != 0 {
		return bounds(op, "bytesPerRow %d is not a multiple of 256", dst.Layout.BytesPerRow)
	}
	if err := checkTextureCopy(op, src, dst.Layout, dst.Buffer.size, size, false); err != nil {
		return err
	}
	return try(op, func() {
		e.v.Call("copyTextureToBuffer", js.ValueOf(src.toJS()), js.ValueOf(dst.toJS()), js.ValueOf(size.toJS()))
	})
}

// CopyTextureToTexture is encoder.copyTextureToTexture(); the browser
// validates the regions.
func (e *CommandEncoder) CopyTextureToTexture(src, dst TexelCopyTextureInfo, size Extent3D) error {
	if src.Texture == nil || dst.Texture == nil {
		return bounds("copyTextureToTexture", "nil texture")
	}
	return try("copyTextureToTexture", func() {
		e.v.Call("copyTextureToTexture", js.ValueOf(src.toJS()), js.ValueOf(dst.toJS()), js.ValueOf(size.norm().toJS()))
	})
}

// ClearBuffer is encoder.clearBuffer(buffer, offset, size); size 0
// clears to the end.
func (e *CommandEncoder) ClearBuffer(b *Buffer, offset, size uint64) error {
	if b == nil {
		return bounds("clearBuffer", "nil buffer")
	}
	if size == 0 {
		if offset > b.size {
			return bounds("clearBuffer", "offset %d past the buffer's %d bytes", offset, b.size)
		}
		size = b.size - offset
	}
	if err := checkRange("clearBuffer", b, offset, size); err != nil {
		return err
	}
	return try("clearBuffer", func() { e.v.Call("clearBuffer", b.v, offset, size) })
}

// Finish is encoder.finish().
func (e *CommandEncoder) Finish() (*CommandBuffer, error) {
	var v js.Value
	if err := try("finish", func() { v = e.v.Call("finish") }); err != nil {
		return nil, err
	}
	return &CommandBuffer{object{v: v, label: e.label}}, nil
}

// renderCommands are the commands a render pass and a render bundle
// share (GPURenderCommandsMixin, GPUBindingCommandsMixin). They return no
// error, to keep the per-draw path lean: the first argument the package
// refuses is kept and returned by End or Finish, and the command is not
// sent.
type renderCommands struct {
	object
	err error
}

func (r *renderCommands) fail(err error) {
	if r.err == nil {
		r.err = err
	}
}

// Err returns the first refused command so far.
func (r *renderCommands) Err() error { return r.err }

// SetPipeline is setPipeline(pipeline).
func (r *renderCommands) SetPipeline(p *RenderPipeline) {
	if p == nil {
		r.fail(bounds("setPipeline", "nil pipeline"))
		return
	}
	r.v.Call("setPipeline", p.v)
}

// SetBindGroup is setBindGroup(index, group, dynamicOffsets).
func (r *renderCommands) SetBindGroup(index uint32, g *BindGroup, dynamicOffsets ...uint32) {
	setBindGroup(r.v, index, g, dynamicOffsets)
}

func setBindGroup(v js.Value, index uint32, g *BindGroup, offs []uint32) {
	var gv any // nil: unbind
	if g != nil {
		gv = g.v
	}
	if len(offs) == 0 {
		v.Call("setBindGroup", index, gv)
		return
	}
	a := make([]any, len(offs))
	for i, o := range offs {
		a[i] = o
	}
	v.Call("setBindGroup", index, gv, a)
}

// SetVertexBuffer is setVertexBuffer(slot, buffer, offset, size); size 0
// binds to the end of the buffer. The range is checked.
func (r *renderCommands) SetVertexBuffer(slot uint32, b *Buffer, offset, size uint64) {
	if err := bufferRange("setVertexBuffer", b, BufferUsageVertex, "VERTEX", offset, &size); err != nil {
		r.fail(err)
		return
	}
	r.v.Call("setVertexBuffer", slot, b.v, offset, size)
}

// SetIndexBuffer is setIndexBuffer(buffer, format, offset, size); size 0
// binds to the end of the buffer. The range is checked.
func (r *renderCommands) SetIndexBuffer(b *Buffer, format IndexFormat, offset, size uint64) {
	if err := bufferRange("setIndexBuffer", b, BufferUsageIndex, "INDEX", offset, &size); err != nil {
		r.fail(err)
		return
	}
	r.v.Call("setIndexBuffer", b.v, string(format), offset, size)
}

// Draw is draw(vertexCount, instanceCount, firstVertex, firstInstance).
func (r *renderCommands) Draw(vertexCount, instanceCount, firstVertex, firstInstance uint32) {
	r.v.Call("draw", vertexCount, instanceCount, firstVertex, firstInstance)
}

// DrawIndexed is drawIndexed(indexCount, instanceCount, firstIndex,
// baseVertex, firstInstance).
func (r *renderCommands) DrawIndexed(indexCount, instanceCount, firstIndex uint32, baseVertex int32, firstInstance uint32) {
	r.v.Call("drawIndexed", indexCount, instanceCount, firstIndex, baseVertex, firstInstance)
}

// DrawIndirect is drawIndirect(buffer, offset); the 16-byte record must
// lie in the buffer.
func (r *renderCommands) DrawIndirect(b *Buffer, offset uint64) {
	if err := checkIndirect("drawIndirect", b, BufferUsageIndirect, offset, 16); err != nil {
		r.fail(err)
		return
	}
	r.v.Call("drawIndirect", b.v, offset)
}

// DrawIndexedIndirect is drawIndexedIndirect(buffer, offset); the
// 20-byte record must lie in the buffer.
func (r *renderCommands) DrawIndexedIndirect(b *Buffer, offset uint64) {
	if err := checkIndirect("drawIndexedIndirect", b, BufferUsageIndirect, offset, 20); err != nil {
		r.fail(err)
		return
	}
	r.v.Call("drawIndexedIndirect", b.v, offset)
}

// RenderPassEncoder is a GPURenderPassEncoder.
type RenderPassEncoder struct{ renderCommands }

// SetViewport is setViewport(x, y, width, height, minDepth, maxDepth).
func (p *RenderPassEncoder) SetViewport(x, y, w, h, minDepth, maxDepth float32) {
	p.v.Call("setViewport", x, y, w, h, minDepth, maxDepth)
}

// SetScissorRect is setScissorRect(x, y, width, height).
func (p *RenderPassEncoder) SetScissorRect(x, y, w, h uint32) {
	p.v.Call("setScissorRect", x, y, w, h)
}

// SetBlendConstant is setBlendConstant(color).
func (p *RenderPassEncoder) SetBlendConstant(c Color) {
	p.v.Call("setBlendConstant", js.ValueOf(c.toJS()))
}

// SetStencilReference is setStencilReference(reference).
func (p *RenderPassEncoder) SetStencilReference(ref uint32) {
	p.v.Call("setStencilReference", ref)
}

// ExecuteBundles is executeBundles(bundles).
func (p *RenderPassEncoder) ExecuteBundles(bundles ...*RenderBundle) {
	a := make([]any, len(bundles))
	for i, b := range bundles {
		if b == nil {
			p.fail(bounds("executeBundles", "bundle %d is nil", i))
			return
		}
		a[i] = b.v
	}
	p.v.Call("executeBundles", a)
}

// End is end(). It returns the first command the package refused, if any,
// or the browser's exception.
func (p *RenderPassEncoder) End() error {
	if err := try("end", func() { p.v.Call("end") }); err != nil {
		return err
	}
	return p.err
}

// RenderBundleEncoder is a GPURenderBundleEncoder.
type RenderBundleEncoder struct{ renderCommands }

// Finish is finish(): the recorded bundle, or the first refused command.
func (e *RenderBundleEncoder) Finish(label string) (*RenderBundle, error) {
	if e.err != nil {
		return nil, e.err
	}
	m := map[string]any{}
	setStr(m, "label", label)
	var v js.Value
	if err := try("finish", func() { v = e.v.Call("finish", js.ValueOf(m)) }); err != nil {
		return nil, err
	}
	return &RenderBundle{object{v: v, label: label}}, nil
}

// ComputePassEncoder is a GPUComputePassEncoder.
type ComputePassEncoder struct {
	object
	err error
}

// SetPipeline is setPipeline(pipeline).
func (p *ComputePassEncoder) SetPipeline(cp *ComputePipeline) {
	if cp == nil {
		if p.err == nil {
			p.err = bounds("setPipeline", "nil pipeline")
		}
		return
	}
	p.v.Call("setPipeline", cp.v)
}

// SetBindGroup is setBindGroup(index, group, dynamicOffsets).
func (p *ComputePassEncoder) SetBindGroup(index uint32, g *BindGroup, dynamicOffsets ...uint32) {
	setBindGroup(p.v, index, g, dynamicOffsets)
}

// DispatchWorkgroups is dispatchWorkgroups(x, y, z).
func (p *ComputePassEncoder) DispatchWorkgroups(x, y, z uint32) {
	p.v.Call("dispatchWorkgroups", x, y, z)
}

// DispatchWorkgroupsIndirect is dispatchWorkgroupsIndirect(buffer,
// offset); the 12-byte record must lie in the buffer.
func (p *ComputePassEncoder) DispatchWorkgroupsIndirect(b *Buffer, offset uint64) {
	if err := checkIndirect("dispatchWorkgroupsIndirect", b, BufferUsageIndirect, offset, 12); err != nil {
		if p.err == nil {
			p.err = err
		}
		return
	}
	p.v.Call("dispatchWorkgroupsIndirect", b.v, offset)
}

// End is end().
func (p *ComputePassEncoder) End() error {
	if err := try("end", func() { p.v.Call("end") }); err != nil {
		return err
	}
	return p.err
}
