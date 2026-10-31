package webgpu

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
)

// Batched encoding. A Recorder records a whole submit (queue writes,
// render and compute passes) as uint32 words in Go; Submit hands them to
// replay.js in one call, which issues the WebGPU calls from JavaScript.
// One syscall/js call costs about ten times the WebGPU call it carries,
// so a frame of many draws is mostly call overhead without this.
//
// The stream format is package wire's (opcodes, layout, the Go reader);
// this file is its only writer, replay.js and wire.Decode its readers.

// Codes for enumerations in the stream; 0 is "not given".
var (
	loadOpCode     = map[LoadOp]uint32{LoadOpLoad: 1, LoadOpClear: 2}
	storeOpCode    = map[StoreOp]uint32{StoreOpStore: 1, StoreOpDiscard: 2}
	indexFormatCod = map[IndexFormat]uint32{IndexFormatUint16: 1, IndexFormatUint32: 2}
)

// The object registry gives every handle used in a stream a small id,
// the index of the browser object in replay.js's table. An id is freed
// (and its table slot cleared) when the handle is garbage collected; a
// Recorder keeps the handles it recorded reachable until Submit returns,
// so a recorded id always names the object it was recorded for.
var registry struct {
	mu   sync.Mutex
	next uint32
	free []uint32
	// drop clears a slot of replay.js's table. Set under GOOS=js; tests
	// set their own.
	drop func(id uint32)
}

// batchID returns the object's id, allocating one on first use, and
// whether the object still has to be put in replay.js's table (Submit
// does it, before replaying).
func (o *object) batchID() (id uint32, unput bool) {
	// The per-draw path: an object already in the table, no lock.
	if id := o.ready.Load(); id != 0 {
		return id, false
	}
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if o.id != 0 {
		return o.id, !o.put
	}
	if n := len(registry.free); n > 0 {
		id = registry.free[n-1]
		registry.free = registry.free[:n-1]
	} else {
		registry.next++
		id = registry.next
	}
	o.id = id
	runtime.AddCleanup(o, freeID, id)
	return id, true
}

// markPut records that the object is in replay.js's table; it reports
// whether it was not yet.
func (o *object) markPut() bool {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if o.put {
		return false
	}
	o.put = true
	o.ready.Store(o.id)
	return true
}

func freeID(id uint32) {
	registry.mu.Lock()
	drop := registry.drop
	registry.mu.Unlock()
	if drop != nil {
		drop(id)
	}
	registry.mu.Lock()
	registry.free = append(registry.free, id)
	registry.mu.Unlock()
}

// ErrRecorderState: a command outside the pass it belongs to, a pass
// left open at Submit, or a Recorder used after Submit.
var ErrRecorderState = errors.New("webgpu: recorder: command out of order")

// Recorder records one submit. Its methods mirror the direct API; like a
// pass encoder's, they return nothing on the per-draw path: the first
// refused command is kept, not recorded, and returned by End or Submit.
// A Recorder is not safe for concurrent use; Reset reuses its memory for
// the next frame.
type Recorder struct {
	dev   *Device
	words []uint32
	data  []byte
	refs  []*object // keeps recorded objects (and so their ids) alive
	unput []*object // objects Submit must put in replay.js's table first
	err   error
	pass  int // 0 none, 1 render, 2 compute
	stage recorderStage
}

// NewRecorder returns a Recorder for the device.
func (d *Device) NewRecorder() *Recorder { return &Recorder{dev: d} }

// Reset forgets everything recorded, keeping the memory.
func (r *Recorder) Reset() {
	r.words, r.data = r.words[:0], r.data[:0]
	clear(r.refs)
	r.refs = r.refs[:0]
	clear(r.unput)
	r.unput = r.unput[:0]
	r.err, r.pass = nil, 0
}

// Err is the first refused command so far.
func (r *Recorder) Err() error { return r.err }

// Len is the number of words recorded, for tests and metrics.
func (r *Recorder) Len() int { return len(r.words) }

func (r *Recorder) fail(err error) {
	if r.err == nil {
		r.err = err
	}
}

func (r *Recorder) id(o *object) uint32 {
	r.refs = append(r.refs, o)
	id, unput := o.batchID()
	if unput {
		r.unput = append(r.unput, o)
	}
	return id
}

func lohi(v uint64) (uint32, uint32) { return uint32(v), uint32(v >> 32) }

func (r *Recorder) in(want int, op string) bool {
	if r.pass != want {
		r.fail(fmt.Errorf("%w: %s", ErrRecorderState, op))
		return false
	}
	return true
}

// WriteBuffer records queue.writeBuffer: the bytes are copied into the
// recorder now and written, in order, when Submit replays the stream.
// Checked like Queue.WriteBuffer.
func (r *Recorder) WriteBuffer(b *Buffer, offset uint64, data []byte) {
	if !r.in(0, "writeBuffer inside a pass") {
		return
	}
	if err := checkWriteBuffer(b, offset, len(data)); err != nil {
		r.fail(err)
		return
	}
	if len(data) == 0 {
		return
	}
	if !dataFits(len(r.data), len(data)) {
		r.fail(bounds("writeBuffer", "more than 4 GiB recorded in one submit"))
		return
	}
	lo, hi := lohi(offset)
	r.words = append(r.words, uint32(opWriteBuffer), r.id(&b.object), lo, hi, uint32(len(r.data)), uint32(len(data)))
	r.data = append(r.data, data...)
}

// CopyBufferToBuffer records encoder.copyBufferToBuffer, checked.
func (r *Recorder) CopyBufferToBuffer(src *Buffer, srcOffset uint64, dst *Buffer, dstOffset, size uint64) {
	if !r.in(0, "copyBufferToBuffer inside a pass") {
		return
	}
	if err := checkCopyBufferToBuffer(src, srcOffset, dst, dstOffset, size); err != nil {
		r.fail(err)
		return
	}
	so, sh := lohi(srcOffset)
	do, dh := lohi(dstOffset)
	zl, zh := lohi(size)
	r.words = append(r.words, uint32(opCopyBufferToBuffer), r.id(&src.object), so, sh, r.id(&dst.object), do, dh, zl, zh)
}

// aspectCode is a TextureAspect in the stream; 0 is "not given".
var aspectCode = map[TextureAspect]uint32{"": 0, TextureAspectAll: 1, TextureAspectStencilOnly: 2, TextureAspectDepthOnly: 3}

// CopyTextureToTexture records encoder.copyTextureToTexture. As on the
// direct path, the browser validates the regions; the textures must not
// be nil and the aspects must be WebGPU's.
func (r *Recorder) CopyTextureToTexture(src, dst TexelCopyTextureInfo, size Extent3D) {
	const op = "copyTextureToTexture"
	if !r.in(0, op+" inside a pass") {
		return
	}
	if src.Texture == nil || dst.Texture == nil {
		r.fail(bounds(op, "nil texture"))
		return
	}
	sa, ok1 := aspectCode[src.Aspect]
	da, ok2 := aspectCode[dst.Aspect]
	if !ok1 || !ok2 {
		r.fail(bounds(op, "aspect %q or %q", src.Aspect, dst.Aspect))
		return
	}
	size = size.norm()
	r.words = append(r.words, uint32(opCopyTextureToTexture),
		r.id(&src.Texture.object), src.MipLevel, src.Origin.X, src.Origin.Y, src.Origin.Z, sa,
		r.id(&dst.Texture.object), dst.MipLevel, dst.Origin.X, dst.Origin.Y, dst.Origin.Z, da,
		size.Width, size.Height, size.DepthOrArrayLayers)
}

// maxData bounds the data stream: its offsets and lengths are uint32
// words. A variable so a test can lower it.
var maxData uint64 = math.MaxUint32

func dataFits(have, more int) bool { return uint64(have)+uint64(more) <= maxData }

func f32(v float64) uint32 { return math.Float32bits(float32(v)) }

// BeginRenderPass records encoder.beginRenderPass and returns the pass
// recorder. Up to 8 color attachments.
func (r *Recorder) BeginRenderPass(desc RenderPassDescriptor) *RenderPassRecorder {
	p := &RenderPassRecorder{r: r}
	if !r.in(0, "beginRenderPass inside a pass") {
		return p
	}
	if len(desc.ColorAttachments) > 8 {
		r.fail(bounds("beginRenderPass", "%d color attachments", len(desc.ColorAttachments)))
		return p
	}
	if _, err := desc.toJS(); err != nil { // the same checks as the direct path
		r.fail(err)
		return p
	}
	w := append(r.words, uint32(opBeginRenderPass), uint32(len(desc.ColorAttachments)))
	for _, c := range desc.ColorAttachments {
		if c == nil {
			w = append(w, 0, 0, 0, 0, 0, 0, 0, 0)
			continue
		}
		var resolve uint32
		if c.ResolveTarget != nil {
			resolve = r.id(&c.ResolveTarget.object)
		}
		w = append(w, r.id(&c.View.object), resolve, loadOpCode[c.LoadOp], storeOpCode[c.StoreOp],
			f32(c.ClearValue.R), f32(c.ClearValue.G), f32(c.ClearValue.B), f32(c.ClearValue.A))
	}
	if ds := desc.DepthStencilAttachment; ds != nil {
		w = append(w, 1, r.id(&ds.View.object), loadOpCode[ds.DepthLoadOp], storeOpCode[ds.DepthStoreOp],
			math.Float32bits(ds.DepthClearValue), b2u(ds.DepthReadOnly),
			loadOpCode[ds.StencilLoadOp], storeOpCode[ds.StencilStoreOp], ds.StencilClearValue, b2u(ds.StencilReadOnly))
	} else {
		w = append(w, 0)
	}
	r.words = w
	r.pass = 1
	p.open = true
	return p
}

func b2u(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}

// BeginComputePass records encoder.beginComputePass.
func (r *Recorder) BeginComputePass() *ComputePassRecorder {
	p := &ComputePassRecorder{r: r}
	if !r.in(0, "beginComputePass inside a pass") {
		return p
	}
	r.words = append(r.words, uint32(opBeginComputePass))
	r.pass = 2
	p.open = true
	return p
}

// finish appends the submit and reports whether the stream is complete.
func (r *Recorder) finish() error {
	if r.pass != 0 {
		r.fail(fmt.Errorf("%w: a pass is still open at Submit", ErrRecorderState))
		r.pass = 0
		return r.err
	}
	r.words = append(r.words, uint32(opSubmit))
	return nil
}

// RenderPassRecorder records a render pass's commands.
type RenderPassRecorder struct {
	r    *Recorder
	open bool
}

func (p *RenderPassRecorder) ok(op string) bool {
	if !p.open {
		p.r.fail(fmt.Errorf("%w: %s after End", ErrRecorderState, op))
		return false
	}
	return true
}

// SetPipeline records setPipeline.
func (p *RenderPassRecorder) SetPipeline(pl *RenderPipeline) {
	if !p.ok("setPipeline") {
		return
	}
	if pl == nil {
		p.r.fail(bounds("setPipeline", "nil pipeline"))
		return
	}
	p.r.words = append(p.r.words, uint32(opSetPipeline), p.r.id(&pl.object))
}

// SetBindGroup records setBindGroup, dynamic offsets included (replayed
// through the Uint32Array overload, without an array per call).
func (p *RenderPassRecorder) SetBindGroup(index uint32, g *BindGroup, dynamicOffsets ...uint32) {
	if p.ok("setBindGroup") {
		p.r.setBindGroup(index, g, dynamicOffsets)
	}
}

func (r *Recorder) setBindGroup(index uint32, g *BindGroup, offs []uint32) {
	var id uint32
	if g != nil {
		id = r.id(&g.object)
	}
	r.words = append(r.words, uint32(opSetBindGroup), index, id, uint32(len(offs)))
	r.words = append(r.words, offs...)
}

// SetVertexBuffer records setVertexBuffer; size 0 binds to the end.
// Checked like the direct path.
func (p *RenderPassRecorder) SetVertexBuffer(slot uint32, b *Buffer, offset, size uint64) {
	if !p.ok("setVertexBuffer") {
		return
	}
	if err := bufferRange("setVertexBuffer", b, BufferUsageVertex, "VERTEX", offset, &size); err != nil {
		p.r.fail(err)
		return
	}
	ol, oh := lohi(offset)
	sl, sh := lohi(size)
	p.r.words = append(p.r.words, uint32(opSetVertexBuffer), slot, p.r.id(&b.object), ol, oh, sl, sh)
}

// SetIndexBuffer records setIndexBuffer; size 0 binds to the end.
func (p *RenderPassRecorder) SetIndexBuffer(b *Buffer, format IndexFormat, offset, size uint64) {
	if !p.ok("setIndexBuffer") {
		return
	}
	if err := bufferRange("setIndexBuffer", b, BufferUsageIndex, "INDEX", offset, &size); err != nil {
		p.r.fail(err)
		return
	}
	f, known := indexFormatCod[format]
	if !known {
		p.r.fail(bounds("setIndexBuffer", "index format %q", format))
		return
	}
	ol, oh := lohi(offset)
	sl, sh := lohi(size)
	p.r.words = append(p.r.words, uint32(opSetIndexBuffer), p.r.id(&b.object), f, ol, oh, sl, sh)
}

// Draw records draw.
func (p *RenderPassRecorder) Draw(vertexCount, instanceCount, firstVertex, firstInstance uint32) {
	if p.ok("draw") {
		p.r.words = append(p.r.words, uint32(opDraw), vertexCount, instanceCount, firstVertex, firstInstance)
	}
}

// DrawIndexed records drawIndexed.
func (p *RenderPassRecorder) DrawIndexed(indexCount, instanceCount, firstIndex uint32, baseVertex int32, firstInstance uint32) {
	if p.ok("drawIndexed") {
		p.r.words = append(p.r.words, uint32(opDrawIndexed), indexCount, instanceCount, firstIndex, uint32(baseVertex), firstInstance)
	}
}

// DrawIndirect records drawIndirect; the record must lie in the buffer.
func (p *RenderPassRecorder) DrawIndirect(b *Buffer, offset uint64) {
	p.indirect(opDrawIndirect, "drawIndirect", b, offset, 16)
}

// DrawIndexedIndirect records drawIndexedIndirect.
func (p *RenderPassRecorder) DrawIndexedIndirect(b *Buffer, offset uint64) {
	p.indirect(opDrawIndexedIndirect, "drawIndexedIndirect", b, offset, 20)
}

func (p *RenderPassRecorder) indirect(op opcode, name string, b *Buffer, offset, n uint64) {
	if !p.ok(name) {
		return
	}
	if err := checkIndirect(name, b, BufferUsageIndirect, offset, n); err != nil {
		p.r.fail(err)
		return
	}
	ol, oh := lohi(offset)
	p.r.words = append(p.r.words, uint32(op), p.r.id(&b.object), ol, oh)
}

// SetViewport records setViewport.
func (p *RenderPassRecorder) SetViewport(x, y, w, h, minDepth, maxDepth float32) {
	if p.ok("setViewport") {
		fb := math.Float32bits
		p.r.words = append(p.r.words, uint32(opSetViewport), fb(x), fb(y), fb(w), fb(h), fb(minDepth), fb(maxDepth))
	}
}

// SetScissorRect records setScissorRect.
func (p *RenderPassRecorder) SetScissorRect(x, y, w, h uint32) {
	if p.ok("setScissorRect") {
		p.r.words = append(p.r.words, uint32(opSetScissorRect), x, y, w, h)
	}
}

// SetBlendConstant records setBlendConstant (as float32).
func (p *RenderPassRecorder) SetBlendConstant(c Color) {
	if p.ok("setBlendConstant") {
		p.r.words = append(p.r.words, uint32(opSetBlendConstant), f32(c.R), f32(c.G), f32(c.B), f32(c.A))
	}
}

// SetStencilReference records setStencilReference.
func (p *RenderPassRecorder) SetStencilReference(ref uint32) {
	if p.ok("setStencilReference") {
		p.r.words = append(p.r.words, uint32(opSetStencilReference), ref)
	}
}

// ExecuteBundles records executeBundles.
func (p *RenderPassRecorder) ExecuteBundles(bundles ...*RenderBundle) {
	if !p.ok("executeBundles") {
		return
	}
	for i, b := range bundles {
		if b == nil {
			p.r.fail(bounds("executeBundles", "bundle %d is nil", i))
			return
		}
	}
	p.r.words = append(p.r.words, uint32(opExecuteBundles), uint32(len(bundles)))
	for _, b := range bundles {
		p.r.words = append(p.r.words, p.r.id(&b.object))
	}
}

// End records end. It returns the first refused command so far.
func (p *RenderPassRecorder) End() error {
	if !p.ok("end") {
		return p.r.err
	}
	p.open = false
	p.r.pass = 0
	p.r.words = append(p.r.words, uint32(opEndPass))
	return p.r.err
}

// ComputePassRecorder records a compute pass's commands.
type ComputePassRecorder struct {
	r    *Recorder
	open bool
}

func (p *ComputePassRecorder) ok(op string) bool {
	if !p.open {
		p.r.fail(fmt.Errorf("%w: %s after End", ErrRecorderState, op))
		return false
	}
	return true
}

// SetPipeline records setPipeline.
func (p *ComputePassRecorder) SetPipeline(pl *ComputePipeline) {
	if !p.ok("setPipeline") {
		return
	}
	if pl == nil {
		p.r.fail(bounds("setPipeline", "nil pipeline"))
		return
	}
	p.r.words = append(p.r.words, uint32(opSetPipeline), p.r.id(&pl.object))
}

// SetBindGroup records setBindGroup.
func (p *ComputePassRecorder) SetBindGroup(index uint32, g *BindGroup, dynamicOffsets ...uint32) {
	if p.ok("setBindGroup") {
		p.r.setBindGroup(index, g, dynamicOffsets)
	}
}

// DispatchWorkgroups records dispatchWorkgroups.
func (p *ComputePassRecorder) DispatchWorkgroups(x, y, z uint32) {
	if p.ok("dispatchWorkgroups") {
		p.r.words = append(p.r.words, uint32(opDispatch), x, y, z)
	}
}

// DispatchWorkgroupsIndirect records dispatchWorkgroupsIndirect.
func (p *ComputePassRecorder) DispatchWorkgroupsIndirect(b *Buffer, offset uint64) {
	if !p.ok("dispatchWorkgroupsIndirect") {
		return
	}
	if err := checkIndirect("dispatchWorkgroupsIndirect", b, BufferUsageIndirect, offset, 12); err != nil {
		p.r.fail(err)
		return
	}
	ol, oh := lohi(offset)
	p.r.words = append(p.r.words, uint32(opDispatchIndirect), p.r.id(&b.object), ol, oh)
}

// End records end.
func (p *ComputePassRecorder) End() error {
	if !p.ok("end") {
		return p.r.err
	}
	p.open = false
	p.r.pass = 0
	p.r.words = append(p.r.words, uint32(opEndPass))
	return p.r.err
}

// bufferRange checks a vertex or index buffer binding; size 0 means to
// the end and is resolved.
func bufferRange(op string, b *Buffer, usage BufferUsage, name string, offset uint64, size *uint64) error {
	if b == nil {
		return bounds(op, "nil buffer")
	}
	if err := checkUsage(op, b, usage, name); err != nil {
		return err
	}
	if *size == 0 {
		if offset > b.size {
			return bounds(op, "offset %d past the buffer's %d bytes", offset, b.size)
		}
		*size = b.size - offset
	}
	return checkRange(op, b, offset, *size)
}

// checkIndirect checks an indirect record of n bytes at offset.
func checkIndirect(op string, b *Buffer, usage BufferUsage, offset, n uint64) error {
	if b == nil {
		return bounds(op, "nil buffer")
	}
	if err := checkUsage(op, b, usage, "INDIRECT"); err != nil {
		return err
	}
	if offset%4 != 0 {
		return bounds(op, "offset %d is not a multiple of 4", offset)
	}
	return checkRange(op, b, offset, n)
}
