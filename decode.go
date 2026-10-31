package webgpu

import "github.com/go-gfx/webgpu/wire"

// The stream format lives in package wire (portable, exported so that a
// stream can be checked and remapped outside the browser, e.g. by a
// virtio-gpu host). These names keep the Recorder's code short; they are
// the wire package's, not a copy.

type (
	opcode = wire.Op
	cmd    = wire.Cmd
)

const (
	opSetPipeline          = wire.SetPipeline
	opSetBindGroup         = wire.SetBindGroup
	opDraw                 = wire.Draw
	opDrawIndexed          = wire.DrawIndexed
	opSetVertexBuffer      = wire.SetVertexBuffer
	opSetIndexBuffer       = wire.SetIndexBuffer
	opSetViewport          = wire.SetViewport
	opSetScissorRect       = wire.SetScissorRect
	opSetBlendConstant     = wire.SetBlendConstant
	opSetStencilReference  = wire.SetStencilReference
	opExecuteBundles       = wire.ExecuteBundles
	opDrawIndirect         = wire.DrawIndirect
	opDrawIndexedIndirect  = wire.DrawIndexedIndirect
	opBeginRenderPass      = wire.BeginRenderPass
	opEndPass              = wire.EndPass
	opBeginComputePass     = wire.BeginComputePass
	opDispatch             = wire.Dispatch
	opDispatchIndirect     = wire.DispatchIndirect
	opWriteBuffer          = wire.WriteBuffer
	opCopyBufferToBuffer   = wire.CopyBufferToBuffer
	opCopyTextureToTexture = wire.CopyTextureToTexture
	opSubmit               = wire.Submit
)

// decode is wire.Decode: tests compare it with the recorder and with
// replay.js, and fuzz it; the browser never runs it.
var decode = wire.Decode
