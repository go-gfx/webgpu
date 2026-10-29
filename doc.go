// Package webgpu is WebGPU for Go programs that run in the browser
// (GOOS=js GOARCH=wasm), in a window or a worker: typed handles named after
// the WebGPU IDL (https://www.w3.org/TR/webgpu/), errors as Go errors,
// promises awaited without blocking the event loop, and quotas an embedder
// can enforce on code it does not trust.
//
// # Shape
//
//	g, err := webgpu.Navigator()                       // navigator.gpu
//	a, err := g.RequestAdapter(ctx, nil)               // refuses a fallback adapter by default
//	d, err := a.RequestDevice(ctx, &webgpu.DeviceDescriptor{Quota: q})
//	b, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: 64, Usage: webgpu.BufferUsageVertex | webgpu.BufferUsageCopyDst})
//	err = d.Queue().WriteBuffer(b, 0, vertices)         // range checked here
//	enc, _ := d.CreateCommandEncoder("frame")
//	pass, _ := enc.BeginRenderPass(desc)
//	pass.SetPipeline(p); pass.SetVertexBuffer(0, b, 0, 0); pass.Draw(3, 1, 0, 0)
//	err = pass.End()                                    // first refused command, if any
//
// Descriptors are Go structs; a zero field is left out of the dictionary so
// the browser applies the specification's default.
//
// # Batched encoding
//
// A syscall/js call costs about ten times the WebGPU call it carries. A
// Recorder records a whole submit as uint32 words in Go and Submit replays
// it with one call into replay.js (ReplayJS, loaded once by
// InstallReplay), which issues the WebGPU calls from JavaScript. Every
// command is checked as in the direct API. Submissions are numbered on the
// device's timeline (package timeline): WaitSubmitted(ctx, seq) waits for
// the GPU, and completing a number completes every earlier one.
//
// # Errors
//
// WebGPU reports most mistakes asynchronously, as validation errors on the
// device that nobody sees without an error scope. This package checks what
// it can in Go first and returns a *BoundsError at the call: out-of-range
// buffer writes and copies, texture regions outside a mip level, linear
// layouts that do not fit their data (the specification's "validating
// linear texture data"), missing usages. Exceptions the browser throws are
// *JSError; device errors are *GPUError, through PushErrorScope and
// PopErrorScope or OnUncapturedError (delivered with the next round trip to
// the GPU process).
//
// # Promises
//
// RequestAdapter, RequestDevice, MapAsync, PopErrorScope,
// OnSubmittedWorkDone and the Create*PipelineAsync methods wait on a
// promise: only the calling goroutine blocks, the browser's event loop keeps
// running. They must not be called from a js.FuncOf callback, whose
// goroutine holds the event loop. Each takes a context.
//
// # Quotas
//
// A Quota bounds GPU memory (buffer and texture bytes, computed by the
// package: TextureBytes), live pipelines, WGSL source size, and the
// features and limits a device may require. It is charged before the
// browser is called and refunded on Destroy, Release, or the loss of the
// device; several devices may share one. Nothing is reachable except
// through a Device the embedder created.
//
// # Surfaces
//
// A Surface is an OffscreenCanvas with a WebGPU context: render into
// CurrentTexture, and Present moves the frame out as an ImageBitmap
// (standalone canvases are never presented by the browser). CanvasSupport
// returns ErrNoCanvasContext where the browser has no such context
// (Firefox outside Windows): the signal to fall back to another renderer.
//
// Outside GOOS=js the package compiles (descriptors, validation, quotas)
// and Navigator does not exist.
package webgpu
