# webgpu

WebGPU for Go in the browser (`GOOS=js GOARCH=wasm`): a typed API named after
the [WebGPU specification](https://www.w3.org/TR/webgpu/), usable from any Go
program that runs in a window or a worker.

**Status: released, v0.1.0** (pre-1.0: a minor version may change the API).
It covers the direct API, batched command encoding (one JavaScript call per
submit) and OffscreenCanvas surfaces. The go-wasmcontainers GPU host runs on
it.

```sh
go get github.com/go-gfx/webgpu@v0.1.0
```

```go
import "github.com/go-gfx/webgpu"
```

## What it covers

- Adapters and devices, refusing a fallback (software) adapter unless asked;
  device loss as a channel.
- Buffers (vertex, index, uniform, storage, indirect), mapping, `WriteBuffer`.
- Textures of every format of the specification, views, samplers,
  `WriteTexture` (dirty-rectangle updates), texture and buffer copies.
- Bind group layouts, pipeline layouts, bind groups, render and compute
  pipelines (sync and async), depth-stencil, blending, multisampling.
- Render passes (indexed and instanced draws, indirect draws, viewport,
  scissor), render bundles, compute passes.
- Error scopes and uncaptured errors as Go errors.
- Surfaces: an OffscreenCanvas with a WebGPU context, frames out as
  `ImageBitmap`s, and a signal for browsers that have no such context.

## Surfaces

```go
if err := webgpu.CanvasSupport(); err != nil {
	// errors.Is(err, webgpu.ErrNoCanvasContext): use another renderer
}
s, err := device.NewSurface(webgpu.SurfaceDescriptor{Width: 640, Height: 480})
tex, err := s.CurrentTexture()     // render into a view of it
bitmap, err := s.Present()         // transferToImageBitmap: post it to the page, or close it
```

A standalone OffscreenCanvas never shows itself (the specification says the
canvas is presented automatically only for a placeholder canvas): `Present`
moves the frame out as an `ImageBitmap` without a copy, for a worker to post to
its page. `NewSurfaceFromCanvas` takes a canvas the page transferred with
`transferControlToOffscreen`; the browser presents that one by itself. The
surface size is bounded by the quota (`SurfaceWidth`, `SurfaceHeight`).

`CanvasSupport` probes `getContext("webgpu")` on an `OffscreenCanvas`. MDN's
compatibility data lists that context in Chrome 113 (every platform from 144),
Safari 26, and Firefox 141 on Windows only. Where it returns
`ErrNoCanvasContext`, a program should fall back to a non-WebGPU renderer.
The browser tests read each presented frame back through a 2D canvas, pixel by
pixel, so a blank frame fails them. That includes the Linux SwiftShader lane,
where headless Chromium needs Skia on Vulkan to present at all.

## Batched encoding: one JavaScript call per submit

A `syscall/js` call costs about ten times the WebGPU call it carries. A
`Recorder` records a whole submit in Go (queue writes, render and compute
passes, dynamic offsets included) as `uint32` words; `Submit` copies them to
JavaScript once and `replay.js` issues the WebGPU calls:

```go
webgpu.InstallReplay(ctx, "/webgpu-replay.js") // once; "" loads the embedded copy from a blob: URL
rec := device.NewRecorder()                    // reuse it frame after frame
rec.WriteBuffer(uniforms, 0, frameData)
p := rec.BeginRenderPass(passDesc)
p.SetPipeline(pipe)
for _, s := range surfaces {
	p.SetBindGroup(0, s.group)
	p.DrawIndexed(s.count, 1, s.first, 0, 0)
}
p.End()
seq, err := rec.Submit()                       // one call; the first refused command, if any
err = device.WaitSubmitted(ctx, seq)           // only when the CPU must wait for the GPU
```

The recorder checks every command as the direct API does. `replay.js`
(`webgpu.ReplayJS`, served as `webgpu.ReplayJSName`) is a plain same-origin
script, so a page with `script-src 'self'` serves it like `wasm_exec.js`; it
evaluates nothing and refuses a malformed stream. Its Go mirror, `wire.Decode`,
reads the same format with the same checks. It is fuzzed, and a browser test
holds the two readers to the same verdict on thousands of mutated streams.

The stream format is exported as package **`wire`**, which has no GOOS
constraint. It contains the opcodes, `Decode`, `StreamVersion` and
`Remap`. `Remap` rewrites every object id of a stream through a mapping
and refuses any id the mapping does not grant. With it, a stream recorded
in one place (for example, a VM guest sending it over the
[go-virtio/gpu webgpu-batch](https://github.com/go-virtio/gpu/blob/main/docs/WEBGPU.md)
context) can be checked and replayed by another table without a second
implementation of the format.

Objects are created remotely the same way. `EncodeDescriptor` turns a
creation descriptor (buffer, texture, view, sampler, shader module, bind
group layout, pipeline layout, bind group, render or compute pipeline)
into bytes. It names each referenced object by its replay id, in a
header (`wire.SplitDescriptor`), and `wire.RemapDescriptor` translates
those ids without reading the body. `DecodeDescriptor` rebuilds the
descriptor with the receiver's objects. It refuses unknown fields, a
wrong reference count and references of the wrong type, and it never
keeps an object that the body itself carries. The decoder is fuzzed
(`FuzzDecodeDescriptor`).

Submissions are numbered on a timeline (package `timeline`, shared by
`Queue.Submit` and `Recorder.Submit`): completing number n completes every
earlier one, the fence model of virtio-gpu, so many waiters cost one
`onSubmittedWorkDone` round trip. The package has no dependency and no notion
of a GPU.

## Errors at the call, not later on the device

WebGPU reports most mistakes asynchronously. This package checks in Go first
and returns a `*webgpu.BoundsError` from the call: a `WriteBuffer` past the end
of its buffer, a `WriteTexture` region outside the mip level or a layout that
does not fit its data, a vertex buffer range past the buffer, a missing usage.
Data from untrusted sources (game maps, container guests) cannot reach the GPU
out of range without the caller being told. All size arithmetic is overflow
checked and tested on 32-bit targets.

## Quotas for embedders

```go
q := &webgpu.Quota{MemoryBytes: 256 << 20, Pipelines: 256, ShaderBytes: 1 << 20}
d, err := adapter.RequestDevice(ctx, &webgpu.DeviceDescriptor{Quota: q})
```

A quota is charged before the browser is called (texture sizes are computed by
the package, mips and samples included) and refunded on `Destroy`, `Release`
or device loss. Optional features and limits are refused unless the quota
lists them, `timestamp-query` included.

## Promises

Methods that wait on a promise (`RequestAdapter`, `RequestDevice`, `MapAsync`,
`PopErrorScope`, `OnSubmittedWorkDone`, `Create*PipelineAsync`) block only the
calling goroutine, take a `context.Context`, and must not be called from a
`js.FuncOf` callback.

## Tests

The descriptor, validation and quota code is platform-neutral and has every
block covered on Linux, macOS, Windows and linux/386. The calls are tested in a
real browser worker by `internal/cmd/wasmbrowser`, a `go test -exec` runner
over Playwright:

```sh
go -C internal/cmd/wasmbrowser build -o /tmp/wasmbrowser .
WEBGPU_REQUIRE_GPU=1 GOOS=js GOARCH=wasm go test -exec=/tmp/wasmbrowser ./...
```

The browser tests render a GLQuake-shaped frame (8-bit index texture through a
palette texture and a sampler, vertex and index buffers, instancing with
per-instance data in a storage buffer, a depth buffer, a dirty-rectangle
texture update, the same frame from a render bundle) and check every pixel. CI
runs them on macOS with a GPU required (a fallback adapter fails the lane) and
on Linux on SwiftShader.

## Performance

Measured on an Apple M4 Max, Chromium 143, Metal adapter, in a worker. The
machine's load was 10 to 16 during these runs, so the numbers are indicative.
One `syscall/js` call costs about 200 ns for `setPipeline`, `setBindGroup` or
`draw`, against 17 ns for the same `draw` issued from JavaScript: the cost is
the call itself (each `Value.Call` decodes its method name).

CPU time per frame of n draws (`setBindGroup` + `draw` each), median of three
runs of `BenchmarkFrame`:

| draws | direct calls | `Recorder` | the same frame in JavaScript |
|---:|---:|---:|---:|
| 100 | 80 µs | 22 µs | 13 µs |
| 1000 | 491 µs | 85 µs | 32 µs |
| 5000 | 2320 µs | 380 µs | 138 µs |

Batching is 5.8 to 6.1 times faster than direct calls from 1000 draws up. The
design estimated about 15 times, from a prototype that recorded with no checks
and no object table; the checks and the table cost the difference. It is still
2.7 times the JavaScript floor: recording costs about 40 ns per draw in wasm
(10 ns natively, `BenchmarkRecordDraws`), and the rest is the replay itself.
All of these timings were taken with the machine's load between 10 and 16, so
they are indicative, not final.

`WriteBuffer` copies through a staging array the device reuses; allocating a
fresh `Uint8Array` per call, as is common, was 3 to 4 times slower. Whether
the source is a 64 KiB array, a view into a 64 MiB one or into the Go
instance's own memory makes no difference: about 5 µs per 64 KiB write, all
spent in the browser's copy (`BenchmarkWriteBuffer64KiBSource`). An earlier
prototype that found the Go-memory view slower had measured a queue that was
backing up. A zero-copy path would save one copy per write but would depend
on `wasm_exec.js` internals, so writes go through the staging copy.

## Licence

BSD 3-Clause, see [LICENSE](LICENSE).
