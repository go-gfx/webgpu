# webgpu

WebGPU for Go in the browser (`GOOS=js GOARCH=wasm`): a typed API named after
the [WebGPU specification](https://www.w3.org/TR/webgpu/), usable from any Go
program that runs in a window or a worker.

**Status: in development, not released.** The direct API and batched command
encoding (one JavaScript call per submit) are in place; OffscreenCanvas
surfaces and the first release are next.

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
evaluates nothing and refuses a malformed stream. Its Go mirror (`decode.go`)
reads the same format with the same checks; it is fuzzed, and a browser test
holds the two readers to the same verdict on thousands of mutated streams.

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

Batching is 5.8 to 6.1 times faster than direct calls from 1000 draws up. It
is still 2.7 times the JavaScript floor: recording costs about 40 ns per draw
in wasm (10 ns natively, `BenchmarkRecordDraws`), and the rest is the replay
itself.

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
