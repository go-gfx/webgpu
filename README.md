# webgpu

WebGPU for Go in the browser (`GOOS=js GOARCH=wasm`): a typed API named after
the [WebGPU specification](https://www.w3.org/TR/webgpu/), usable from any Go
program that runs in a window or a worker.

**Status: in development, not released.** The direct API is in place
(this README describes it); batched command encoding (one JavaScript call per
submit), OffscreenCanvas surfaces and the first release are next.

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

Measured on an Apple M4 Max, Chromium 143, Metal adapter, in a worker (machine
load 9 to 17: indicative). One `syscall/js` call costs about 200 ns for
`setPipeline`, `setBindGroup` or `draw`, against 17 ns for the same `draw`
issued from JavaScript: the cost is the call itself (each `Value.Call` decodes
its method name). A frame of 1000 draws costs about 0.5 ms of CPU through
direct calls. Batched encoding (next) records commands in Go and replays them
in one call: about 40 µs for the same frame in the prototype.

`WriteBuffer` copies through a staging array the device reuses; allocating a
fresh `Uint8Array` per call, as is common, was 3 to 4 times slower.

## Licence

BSD 3-Clause, see [LICENSE](LICENSE).
