# Read-back: cost of one round trip

`batch.TestReadLatency` (`batch/device_js_test.go`) times `Context.Stream` on a
`DeviceBackend`: replay, wait for the GPU, `mapAsync`, copy into the window,
`unmap`. Apple M4 Max, Chromium (Playwright), worker, Metal adapter
(`vendor "apple", architecture "metal-3"`), 1 submit per round, the four lanes
interleaved round by round, 3 warm-up rounds then 30 measured.

| window | read only (buffer already holds the data) | dispatch + copy + read |
|---:|---:|---:|
| 4 KiB (1024 words) | 124 us | 270 us |
| 1 MiB (262 144 words) | 155 us | 320 us |

Medians. The run was made at a 1-minute load of 5.0 (the rule is at most 8).
Two further runs, made while the load rose to 17 and so not valid, gave 135 to
169 us and 274 to 344 us for 4 KiB, 165 to 205 us and 324 to 395 us for 1 MiB:
the same ordering, about 25 % slower.

What it says: the fixed cost of a read is about 120 us, independent of the
window up to 1 MiB (the copy of 1 MiB adds about 30 us); the compute pass, the
buffer copy and the wait for the GPU add about 150 us. This is the host side
only; a WebAssembly guest adds its own crossing (the 38 us of the
`wasi:webgpu` round trip in go-wasmcontainers/gpu DESIGN.md is the reference).
For a decoder emitting one token per submit, the read-back of the sampled token
(4 bytes) costs the 4 KiB figure at most.

## Firefox: completions arrive on a 100 ms timer

Firefox 159 (macOS, Apple silicon) resolves every `onSubmittedWorkDone` and
`mapAsync` on the tick of a 100 ms timer, whatever the size of the read and
whether or not a compute pass ran: a bare page doing submit + `mapAsync` takes
100 ms, and `onSubmittedWorkDone` then `mapAsync` takes 200 ms (an empty
`onSubmittedWorkDone` and a `mapAsync` of an idle buffer take 100 ms too). The
cause is in Firefox: `dom/webgpu/ipc/WebGPUParent.cpp` starts a timer at
`POLL_TIME_MS = 100` that calls `wgpu_server_poll_all_devices(.., false)`, and
`gfx/wgpu_bindings/src/server.rs` polls after neither a map nor a work-done
request (the only blocking poll is the canvas snapshot's). It is not the
Worker, `Atomics.wait` or the event loop: the control page is bare and the
numbers are the same in a Worker.

`Queue.OnSubmittedWorkDone`, `Device.WaitSubmitted` and `Buffer.MapAsync` work
around it: while they wait they kick the queue with an empty
`GPUQueue.submit([])` (first after 500 us, then at doubling gaps up to 8 ms;
`kickLoop`). A submit ends with a non-blocking poll in wgpu-core, which fires
the completions that are ready. In a bare Firefox page the same kick takes a
map from 100 ms to about 0.3 ms. Other browsers resolve before the first kick
or ignore it. A write (`queue.writeBuffer`) is not a kick: it did not help.

Reproduce:

```sh
go -C internal/cmd/wasmbrowser build -o /tmp/wasmbrowser .
WEBGPU_REQUIRE_GPU=1 GOOS=js GOARCH=wasm go test -count=1 -v -run TestReadLatency -exec=/tmp/wasmbrowser ./batch/
```
