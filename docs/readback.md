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

Reproduce:

```sh
go -C internal/cmd/wasmbrowser build -o /tmp/wasmbrowser .
WEBGPU_REQUIRE_GPU=1 GOOS=js GOARCH=wasm go test -count=1 -v -run TestReadLatency -exec=/tmp/wasmbrowser ./batch/
```
