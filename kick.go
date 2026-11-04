package webgpu

import "time"

// Firefox delivers a GPU completion (GPUQueue.onSubmittedWorkDone,
// GPUBuffer.mapAsync) only when its GPU process polls the device, and it
// polls on a fixed 100 ms timer (dom/webgpu/ipc/WebGPUParent.cpp,
// POLL_TIME_MS = 100, WebGPUParent::MaintainDevices; nothing polls after a
// map or a work-done request). A loop that submits, waits and submits again
// therefore pays 100 ms per wait, 200 ms for a work-done wait followed by a
// map (go-wasmcontainers/gpu#25). A GPUQueue.submit, however, ends with a
// non-blocking poll in wgpu-core, and an empty submit is free: kicking the
// queue with one, while a wait is pending, delivers the completion as soon as
// the GPU has it (measured on Firefox 159: 100 ms -> 0.3 ms for a map, in a
// page and in a Worker). Other browsers resolve before the first kick, or
// ignore it.
const (
	kickFirst = 500 * time.Microsecond // before the first kick: Chromium and Safari are done by then
	kickMax   = 8 * time.Millisecond   // the gap doubles up to this, so a long GPU job costs ~125 kicks/s
)

// kickLoop calls submit, at doubling gaps from kickFirst to kickMax, until
// stop is closed. It is run in its own goroutine for the duration of a wait.
func kickLoop(stop <-chan struct{}, submit func()) {
	gap := kickFirst
	t := time.NewTimer(gap)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			submit()
			gap = min(gap*2, kickMax)
			t.Reset(gap)
		}
	}
}
