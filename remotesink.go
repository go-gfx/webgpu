package webgpu

import "github.com/go-gfx/webgpu/wire"

// RemoteSink carries the work of a remote device (NewRemoteDevice, built
// outside the browser) to where a real device runs it — for example to a
// virtio-gpu host through go-virtio/gpu's webgpu-batch context.
//
// Object ids are replay ids (ReplayID): the ids streams and descriptors
// name objects by. The receiver translates them to its own objects
// (wire.RemapDescriptor, wire.Remap).
type RemoteSink interface {
	// Create makes object id of kind from a wire descriptor
	// (EncodeDescriptor's bytes).
	Create(id uint32, kind wire.Kind, desc []byte) error
	// Destroy releases object id.
	Destroy(id uint32) error
	// Submit executes one Recorder submit and returns once it has
	// executed (a fenced virtio submission), or with the error that lost
	// the remote device.
	Submit(words []uint32, data []byte) error
}
