package webgpu

import (
	"errors"

	"github.com/go-gfx/webgpu/wire"
)

// Read-back through the batched stream.
//
// GPUBuffer.mapAsync is asynchronous, which a stream replayed in one call
// cannot express, and a remote device has no mapped memory to hand out.
// Recorder.ReadBuffer therefore records wire.ReadBuffer: when the submit's
// GPU work has completed, the replayer copies a range of a MAP_READ buffer
// into a read window of the submit — mapAsync(READ), getMappedRange, copy,
// unmap, all on the replayer's side — and the bytes come back with the
// submit. The rules are mapAsync's (WebGPU, GPUBuffer.mapAsync): the buffer
// needs MAP_READ, the offset is a multiple of 8, the size a multiple of 4,
// and the range lies inside the buffer. The window is bounded per submit
// (Quota.ReadBytes, wire.MaxReadWindow), so a client cannot make the host
// copy without limit.

// ErrNotRead: a Readback whose submit has not completed (or failed).
var ErrNotRead = errors.New("webgpu: the read-back has not completed")

// ErrMapState: MapAsync on a buffer that is already mapped; WebGPU requires
// the "unmapped" state.
var ErrMapState = errors.New("webgpu: buffer is not unmapped")

// Readback is the result of one Recorder.ReadBuffer: its bytes are valid
// after the Recorder's Submit returned without error.
type Readback struct {
	buf    *Buffer
	offset uint64
	size   uint32
	win    uint32 // offset in the submit's read window
	data   []byte
	err    error
}

// Bytes returns the bytes read, owned by the caller. It is ErrNotRead until
// the submit that carries the read has succeeded, and the recording error
// if the read was refused.
func (rb *Readback) Bytes() ([]byte, error) {
	if rb.err != nil {
		return nil, rb.err
	}
	if rb.data == nil {
		return nil, ErrNotRead
	}
	return rb.data, nil
}

// checkReadBuffer applies mapAsync's validation to a read-back range.
func checkReadBuffer(b *Buffer, off, size uint64) error {
	const op = "readBuffer"
	if b == nil {
		return bounds(op, "nil buffer")
	}
	if err := checkUsage(op, b, BufferUsageMapRead, "MAP_READ"); err != nil {
		return err
	}
	if off%8 != 0 || size%4 != 0 || size == 0 {
		return bounds(op, "offset %d must be a multiple of 8 and size %d a nonzero multiple of 4", off, size)
	}
	if size > wire.MaxReadWindow {
		return bounds(op, "size %d above the %d-byte read window", size, wire.MaxReadWindow)
	}
	return checkRange(op, b, off, size)
}

// ReadBuffer records a read of [offset, offset+size) of b, a MAP_READ
// buffer, after this submit's GPU work. Record the copy into b before it
// (CopyBufferToBuffer); the read sees everything the stream does. Submit
// delivers the bytes to the returned Readback. A refused read is kept as
// the Recorder's error, as every command's is, and in the Readback.
func (r *Recorder) ReadBuffer(b *Buffer, offset, size uint64) *Readback {
	rb := &Readback{buf: b, offset: offset}
	if !r.in(0, "readBuffer inside a pass") {
		rb.err = r.err
		return rb
	}
	err := checkReadBuffer(b, offset, size)
	if err == nil {
		err = r.dev.quota.checkRead(r.win + size)
	}
	if err == nil && r.win+size > wire.MaxReadWindow {
		err = bounds("readBuffer", "read window of %d bytes in one submit", r.win+size)
	}
	if err != nil {
		r.fail(err)
		rb.err = err
		return rb
	}
	rb.size, rb.win = uint32(size), uint32(r.win)
	r.win += size
	lo, hi := lohi(offset)
	r.words = append(r.words, uint32(opReadBuffer), r.id(&b.object), lo, hi, rb.size, rb.win)
	r.reads = append(r.reads, rb)
	return rb
}

// deliver gives every read of the recorder its bytes, from the window the
// submit filled.
func (r *Recorder) deliver(window []byte) {
	for _, rb := range r.reads {
		rb.data = window[rb.win : rb.win+rb.size : rb.win+rb.size]
	}
}

// ReadSink is a RemoteSink that also returns read-backs: SubmitRead
// executes the submit as Submit does and, once its GPU work has completed,
// fills window (len = the submit's read window) as the stream's ReadBuffer
// commands say. A remote device whose sink is not a ReadSink refuses a
// submit that reads.
type ReadSink interface {
	RemoteSink
	SubmitRead(words []uint32, data []byte, window []byte) error
}

// checkRead refuses a read window above the quota's ReadBytes.
func (q *Quota) checkRead(n uint64) error {
	if q == nil || q.ReadBytes <= 0 || n <= uint64(q.ReadBytes) {
		return nil
	}
	return &QuotaError{Resource: "read window", Limit: int64(q.ReadBytes), Want: int64(n)}
}
