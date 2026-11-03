// Package batch is the host side of a batched GPU command stream, without a
// transport: the table of a guest's object ids, the creation and
// destruction of its objects, the replay of its streams and, for streams
// that read back (wire.ReadBuffer), the bounded read window.
//
// A guest records with a remote device (webgpu.NewRemoteDevice) or a
// webgpu.Recorder and sends three kinds of message: create an object from a
// wire descriptor, destroy objects, run a stream. A transport — go-virtio's
// webgpu-batch frames over virtio-gpu, a WebAssembly host module — decodes
// its own framing and calls Context's three methods; the semantics are
// here once. The ids a guest uses are private to its Context and are
// translated (wire.Remap, wire.RemapDescriptor) to the ids the Backend
// returned before the Backend sees anything, so a stream can only name
// objects its own context created.
//
// The package is portable: the Backend that runs the work is what needs a
// GPU (a browser device, a native one, or a test's recording fake).
package batch

import (
	"errors"
	"fmt"
	"math"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
)

// Backend runs a context's work. A Context has checked and translated
// everything it passes: ids are the ones Create returned.
type Backend interface {
	// Create makes an object of a wire kind from its descriptor (object
	// references already translated) and returns the id the backend's
	// replayer knows it by (nonzero).
	Create(kind uint32, desc []byte) (id uint32, err error)
	// Destroy releases an object Create returned.
	Destroy(id uint32)
	// Replay executes one stream whose ids are all ids Create returned, and
	// returns once the GPU has executed it.
	Replay(words []uint32, data []byte) error
	// Close releases everything the backend holds for the context.
	Close()
}

// Reader is a Backend that can read buffers back. Read copies
// [offset, offset+len(dst)) of buffer id into dst after a Replay has
// returned (mapAsync, getMappedRange, copy, unmap); the Context has checked
// the range, the buffer's MAP_READ usage and the window.
type Reader interface {
	Backend
	Read(id uint32, offset uint64, dst []byte) error
}

// DefaultMaxRead is the read window a Config without MaxRead grants, 1 MiB.
const DefaultMaxRead = 1 << 20

// Config bounds a Context.
type Config struct {
	// MaxObjects, if nonzero, bounds the guest ids: a guest id is in
	// [1, MaxObjects).
	MaxObjects uint32
	// MaxRead bounds the read window of one stream: the window a caller
	// passes, and the bytes the stream's reads return together. 0 is
	// DefaultMaxRead; it is capped at wire.MaxReadWindow.
	MaxRead uint32
	// MaxWorkgroups bounds a Dispatch's counts per dimension; 0 is
	// wire.DefaultMaxWorkgroups (WebGPU's default limit).
	MaxWorkgroups uint32
	// WorkgroupInvocations is the most invocations one workgroup can hold,
	// the device's maxComputeInvocationsPerWorkgroup (0: 256, WebGPU's
	// default). A Dispatch is counted at its workgroup counts times this.
	WorkgroupInvocations uint32
	// MaxDispatchInvocations bounds one Dispatch, and MaxStreamInvocations
	// the Dispatches of one stream together, in invocations (0: no bound);
	// see wire.CheckInvocations. A DispatchIndirect is not counted.
	MaxDispatchInvocations, MaxStreamInvocations uint64
}

// Errors. Every refusal wraps one; after any error the Context is lost.
var (
	// ErrRefused: a message the context's own state refuses (an id in use,
	// an unknown kind, an id not live).
	ErrRefused = errors.New("batch: message refused")
	// ErrLost: the context failed earlier and accepts nothing more.
	ErrLost = errors.New("batch: context lost")
	// ErrRead: a read-back the context refuses: no window, a buffer that is
	// unknown or lacks MAP_READ, a range outside the buffer or the window,
	// a window or reads above the quota.
	ErrRead = errors.New("batch: read-back refused")

	// ErrInvalid: the GPU refused an object the guest described: WGSL that
	// does not compile, a pipeline or bind group that fails WebGPU validation.
	// The error carries the browser's message (the WGSL line and column).
	ErrInvalid = errors.New("batch: the GPU refused the object")
)

type bufInfo struct {
	size  uint64
	usage webgpu.BufferUsage
}

// Context is one guest's objects on a Backend. It is not safe for
// concurrent use.
type Context struct {
	be      Backend
	rd      Reader
	cfg     Config
	ids     map[uint32]uint32  // guest id -> backend id
	bufs    map[uint32]bufInfo // backend id -> buffer size and usage
	lost    error
	maxRead uint64
	peak    uint64
}

// PeakInvocations is the most invocations one stream of this context asked
// for (wire.CheckInvocations' total): what a quota would have to allow.
func (c *Context) PeakInvocations() uint64 { return c.peak }

// New returns a Context on be. If be is a Reader, streams may read back.
func New(be Backend, cfg Config) *Context {
	c := &Context{be: be, cfg: cfg, ids: map[uint32]uint32{}, bufs: map[uint32]bufInfo{}}
	c.rd, _ = be.(Reader)
	c.maxRead = uint64(cfg.MaxRead)
	if c.maxRead == 0 {
		c.maxRead = DefaultMaxRead
	}
	c.maxRead = min(c.maxRead, wire.MaxReadWindow)
	return c
}

// Lost returns the error that lost the context, or nil.
func (c *Context) Lost() error { return c.lost }

// Lookup returns the backend's id for guest object id, for a transport that
// gives an object more than the three messages do (virtio's BIND_RESOURCE).
func (c *Context) Lookup(id uint32) (uint32, bool) { return c.host(id) }

// Live is the number of guest objects alive.
func (c *Context) Live() int { return len(c.ids) }

func (c *Context) fail(err error) error {
	c.lost = err
	return err
}

func (c *Context) host(id uint32) (uint32, bool) { h, ok := c.ids[id]; return h, ok }

// Create makes guest object id of a wire kind from a descriptor (which is
// translated in place: pass a copy the caller does not keep).
func (c *Context) Create(id, kind uint32, desc []byte) error {
	if c.lost != nil {
		return fmt.Errorf("%w: %v", ErrLost, c.lost)
	}
	if _, live := c.ids[id]; live || id == 0 || c.cfg.MaxObjects != 0 && id >= c.cfg.MaxObjects || !wire.Kind(kind).Valid() {
		return c.fail(fmt.Errorf("%w: CREATE %d", ErrRefused, id))
	}
	if err := wire.RemapDescriptor(desc, c.host); err != nil {
		return c.fail(err)
	}
	h, err := c.be.Create(kind, desc)
	if err != nil {
		return c.fail(err)
	}
	c.ids[id] = h
	if wire.Kind(kind) == wire.KindBuffer {
		// A buffer's size and usage bound its reads. A descriptor the
		// decoder refuses leaves the buffer unreadable, not the context
		// lost: the backend accepted it.
		d, err := webgpu.DecodeDescriptor(wire.KindBuffer, desc, nil) // a buffer names no object
		if bd, ok := d.(webgpu.BufferDescriptor); err == nil && ok {
			c.bufs[h] = bufInfo{size: bd.Size, usage: bd.Usage}
		}
	}
	return nil
}

// Destroy releases guest objects; each must be live.
func (c *Context) Destroy(ids ...uint32) error {
	if c.lost != nil {
		return fmt.Errorf("%w: %v", ErrLost, c.lost)
	}
	for _, id := range ids {
		h, live := c.ids[id]
		if !live {
			return c.fail(fmt.Errorf("%w: DESTROY %d", ErrRefused, id))
		}
		c.be.Destroy(h)
		delete(c.ids, id)
		delete(c.bufs, h)
	}
	return nil
}

// Stream runs a guest's stream: its ids are translated in place, so words
// must be memory the caller owns (a copy of the guest's). Its ReadBuffer
// commands, checked before anything runs, are served after the replay into
// window, which the caller owns and the stream writes only inside; a stream
// that reads needs a window, and a Backend that is a Reader.
func (c *Context) Stream(words []uint32, data []byte, window []byte) error {
	if c.lost != nil {
		return fmt.Errorf("%w: %v", ErrLost, c.lost)
	}
	// Data offsets are uint32: bytes past 4 GiB are unaddressable, so the
	// bound the stream is checked against is clamped, not refused.
	cmds, err := wire.RemapCmds(words, uint32(min(uint64(len(data)), math.MaxUint32)), c.host)
	if err != nil {
		return c.fail(err)
	}
	// What wire.Decode cannot know and WebGPU would only refuse later: buffer
	// ranges and dispatch sizes, checked here before the backend is called.
	if err := wire.CheckRanges(cmds, func(id uint32) (uint64, bool) { b, ok := c.bufs[id]; return b.size, ok }, c.cfg.MaxWorkgroups); err != nil {
		return c.fail(err)
	}
	wgi := c.cfg.WorkgroupInvocations
	if wgi == 0 {
		wgi = 256
	}
	total, err := wire.CheckInvocations(cmds, wgi, c.cfg.MaxDispatchInvocations, c.cfg.MaxStreamInvocations)
	if err != nil {
		return c.fail(err)
	}
	c.peak = max(c.peak, total)
	var reads []wire.Read
	for _, cmd := range cmds {
		if r, ok := wire.ReadOf(cmd); ok {
			reads = append(reads, r)
		}
	}
	if len(reads) > 0 {
		if err := c.checkReads(reads, window); err != nil {
			return c.fail(err)
		}
	}
	if err := c.be.Replay(words, data); err != nil {
		return c.fail(err)
	}
	for _, r := range reads {
		dst := window[r.WindowOffset : r.WindowOffset+r.Size]
		if err := c.rd.Read(r.ID, r.Offset, dst); err != nil {
			return c.fail(err)
		}
	}
	return nil
}

// checkReads applies, to a stream's reads (ids already the backend's), every
// rule that does not need the GPU: a Reader and a window within the quota,
// MAP_READ buffers, ranges inside buffers and inside the window.
func (c *Context) checkReads(reads []wire.Read, window []byte) error {
	if c.rd == nil {
		return fmt.Errorf("%w: the backend cannot read back", ErrRead)
	}
	if uint64(len(window)) > c.maxRead {
		return fmt.Errorf("%w: a %d-byte window above the %d-byte quota", ErrRead, len(window), c.maxRead)
	}
	var total uint64
	for _, r := range reads {
		b, ok := c.bufs[r.ID]
		switch {
		case !ok:
			return fmt.Errorf("%w: object %d is not a buffer with a known descriptor", ErrRead, r.ID)
		case b.usage&webgpu.BufferUsageMapRead == 0:
			return fmt.Errorf("%w: buffer %d lacks MAP_READ", ErrRead, r.ID)
		case r.Offset > b.size || uint64(r.Size) > b.size-r.Offset:
			return fmt.Errorf("%w: range [%d, +%d) outside the buffer's %d bytes", ErrRead, r.Offset, r.Size, b.size)
		case uint64(r.WindowOffset)+uint64(r.Size) > uint64(len(window)):
			return fmt.Errorf("%w: window range [%d, +%d) outside the %d-byte window", ErrRead, r.WindowOffset, r.Size, len(window))
		}
		total += uint64(r.Size)
		if total > c.maxRead {
			return fmt.Errorf("%w: reads above the %d-byte quota", ErrRead, c.maxRead)
		}
	}
	return nil
}

// Close forgets every object and closes the backend, which releases
// whatever it holds for the context.
func (c *Context) Close() {
	clear(c.ids)
	clear(c.bufs)
	c.be.Close()
}
