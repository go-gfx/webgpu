//go:build !js

package batch

import (
	"context"
	"errors"
	"testing"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
)

// fake is a Backend that records, and a Reader whose bytes are a function
// of the offset, so a test knows what a window must hold.
type fake struct {
	next    uint32
	objs    map[uint32]uint32 // id -> kind
	replays int
	reads   [][2]uint64
	closed  bool
	failAt  string
}

func newFake() *fake { return &fake{objs: map[uint32]uint32{}} }

func (f *fake) Create(kind uint32, desc []byte) (uint32, error) {
	if f.failAt == "create" {
		return 0, errors.New("create failed")
	}
	f.next++
	f.objs[f.next] = kind
	return f.next, nil
}
func (f *fake) Destroy(id uint32) { delete(f.objs, id) }
func (f *fake) Replay(words []uint32, data []byte) error {
	f.replays++
	if f.failAt == "replay" {
		return errors.New("replay failed")
	}
	return nil
}
func (f *fake) Close() { f.closed = true }

type readFake struct{ *fake }

func (f readFake) Read(id uint32, offset uint64, dst []byte) error {
	f.reads = append(f.reads, [2]uint64{offset, uint64(len(dst))})
	if f.failAt == "read" {
		return errors.New("read failed")
	}
	for i := range dst {
		dst[i] = byte(offset) + byte(i)
	}
	return nil
}

// toBatch is the RemoteSink a guest's remote device talks to; it is the
// whole of a transport, minus framing.
type toBatch struct{ c *Context }

func (s toBatch) Create(id uint32, kind wire.Kind, desc []byte) error {
	return s.c.Create(id, uint32(kind), append([]byte(nil), desc...))
}
func (s toBatch) Destroy(id uint32) error { return s.c.Destroy(id) }
func (s toBatch) Submit(w []uint32, d []byte) error {
	return s.c.Stream(append([]uint32(nil), w...), append([]byte(nil), d...), nil)
}
func (s toBatch) SubmitRead(w []uint32, d []byte, window []byte) error {
	return s.c.Stream(append([]uint32(nil), w...), append([]byte(nil), d...), window)
}

func guest(t *testing.T, cfg Config, be Backend) (*webgpu.Device, *Context) {
	t.Helper()
	c := New(be, cfg)
	return webgpu.NewRemoteDevice(toBatch{c}, "guest", nil), c
}

func buffer(t *testing.T, d *webgpu.Device, size uint64, u webgpu.BufferUsage) *webgpu.Buffer {
	t.Helper()
	b, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: size, Usage: u})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestReadBack: a guest copies into a MAP_READ buffer and reads it back
// through MapAsync and through a Recorder; the bytes are the backend's.
func TestReadBack(t *testing.T) {
	f := newFake()
	d, c := guest(t, Config{}, readFake{f})
	src := buffer(t, d, 64, webgpu.BufferUsageCopySrc|webgpu.BufferUsageCopyDst)
	out := buffer(t, d, 64, webgpu.BufferUsageMapRead|webgpu.BufferUsageCopyDst)
	r := d.NewRecorder()
	r.CopyBufferToBuffer(src, 0, out, 0, 64)
	a := r.ReadBuffer(out, 8, 16)
	b := r.ReadBuffer(out, 32, 8)
	if _, err := r.Submit(); err != nil {
		t.Fatal(err)
	}
	ga, _ := a.Bytes()
	gb, _ := b.Bytes()
	if len(ga) != 16 || ga[0] != 8 || ga[15] != 23 || len(gb) != 8 || gb[0] != 32 {
		t.Fatalf("a %v b %v", ga, gb)
	}
	if err := out.MapAsync(context.Background(), webgpu.MapModeRead, 0, 0); err != nil {
		t.Fatal(err)
	}
	all := make([]byte, 64)
	if err := out.ReadMapped(0, all); err != nil || all[63] != 63 {
		t.Fatalf("%v %v", all, err)
	}
	if c.Lost() != nil || f.replays != 2 || len(f.reads) != 3 {
		t.Fatalf("lost %v, %d replays, %d reads", c.Lost(), f.replays, len(f.reads))
	}
}

// TestReadRefusals: each refusal is an error at the call, loses the
// context, and reaches neither the replay nor the reads.
func TestReadRefusals(t *testing.T) {
	setup := func(be Backend, cfg Config) (*Context, []uint32, uint32, uint32) {
		c := New(be, cfg)
		mk := func(id uint32, size uint64, u webgpu.BufferUsage) {
			kind, desc, err := webgpu.EncodeDescriptor(webgpu.BufferDescriptor{Size: size, Usage: u})
			if err != nil {
				t.Fatal(err)
			}
			if err := c.Create(id, uint32(kind), desc); err != nil {
				t.Fatal(err)
			}
		}
		mk(1, 64, webgpu.BufferUsageMapRead|webgpu.BufferUsageCopyDst)
		mk(2, 64, webgpu.BufferUsageStorage)
		return c, nil, 1, 2
	}
	rd := func(id uint32, off, size, win uint32) []uint32 {
		return []uint32{uint32(wire.ReadBuffer), id, off, 0, size, win, uint32(wire.Submit)}
	}
	for _, tc := range []struct {
		name   string
		cfg    Config
		be     func(*fake) Backend
		words  []uint32
		window int
	}{
		{"out of range of the buffer", Config{}, rb, rd(1, 32, 36, 0), 64},
		{"offset at the end plus 8", Config{}, rb, rd(1, 64, 8, 0), 64},
		{"over the window", Config{}, rb, rd(1, 0, 16, 8), 16},
		{"no window", Config{}, rb, rd(1, 0, 8, 0), 0},
		{"no MAP_READ", Config{}, rb, rd(2, 0, 8, 0), 64},
		{"unknown id", Config{}, rb, rd(9, 0, 8, 0), 64},
		{"unaligned offset", Config{}, rb, rd(1, 4, 8, 0), 64},
		{"unaligned size", Config{}, rb, rd(1, 0, 6, 0), 64},
		{"window above the quota", Config{MaxRead: 32}, rb, rd(1, 0, 8, 0), 64},
		{"reads above the quota", Config{MaxRead: 32}, rb, append(rd(1, 0, 24, 0)[:6:6], rd(1, 24, 16, 24)...), 40},
		{"backend cannot read", Config{}, func(f *fake) Backend { return f }, rd(1, 0, 8, 0), 64},
	} {
		f := newFake()
		c, _, _, _ := setup(tc.be(f), tc.cfg)
		err := c.Stream(tc.words, nil, make([]byte, tc.window))
		if err == nil {
			t.Errorf("%s: accepted", tc.name)
			continue
		}
		if f.replays != 0 {
			t.Errorf("%s: replayed", tc.name)
		}
		if c.Lost() == nil || !errors.Is(c.Stream([]uint32{uint32(wire.Submit)}, nil, nil), ErrLost) {
			t.Errorf("%s: context not lost", tc.name)
		}
	}
	// The same read, valid, works.
	f := newFake()
	c, _, _, _ := setup(rb(f), Config{})
	win := make([]byte, 16)
	if err := c.Stream(rd(1, 8, 8, 4), nil, win); err != nil || win[4] != 8 || win[0] != 0 || win[12] != 0 {
		t.Fatalf("%v %v", win, err)
	}
}

func rb(f *fake) Backend { return readFake{f} }

func TestContextMessages(t *testing.T) {
	f := newFake()
	c := New(f, Config{MaxObjects: 8})
	kind, desc, _ := webgpu.EncodeDescriptor(webgpu.BufferDescriptor{Size: 16, Usage: webgpu.BufferUsageStorage})
	clone := func() []byte { return append([]byte(nil), desc...) }
	if err := c.Create(1, uint32(kind), clone()); err != nil || c.Live() != 1 {
		t.Fatal(err)
	}
	if err := c.Destroy(1); err != nil || c.Live() != 0 || len(f.objs) != 0 {
		t.Fatal(err)
	}
	for name, f := range map[string]func(*Context) error{
		"id in use":    func(c *Context) error { c.Create(1, uint32(kind), clone()); return c.Create(1, uint32(kind), clone()) },
		"id 0":         func(c *Context) error { return c.Create(0, uint32(kind), clone()) },
		"id too large": func(c *Context) error { return c.Create(8, uint32(kind), clone()) },
		"bad kind":     func(c *Context) error { return c.Create(1, 99, clone()) },
		"bad desc":     func(c *Context) error { return c.Create(1, uint32(kind), []byte{1}) },
		"foreign ref": func(c *Context) error {
			return c.Create(1, uint32(wire.KindTextureView), []byte{1, 0, 0, 0, 5, 0, 0, 0})
		},
		"destroy dead": func(c *Context) error { return c.Destroy(3) },
		"stream id":    func(c *Context) error { return c.Stream([]uint32{uint32(wire.SetPipeline), 3}, nil, nil) },
	} {
		c := New(newFake(), Config{MaxObjects: 8})
		if err := f(c); err == nil || c.Lost() == nil {
			t.Errorf("%s: %v", name, err)
		}
		if err := c.Create(2, uint32(kind), clone()); !errors.Is(err, ErrLost) {
			t.Errorf("%s: after loss: %v", name, err)
		}
		if err := c.Destroy(2); !errors.Is(err, ErrLost) {
			t.Errorf("%s: destroy after loss: %v", name, err)
		}
	}
	for _, at := range []string{"create", "replay", "read"} {
		f := newFake()
		f.failAt = at
		c := New(readFake{f}, Config{})
		var err error
		if at == "create" {
			err = c.Create(1, uint32(kind), clone())
		} else {
			f.failAt = ""
			k, d, _ := webgpu.EncodeDescriptor(webgpu.BufferDescriptor{Size: 16, Usage: webgpu.BufferUsageMapRead})
			_ = c.Create(1, uint32(k), d)
			f.failAt = at
			err = c.Stream([]uint32{uint32(wire.ReadBuffer), 1, 0, 0, 4, 0, uint32(wire.Submit)}, nil, make([]byte, 4))
		}
		if err == nil || c.Lost() == nil {
			t.Errorf("backend %s failure: %v", at, err)
		}
	}
	c.Close()
	if !f.closed {
		t.Error("Close did not close the backend")
	}
}

// FuzzStream: whatever a guest sends, the context never panics, a refused
// stream replays nothing, and the reads of an accepted one touch only the
// window.
func FuzzStream(f *testing.F) {
	f.Add([]byte{34, 0, 0, 0, 1, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0}, uint16(32), uint32(64))
	f.Add([]byte{34, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0}, uint16(0), uint32(8))
	f.Fuzz(func(t *testing.T, raw []byte, win uint16, size uint32) {
		fk := newFake()
		c := New(readFake{fk}, Config{MaxRead: 1 << 10})
		for id, u := range map[uint32]webgpu.BufferUsage{1: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst, 2: webgpu.BufferUsageStorage} {
			k, d, _ := webgpu.EncodeDescriptor(webgpu.BufferDescriptor{Size: uint64(size), Usage: u})
			if c.Create(id, uint32(k), d) != nil {
				return
			}
		}
		w := make([]uint32, len(raw)/4)
		for i := range w {
			w[i] = uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
		}
		const guard = 8
		buf := make([]byte, int(win)+2*guard)
		for i := range buf {
			buf[i] = 0xAA
		}
		err := c.Stream(w, nil, buf[guard:guard+int(win)])
		for i := 0; i < guard; i++ {
			if buf[i] != 0xAA || buf[len(buf)-1-i] != 0xAA {
				t.Fatal("a read wrote outside the window")
			}
		}
		if err != nil && c.Lost() == nil {
			t.Fatal("an error that did not lose the context")
		}
		for _, r := range fk.reads {
			if r[0]+r[1] > uint64(size) || r[1] > uint64(win) {
				t.Fatalf("read %v outside buffer %d or window %d", r, size, win)
			}
		}
	})
}
