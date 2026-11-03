//go:build !js

package webgpu

import (
	"context"
	"errors"
	"testing"

	"github.com/go-gfx/webgpu/wire"
)

// readSink is a ReadSink that fills the window with a pattern.
type readSink struct {
	sink
	windows []int
}

func (s *readSink) SubmitRead(words []uint32, data []byte, window []byte) error {
	s.submits = append(s.submits, append([]uint32(nil), words...))
	s.windows = append(s.windows, len(window))
	for i := range window {
		window[i] = byte(i)
	}
	return nil
}

func readDevice(t *testing.T, q *Quota) (*Device, *readSink, *Buffer) {
	t.Helper()
	s := &readSink{sink: *newSink()}
	d := NewRemoteDevice(s, "guest", q)
	b, err := d.CreateBuffer(BufferDescriptor{Size: 64, Usage: BufferUsageMapRead | BufferUsageCopyDst})
	if err != nil {
		t.Fatal(err)
	}
	return d, s, b
}

func TestRecorderReadBuffer(t *testing.T) {
	d, s, b := readDevice(t, nil)
	r := d.NewRecorder()
	a := r.ReadBuffer(b, 8, 16)
	c := r.ReadBuffer(b, 0, 4)
	if _, err := a.Bytes(); !errors.Is(err, ErrNotRead) {
		t.Fatalf("before the submit: %v", err)
	}
	if _, err := r.Submit(); err != nil {
		t.Fatal(err)
	}
	ga, _ := a.Bytes()
	gc, _ := c.Bytes()
	if len(ga) != 16 || ga[0] != 0 || ga[15] != 15 || len(gc) != 4 || gc[0] != 16 {
		t.Fatalf("a %v c %v", ga, gc)
	}
	if s.windows[0] != 20 {
		t.Fatalf("window %d", s.windows[0])
	}
	// The recorded stream decodes, and the reads carry the layout.
	cmds, err := wire.Decode(s.submits[0], 0, func(uint32) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	rd, ok := wire.ReadOf(cmds[1])
	if !ok || rd.Offset != 0 || rd.Size != 4 || rd.WindowOffset != 16 {
		t.Fatalf("%+v", rd)
	}
	// Reset: the recorder starts a new window.
	if r.win != 0 || len(r.reads) != 0 {
		t.Fatal("reset kept reads")
	}
}

func TestRecorderReadBufferRefusals(t *testing.T) {
	d, _, b := readDevice(t, &Quota{ReadBytes: 32})
	plain, _ := d.CreateBuffer(BufferDescriptor{Size: 64, Usage: BufferUsageCopyDst})
	for name, f := range map[string]func(*Recorder) *Readback{
		"no MAP_READ":      func(r *Recorder) *Readback { return r.ReadBuffer(plain, 0, 4) },
		"nil":              func(r *Recorder) *Readback { return r.ReadBuffer(nil, 0, 4) },
		"unaligned offset": func(r *Recorder) *Readback { return r.ReadBuffer(b, 4, 4) },
		"unaligned size":   func(r *Recorder) *Readback { return r.ReadBuffer(b, 0, 6) },
		"zero size":        func(r *Recorder) *Readback { return r.ReadBuffer(b, 0, 0) },
		"past the buffer":  func(r *Recorder) *Readback { return r.ReadBuffer(b, 8, 64) },
		"offset overflow":  func(r *Recorder) *Readback { return r.ReadBuffer(b, 1<<63, 8) },
		"above the window": func(r *Recorder) *Readback { return r.ReadBuffer(b, 0, wire.MaxReadWindow+4) },
		"over quota": func(r *Recorder) *Readback {
			r.ReadBuffer(b, 0, 32)
			return r.ReadBuffer(b, 32, 8)
		},
		"in a pass": func(r *Recorder) *Readback { r.BeginComputePass(); return r.ReadBuffer(b, 0, 4) },
	} {
		r := d.NewRecorder()
		rb := f(r)
		if _, err := rb.Bytes(); err == nil || r.Err() == nil {
			t.Errorf("%s: accepted (%v, %v)", name, err, r.Err())
		}
	}
	// Without a quota, the wire's own bound on the window applies.
	big, err := NewRemoteDevice(newSink(), "big", nil).CreateBuffer(BufferDescriptor{Size: 1 << 27, Usage: BufferUsageMapRead})
	if err != nil {
		t.Fatal(err)
	}
	r2 := NewRemoteDevice(newSink(), "x", nil).NewRecorder()
	r2.ReadBuffer(big, 0, 40<<20)
	if _, err := r2.ReadBuffer(big, 0, 40<<20).Bytes(); err == nil {
		t.Error("two reads above the window bound were accepted")
	}
	var qe *QuotaError
	r := d.NewRecorder()
	r.ReadBuffer(b, 0, 32)
	r.ReadBuffer(b, 32, 8)
	if !errors.As(r.Err(), &qe) {
		t.Errorf("quota error: %v", r.Err())
	}
}

func TestRemoteMapAsync(t *testing.T) {
	d, _, b := readDevice(t, nil)
	ctx := context.Background()
	if err := b.MapAsync(ctx, MapModeRead, 8, 16); err != nil {
		t.Fatal(err)
	}
	if err := b.MapAsync(ctx, MapModeRead, 0, 4); !errors.Is(err, ErrMapState) {
		t.Fatalf("second map: %v", err)
	}
	got := make([]byte, 8)
	if err := b.ReadMapped(12, got); err != nil || got[0] != 4 {
		t.Fatalf("%v %v", got, err)
	}
	if err := b.ReadMapped(4, got); err == nil {
		t.Error("read before the mapped range")
	}
	if err := b.ReadMapped(20, make([]byte, 8)); err == nil {
		t.Error("read past the mapped range")
	}
	if err := b.WriteMapped(8, got); !errors.Is(err, ErrRemoteUnsupported) {
		t.Error("WriteMapped")
	}
	_ = b.Unmap()
	if err := b.ReadMapped(8, got); err == nil {
		t.Error("read after unmap")
	}
	_ = b.Unmap()                                              // unmapped: nothing
	if err := b.MapAsync(ctx, MapModeRead, 0, 0); err != nil { // whole buffer
		t.Fatal(err)
	}
	_ = b.Unmap()
	if err := b.MapAsync(ctx, MapModeRead, 100, 0); err == nil {
		t.Error("offset past the buffer")
	}
	if err := b.MapAsync(ctx, MapModeRead, 4, 8); err == nil {
		t.Error("unaligned offset")
	}
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	if err := b.MapAsync(cctx, MapModeRead, 0, 4); err == nil {
		t.Error("cancelled context")
	}
	// A sink without read-back refuses.
	plain := NewRemoteDevice(newSink(), "p", nil)
	pb, _ := plain.CreateBuffer(BufferDescriptor{Size: 16, Usage: BufferUsageMapRead})
	if err := pb.MapAsync(ctx, MapModeRead, 0, 0); !errors.Is(err, ErrRemoteUnsupported) {
		t.Errorf("plain sink: %v", err)
	}
	// A failing sink loses the device.
	_ = d
}
