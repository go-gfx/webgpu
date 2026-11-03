//go:build js

package batch

import (
	"context"
	"errors"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
)

// End to end on the browser's GPU (run in a worker by internal/cmd/wasmbrowser;
// WEBGPU_REQUIRE_GPU=1 fails a fallback adapter). The client is guest-like:
// it owns bare handles for their ids, creates objects through the Context
// from wire descriptors, and records the stream as words, as a guest does.

func tctx(t *testing.T) context.Context {
	c, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return c
}

func gpuDevice(t *testing.T) *webgpu.Device {
	t.Helper()
	g, err := webgpu.Navigator()
	if err != nil {
		t.Fatal(err)
	}
	a, err := g.RequestAdapter(tctx(t), &webgpu.RequestAdapterOptions{AllowFallback: os.Getenv("WEBGPU_ALLOW_FALLBACK") == "1"})
	if err != nil {
		t.Fatal(err)
	}
	info := a.Info()
	t.Logf("adapter: vendor %q, architecture %q, description %q, fallback %v", info.Vendor, info.Architecture, info.Description, info.Fallback)
	if os.Getenv("WEBGPU_REQUIRE_GPU") == "1" && info.Fallback {
		t.Fatal("a fallback adapter where a GPU is required")
	}
	d, err := a.RequestDevice(tctx(t), &webgpu.DeviceDescriptor{Label: t.Name()})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Destroy)
	if err := webgpu.InstallReplay(tctx(t), ""); err != nil {
		t.Fatal(err)
	}
	return d
}

type guestCtx struct {
	t *testing.T
	c *Context
}

// make creates the object h stands for (its replay id is the guest id).
func (g guestCtx) make(h, desc any) uint32 {
	g.t.Helper()
	id, err := webgpu.ReplayID(h)
	if err != nil {
		g.t.Fatal(err)
	}
	kind, b, err := webgpu.EncodeDescriptor(desc)
	if err != nil {
		g.t.Fatal(err)
	}
	if err := g.c.Create(id, uint32(kind), b); err != nil {
		g.t.Fatalf("create %d: %v", id, err)
	}
	return id
}

const shader = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&out)) { out[id.x] = id.x * 2654435761u + 7u; }
}`

// setup creates the compute pipeline, a storage buffer and a MAP_READ
// buffer of n words, and returns the ids.
func (g guestCtx) setup(n uint64) (pipe, bg, src, dst uint32) {
	m, bgl, pl := new(webgpu.ShaderModule), new(webgpu.BindGroupLayout), new(webgpu.PipelineLayout)
	g.make(m, webgpu.ShaderModuleDescriptor{Code: shader})
	g.make(bgl, webgpu.BindGroupLayoutDescriptor{Entries: []webgpu.BindGroupLayoutEntry{
		{Binding: 0, Visibility: webgpu.ShaderStageCompute, Buffer: &webgpu.BufferBindingLayout{Type: webgpu.BufferBindingTypeStorage}}}})
	g.make(pl, webgpu.PipelineLayoutDescriptor{BindGroupLayouts: []*webgpu.BindGroupLayout{bgl}})
	sb, db := new(webgpu.Buffer), new(webgpu.Buffer)
	src = g.make(sb, webgpu.BufferDescriptor{Size: 4 * n, Usage: webgpu.BufferUsageStorage | webgpu.BufferUsageCopySrc})
	dst = g.make(db, webgpu.BufferDescriptor{Size: 4 * n, Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst})
	gp, gb := new(webgpu.ComputePipeline), new(webgpu.BindGroup)
	pipe = g.make(gp, webgpu.ComputePipelineDescriptor{Layout: pl, Compute: webgpu.ProgrammableStage{Module: m, EntryPoint: "main"}})
	bg = g.make(gb, webgpu.BindGroupDescriptor{Layout: bgl, Entries: []webgpu.BindGroupEntry{{Binding: 0, Buffer: sb}}})
	return
}

// stream records: dispatch, copy to the MAP_READ buffer, read [off, off+size).
func stream(pipe, bg, src, dst uint32, groups uint32, copyBytes uint32, reads ...[3]uint32) []uint32 {
	w := []uint32{
		uint32(wire.BeginComputePass),
		uint32(wire.SetPipeline), pipe,
		uint32(wire.SetBindGroup), 0, bg, 0,
		uint32(wire.Dispatch), groups, 1, 1,
		uint32(wire.EndPass),
		uint32(wire.CopyBufferToBuffer), src, 0, 0, dst, 0, 0, copyBytes, 0,
	}
	for _, r := range reads { // offset, size, window offset
		w = append(w, uint32(wire.ReadBuffer), dst, r[0], 0, r[1], r[2])
	}
	return append(w, uint32(wire.Submit))
}

func expected(i uint32) uint32 { return i*2654435761 + 7 }

func check(t *testing.T, got []byte, first uint32) {
	t.Helper()
	for i := 0; i < len(got)/4; i++ {
		v := uint32(got[4*i]) | uint32(got[4*i+1])<<8 | uint32(got[4*i+2])<<16 | uint32(got[4*i+3])<<24
		if want := expected(first + uint32(i)); v != want {
			t.Fatalf("element %d: %#x, want %#x", first+uint32(i), v, want)
		}
	}
}

// TestReadBackOnGPU: a compute shader writes known values, they are copied
// to a MAP_READ buffer and read back through the window; every element
// matches. 4 KiB, 1 MiB and a window that holds two disjoint ranges.
func TestReadBackOnGPU(t *testing.T) {
	d := gpuDevice(t)
	for _, n := range []uint32{1024, 262144} {
		c := New(NewDeviceBackend(d), Config{MaxRead: 1 << 20})
		g := guestCtx{t, c}
		pipe, bg, src, dst := g.setup(uint64(n))
		win := make([]byte, 4*n)
		if err := c.Stream(stream(pipe, bg, src, dst, (n+63)/64, 4*n, [3]uint32{0, 4 * n, 0}), nil, win); err != nil {
			t.Fatal(err)
		}
		check(t, win, 0)
		t.Logf("%d bytes: all %d elements match", len(win), n)
		c.Close()
	}
	c := New(NewDeviceBackend(d), Config{})
	g := guestCtx{t, c}
	pipe, bg, src, dst := g.setup(1024)
	win := make([]byte, 128)
	if err := c.Stream(stream(pipe, bg, src, dst, 16, 4096, [3]uint32{800, 64, 0}, [3]uint32{4000, 64, 64}), nil, win); err != nil {
		t.Fatal(err)
	}
	check(t, win[:64], 200)
	check(t, win[64:], 1000)
	c.Close()
}

// TestReadRefusalsOnGPU: out of range, over quota, and a fallback adapter.
func TestReadRefusalsOnGPU(t *testing.T) {
	d := gpuDevice(t)
	for name, tc := range map[string]struct {
		cfg   Config
		reads [][3]uint32
		win   int
	}{
		"out of range":        {Config{}, [][3]uint32{{4096 - 32, 64, 0}}, 64},
		"offset past the end": {Config{}, [][3]uint32{{8192, 8, 0}}, 8},
		"outside the window":  {Config{}, [][3]uint32{{0, 64, 32}}, 64},
		"over quota":          {Config{MaxRead: 1024}, [][3]uint32{{0, 2048, 0}}, 2048},
	} {
		c := New(NewDeviceBackend(d), tc.cfg)
		g := guestCtx{t, c}
		pipe, bg, src, dst := g.setup(1024)
		err := c.Stream(stream(pipe, bg, src, dst, 16, 4096, tc.reads...), nil, make([]byte, tc.win))
		if !errors.Is(err, ErrRead) {
			t.Errorf("%s: %v", name, err)
		}
		if c.Lost() == nil {
			t.Errorf("%s: context not lost", name)
		}
		c.Close()
	}
	// A fallback adapter is refused unless allowed: the request below asks
	// for one without AllowFallback.
	g, err := webgpu.Navigator()
	if err != nil {
		t.Fatal(err)
	}
	a, err := g.RequestAdapter(tctx(t), &webgpu.RequestAdapterOptions{ForceFallbackAdapter: true})
	if err == nil {
		t.Errorf("a forced fallback adapter was granted (fallback %v)", a.Info().Fallback)
	} else if !errors.Is(err, webgpu.ErrFallbackAdapter) && !errors.Is(err, webgpu.ErrNoAdapter) {
		t.Logf("fallback refused with: %v", err)
	}
}

// TestReadLatency measures the round trip of one read-back submit (replay,
// GPU wait, mapAsync, copy, unmap) for 4 KiB and 1 MiB windows, interleaved,
// with and without the compute work. It logs; it asserts nothing.
func TestReadLatency(t *testing.T) {
	d := gpuDevice(t)
	type lane struct {
		name string
		c    *Context
		w    []uint32
		win  []byte
	}
	var lanes []lane
	for _, n := range []uint32{1024, 262144} {
		c := New(NewDeviceBackend(d), Config{MaxRead: 1 << 20})
		g := guestCtx{t, c}
		pipe, bg, src, dst := g.setup(uint64(n))
		label := map[uint32]string{1024: "4KiB", 262144: "1MiB"}[n]
		lanes = append(lanes,
			lane{label + " dispatch+copy+read", c, stream(pipe, bg, src, dst, (n+63)/64, 4*n, [3]uint32{0, 4 * n, 0}), make([]byte, 4*n)},
			lane{label + " read only", c, []uint32{uint32(wire.ReadBuffer), dst, 0, 0, 4 * n, 0, uint32(wire.Submit)}, make([]byte, 4*n)})
	}
	times := make([][]float64, len(lanes))
	for round := 0; round < 33; round++ {
		for i, l := range lanes {
			w := slices.Clone(l.w)
			start := time.Now()
			if err := l.c.Stream(w, nil, l.win); err != nil {
				t.Fatal(err)
			}
			if round > 2 { // three warm-up rounds
				times[i] = append(times[i], float64(time.Since(start).Microseconds()))
			}
		}
	}
	for i, l := range lanes {
		slices.Sort(times[i])
		t.Logf("%-26s median %7.0f us  min %7.0f  max %7.0f  (30 rounds)", l.name, times[i][len(times[i])/2], times[i][0], times[i][len(times[i])-1])
	}
}
