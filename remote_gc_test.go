//go:build !js

package webgpu

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/go-gfx/webgpu/wire"
)

// hostSink is a remote device's host as far as ids go, like batch.Context: it
// refuses the CREATE of an id that is live and the DESTROY of one that is not.
type hostSink struct {
	mu   sync.Mutex // Destroy also comes from the runtime's cleanup goroutine
	live map[uint32]bool
}

func (s *hostSink) isLive(id uint32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.live[id]
}

func (s *hostSink) Create(id uint32, _ wire.Kind, _ []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if id == 0 || s.live[id] {
		return fmt.Errorf("batch: message refused: CREATE %d", id)
	}
	s.live[id] = true
	return nil
}
func (s *hostSink) Destroy(id uint32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.live[id] {
		return fmt.Errorf("batch: message refused: DESTROY %d", id)
	}
	delete(s.live, id)
	return nil
}
func (s *hostSink) Submit([]uint32, []byte) error { return nil }

func collect() {
	for i := 0; i < 5; i++ {
		runtime.GC()
		time.Sleep(20 * time.Millisecond) // cleanups run in a goroutine of their own
	}
}

// TestCollectedHandleNeverYieldsALiveID: a handle garbage collected without
// Destroy has its host object destroyed before its id is reused, so the next
// CREATE is not refused; a handle destroyed explicitly is not destroyed twice.
func TestCollectedHandleNeverYieldsALiveID(t *testing.T) {
	s := &hostSink{live: map[uint32]bool{}}
	dev := NewRemoteDevice(s, "gc", nil)
	desc := BufferDescriptor{Size: 16, Usage: BufferUsageCopyDst}
	func() {
		for i := 0; i < 32; i++ {
			if _, err := dev.CreateBuffer(desc); err != nil { // dropped at once
				t.Fatal(err)
			}
		}
		b, err := dev.CreateBuffer(desc)
		if err != nil {
			t.Fatal(err)
		}
		b.Destroy() // destroyed by hand, then collected
	}()
	collect()
	var keep []*Buffer
	for i := 0; i < 64; i++ {
		b, err := dev.CreateBuffer(desc)
		if err != nil {
			t.Fatalf("buffer %d after a collection: %v", i, err)
		}
		keep = append(keep, b)
	}
	collect()
	if dev.isLost() {
		t.Fatal("the device was lost")
	}
	runtime.KeepAlive(keep)
}

// TestBindGroupKeepsItsResources: the host's bind group uses its buffer and
// layout, so collecting the Go handles the caller dropped must not destroy
// them while the bind group is reachable.
func TestBindGroupKeepsItsResources(t *testing.T) {
	s := &hostSink{live: map[uint32]bool{}}
	dev := NewRemoteDevice(s, "deps", nil)
	var bg *BindGroup
	var bufID uint32
	func() {
		buf, err := dev.CreateBuffer(BufferDescriptor{Size: 16, Usage: BufferUsageUniform | BufferUsageCopyDst})
		if err != nil {
			t.Fatal(err)
		}
		bufID = idOf(&buf.object)
		bgl, err := dev.CreateBindGroupLayout(BindGroupLayoutDescriptor{Entries: []BindGroupLayoutEntry{
			{Binding: 0, Visibility: ShaderStageVertex, Buffer: &BufferBindingLayout{Type: BufferBindingTypeUniform}}}})
		if err != nil {
			t.Fatal(err)
		}
		bg, err = dev.CreateBindGroup(BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, Buffer: buf}}})
		if err != nil {
			t.Fatal(err)
		}
	}()
	collect()
	if !s.isLive(bufID) {
		t.Fatal("the bind group's buffer was destroyed while the bind group is reachable")
	}
	runtime.KeepAlive(bg)
}

// TestDescDeps: the objects each creation descriptor names.
func TestDescDeps(t *testing.T) {
	tex, view, smp, buf := &Texture{}, &TextureView{}, &Sampler{}, &Buffer{}
	lay, mod, mod2 := &BindGroupLayout{}, &ShaderModule{}, &ShaderModule{}
	pl := &PipelineLayout{}
	for _, c := range []struct {
		name string
		desc any
		want int
	}{
		{"view", TextureViewCreation{Texture: tex}, 1},
		{"pipeline layout", PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{lay, lay}}, 2},
		{"bind group", BindGroupDescriptor{Layout: lay, Entries: []BindGroupEntry{{Buffer: buf}, {Sampler: smp}, {TextureView: view}}}, 4},
		{"render, all", RenderPipelineDescriptor{Layout: pl, Vertex: VertexState{Module: mod}, Fragment: &FragmentState{Module: mod2}}, 3},
		{"render, none", RenderPipelineDescriptor{Vertex: VertexState{Module: mod}}, 1},
		{"compute", ComputePipelineDescriptor{Layout: pl, Compute: ProgrammableStage{Module: mod}}, 2},
		{"compute, no layout", ComputePipelineDescriptor{Compute: ProgrammableStage{Module: mod}}, 1},
		{"buffer", BufferDescriptor{}, 0},
	} {
		if got := len(descDeps(c.desc)); got != c.want {
			t.Errorf("%s: %d dependencies, want %d", c.name, got, c.want)
		}
	}
}
