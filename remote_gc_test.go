//go:build !js

package webgpu

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/go-gfx/webgpu/wire"
)

// hostSink is a remote device's host as far as ids go, like batch.Context: it
// refuses the CREATE of an id that is live and the DESTROY of one that is not.
type hostSink struct {
	live map[uint32]bool
}

func (s *hostSink) Create(id uint32, _ wire.Kind, _ []byte) error {
	if id == 0 || s.live[id] {
		return fmt.Errorf("batch: message refused: CREATE %d", id)
	}
	s.live[id] = true
	return nil
}
func (s *hostSink) Destroy(id uint32) error {
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
