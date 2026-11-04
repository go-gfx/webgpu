package webgpu

import (
	"sync/atomic"
	"testing"
	"time"
)

func TestKickLoopKicksUntilStopped(t *testing.T) {
	stop := make(chan struct{})
	var n atomic.Int64
	done := make(chan struct{})
	go func() { kickLoop(stop, func() { n.Add(1) }); close(done) }()
	deadline := time.After(5 * time.Second)
	for n.Load() < 3 {
		select {
		case <-deadline:
			t.Fatalf("only %d kicks in 5 s", n.Load())
		default:
			time.Sleep(time.Millisecond)
		}
	}
	close(stop)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("kickLoop did not return after stop")
	}
	after := n.Load()
	time.Sleep(3 * kickMax)
	if n.Load() != after {
		t.Errorf("kicked after stop: %d -> %d", after, n.Load())
	}
}

func TestKickLoopNeverKicksWhenStoppedFirst(t *testing.T) {
	stop := make(chan struct{})
	close(stop)
	var n atomic.Int64
	kickLoop(stop, func() { n.Add(1) })
	if n.Load() != 0 {
		t.Errorf("%d kicks for an already finished wait", n.Load())
	}
}

func TestKickGapsDoubleToTheCap(t *testing.T) {
	gap := kickFirst
	steps := 0
	for gap < kickMax {
		gap = min(gap*2, kickMax)
		steps++
	}
	if steps != 4 || gap != kickMax {
		t.Errorf("gap reached %v in %d steps", gap, steps)
	}
}
