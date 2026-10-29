package timeline

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestOrderAndMonotonicity(t *testing.T) {
	var tl Timeline
	if tl.Next() != 1 || tl.Next() != 2 || tl.Next() != 3 || tl.Submitted() != 3 {
		t.Fatal("numbering")
	}
	ctx := context.Background()
	done := make(chan uint64, 3)
	var wg sync.WaitGroup
	for _, s := range []uint64{3, 1, 2} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := tl.Wait(ctx, s); err != nil {
				t.Error(err)
			}
			done <- s
		}()
	}
	for tl.waiting() < 3 {
		time.Sleep(time.Millisecond)
	}
	tl.Signal(2) // completes 1 and 2, not 3
	a, b := <-done, <-done
	if a+b != 3 {
		t.Fatalf("woke %d and %d", a, b)
	}
	select {
	case s := <-done:
		t.Fatalf("woke %d early", s)
	case <-time.After(20 * time.Millisecond):
	}
	tl.Signal(1) // backwards: ignored
	if tl.Completed() != 2 {
		t.Fatal(tl.Completed())
	}
	tl.Signal(3)
	wg.Wait()
	if err := tl.Wait(ctx, 2); err != nil { // already completed
		t.Fatal(err)
	}
}

func (t *Timeline) waiting() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.waiters)
}

func TestFailAndCancel(t *testing.T) {
	var tl Timeline
	tl.Next()
	tl.Next()
	tl.Signal(1)
	lost := errors.New("lost")
	errc := make(chan error, 1)
	go func() { errc <- tl.Wait(context.Background(), 2) }()
	for tl.waiting() < 1 {
		time.Sleep(time.Millisecond)
	}
	tl.Fail(lost)
	tl.Fail(errors.New("second")) // first failure wins
	if err := <-errc; !errors.Is(err, lost) {
		t.Fatal(err)
	}
	if err := tl.Wait(context.Background(), 5); !errors.Is(err, lost) {
		t.Fatal(err)
	}
	if err := tl.Wait(context.Background(), 1); err != nil {
		t.Fatal("a completed number after Fail:", err)
	}

	var t2 Timeline
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		for t2.waiting() < 1 {
			time.Sleep(time.Millisecond)
		}
		cancel()
	}()
	if err := t2.Wait(ctx, 1); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if t2.waiting() != 0 {
		t.Fatal("canceled waiter kept")
	}
	// A waiter woken by Signal racing with Fail sees completion.
	var t3 Timeline
	go func() {
		for t3.waiting() < 1 {
			time.Sleep(time.Millisecond)
		}
		t3.Signal(1)
	}()
	if err := t3.Wait(context.Background(), 1); err != nil {
		t.Fatal(err)
	}
}
