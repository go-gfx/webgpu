// Package timeline is a monotonic synchronisation timeline: work is
// numbered 1, 2, 3, ... as it is submitted, and completing number n
// completes every number up to n. It is the fence model of virtio-gpu
// (virtio specification, device-types/gpu/description.tex: with a ring
// index, "fence_id acts as a sequence number on the synchronization
// timeline", and when it completes "the device MUST send a response for
// all outstanding commands with a sequence number less than or equal to
// fence_id"), and of WebGPU's queue (onSubmittedWorkDone resolves once all
// work submitted so far is done).
//
// It has no dependency and no notion of a GPU: one implementation for any
// producer that numbers submissions and learns of completions in order.
package timeline

import (
	"context"
	"sync"
)

// Timeline is safe for concurrent use. The zero value is ready: nothing
// submitted, nothing completed.
type Timeline struct {
	mu        sync.Mutex
	submitted uint64
	completed uint64
	err       error
	waiters   []waiter
}

type waiter struct {
	seq uint64
	ch  chan struct{}
}

// Next numbers one more submission and returns its number.
func (t *Timeline) Next() uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.submitted++
	return t.submitted
}

// Submitted is the number of the last submission.
func (t *Timeline) Submitted() uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.submitted
}

// Completed is the highest completed number.
func (t *Timeline) Completed() uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.completed
}

// Signal completes every number up to seq. A seq at or below Completed
// is ignored: completions only move forward.
func (t *Timeline) Signal(seq uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if seq <= t.completed {
		return
	}
	t.completed = seq
	keep := t.waiters[:0]
	for _, w := range t.waiters {
		if w.seq <= seq {
			close(w.ch)
		} else {
			keep = append(keep, w)
		}
	}
	clear(t.waiters[len(keep):])
	t.waiters = keep
}

// Fail ends the timeline: every waiter, now and later, gets err for a
// number not yet completed (a lost device completes nothing more).
func (t *Timeline) Fail(err error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.err != nil {
		return
	}
	t.err = err
	for _, w := range t.waiters {
		close(w.ch)
	}
	t.waiters = nil
}

// Wait returns when seq is completed, the timeline failed, or ctx is
// done. It returns at once for a number already completed.
func (t *Timeline) Wait(ctx context.Context, seq uint64) error {
	t.mu.Lock()
	if seq <= t.completed {
		t.mu.Unlock()
		return nil
	}
	if t.err != nil {
		t.mu.Unlock()
		return t.err
	}
	ch := make(chan struct{})
	t.waiters = append(t.waiters, waiter{seq, ch})
	t.mu.Unlock()
	select {
	case <-ch:
		t.mu.Lock()
		defer t.mu.Unlock()
		if seq <= t.completed {
			return nil
		}
		return t.err
	case <-ctx.Done():
		t.mu.Lock()
		for i, w := range t.waiters {
			if w.ch == ch {
				t.waiters = append(t.waiters[:i], t.waiters[i+1:]...)
				break
			}
		}
		t.mu.Unlock()
		return ctx.Err()
	}
}
