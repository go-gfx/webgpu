//go:build js

package webgpu

import (
	"context"
	"errors"
	"syscall/js"
	"unsafe"
)

// ErrNoReplay: replay.js is not loaded in this window or worker; see
// InstallReplay.
var ErrNoReplay = errors.New("webgpu: replay.js is not loaded (InstallReplay)")

// recorderStage holds a Recorder's two JavaScript arrays, reused from
// submit to submit and grown by doubling.
type recorderStage struct {
	cmd, data       js.Value
	cmdCap, dataCap int
}

func replayer() js.Value { return jsGlobal.Get("goWebGPUReplay") }

func init() {
	registry.drop = func(id uint32) {
		if r := replayer(); !nullish(r) {
			r.Call("drop", id)
		}
	}
}

// ReplayInstalled reports whether replay.js is loaded here.
func ReplayInstalled() bool { return !nullish(replayer()) }

// InstallReplay loads replay.js into this window or worker, once. url is
// where the page serves ReplayJS (same origin under a strict CSP); empty,
// the embedded copy is loaded from a blob: URL, which a CSP of
// script-src 'self' refuses. In a worker the script is loaded with
// importScripts (classic workers only); in a window with a <script>
// element.
func InstallReplay(ctx context.Context, url string) error {
	if ReplayInstalled() {
		return nil
	}
	if url == "" {
		var err error
		if err = try("InstallReplay", func() {
			blob := jsGlobal.Get("Blob").New([]any{ReplayJS}, map[string]any{"type": "text/javascript"})
			url = jsGlobal.Get("URL").Call("createObjectURL", blob).String()
		}); err != nil {
			return err
		}
		defer jsGlobal.Get("URL").Call("revokeObjectURL", url)
	}
	if is := jsGlobal.Get("importScripts"); is.Type() == js.TypeFunction {
		if err := try("importScripts", func() { is.Invoke(url) }); err != nil {
			return err
		}
	} else {
		doc := jsGlobal.Get("document")
		if nullish(doc) {
			return errors.New("webgpu: InstallReplay: neither a classic worker nor a window")
		}
		var p js.Value
		var onload, onerror js.Func
		if err := try("InstallReplay", func() {
			el := doc.Call("createElement", "script")
			el.Set("src", url)
			p = jsGlobal.Get("Promise").New(js.FuncOf(func(_ js.Value, a []js.Value) any {
				onload = js.FuncOf(func(js.Value, []js.Value) any { a[0].Invoke(); return nil })
				onerror = js.FuncOf(func(js.Value, []js.Value) any {
					a[1].Invoke(jsGlobal.Get("Error").New("could not load " + url))
					return nil
				})
				el.Set("onload", onload)
				el.Set("onerror", onerror)
				return nil
			}))
			doc.Get("head").Call("appendChild", el)
		}); err != nil {
			return err
		}
		_, err := await(ctx, "InstallReplay", p)
		onload.Release()
		onerror.Release()
		if err != nil {
			return err
		}
	}
	if !ReplayInstalled() {
		return ErrNoReplay
	}
	return nil
}

// Submit replays everything recorded in one call to JavaScript (one
// command encoder, finished and submitted at the end), then resets the
// Recorder for the next frame. It returns the first refused command, if
// any (refused commands were not recorded; the others were submitted),
// or what the browser threw. Every pass must be ended.
//
// seq numbers the submission on the device's timeline (from 1, shared
// with Queue.Submit); Device.WaitSubmitted(ctx, seq) waits for the GPU to
// finish it. It is 0 when nothing was replayed.
func (r *Recorder) Submit() (seq uint64, err error) {
	defer r.Reset()
	if err := r.finish(); err != nil {
		return 0, err
	}
	if r.dev.isLost() {
		return 0, ErrDestroyed
	}
	rp := replayer()
	if nullish(rp) {
		return 0, ErrNoReplay
	}
	for _, o := range r.unput {
		if o.markPut() {
			rp.Call("put", o.id, o.v)
		}
	}
	st := &r.stage
	nb := 4 * len(r.words)
	if nb > st.cmdCap {
		st.cmdCap = max(4096, 2*st.cmdCap, nb)
		st.cmd = jsUint8Array.New(st.cmdCap)
	}
	if len(r.data) > st.dataCap || st.data.IsUndefined() {
		st.dataCap = max(4096, 2*st.dataCap, len(r.data))
		st.data = jsUint8Array.New(st.dataCap)
	}
	js.CopyBytesToJS(st.cmd, unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(r.words))), nb))
	if len(r.data) > 0 {
		js.CopyBytesToJS(st.data, r.data)
	}
	if err := try("replay", func() {
		rp.Call("replay", r.dev.v, r.dev.queue.v, st.cmd, len(r.words), st.data, len(r.data))
	}); err != nil {
		return 0, err
	}
	return r.dev.tl.Next(), r.err
}
