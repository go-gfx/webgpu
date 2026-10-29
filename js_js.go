//go:build js

package webgpu

import (
	"context"
	"fmt"
	"syscall/js"
)

type jsValue = js.Value

func (o *object) ref() any { return o.v }

// Value returns the underlying browser object, for interoperation with
// other syscall/js code. Using it to destroy or mutate the object behind
// the package's back defeats its accounting.
func (o *object) Value() js.Value { return o.v }

var (
	jsGlobal     = js.Global()
	jsUint8Array = jsGlobal.Get("Uint8Array")
)

// try runs f, which calls the browser, and turns a thrown exception into
// a *JSError.
func try(op string, f func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			switch e := r.(type) {
			case js.Error:
				err = &JSError{Op: op, Name: str(e.Value.Get("name")), Message: str(e.Value.Get("message"))}
			case error:
				err = &JSError{Op: op, Name: "Error", Message: e.Error()}
			default:
				err = &JSError{Op: op, Name: "Error", Message: fmt.Sprint(r)}
			}
		}
	}()
	f()
	return nil
}

func str(v js.Value) string {
	if v.Type() == js.TypeString {
		return v.String()
	}
	if v.IsUndefined() || v.IsNull() {
		return ""
	}
	return v.Call("toString").String()
}

func nullish(v js.Value) bool { return v.IsUndefined() || v.IsNull() }

// await waits for a promise without blocking the event loop: only the
// calling goroutine waits. It must not be called from a js.FuncOf
// callback, whose goroutine holds the event loop.
func await(ctx context.Context, op string, p js.Value) (js.Value, error) {
	type res struct {
		v   js.Value
		err error
	}
	ch := make(chan res, 1)
	ok := js.FuncOf(func(_ js.Value, a []js.Value) any {
		v := js.Undefined()
		if len(a) > 0 {
			v = a[0]
		}
		ch <- res{v: v}
		return nil
	})
	fail := js.FuncOf(func(_ js.Value, a []js.Value) any {
		e := &JSError{Op: op, Name: "Error", Message: "rejected"}
		if len(a) > 0 {
			e.Name, e.Message = str(a[0].Get("name")), str(a[0].Get("message"))
			if e.Message == "" {
				e.Message = str(a[0])
			}
		}
		ch <- res{err: e}
		return nil
	})
	release := func() {
		ok.Release()
		fail.Release()
	}
	if err := try(op, func() { p.Call("then", ok, fail) }); err != nil {
		release()
		return js.Value{}, err
	}
	select {
	case r := <-ch:
		release()
		return r.v, r.err
	case <-ctx.Done():
		go func() { <-ch; release() }()
		return js.Value{}, ctx.Err()
	}
}

// bytesToJS copies b into a Uint8Array of a reusable staging area and
// returns it. The staging array only grows, by powers of two.
type staging struct{ u js.Value }

func (s *staging) load(b []byte) js.Value {
	if s.u.IsUndefined() || s.u.Get("length").Int() < len(b) {
		n := 4096
		for n < len(b) {
			n <<= 1
		}
		s.u = jsUint8Array.New(n)
	}
	js.CopyBytesToJS(s.u, b)
	return s.u
}
