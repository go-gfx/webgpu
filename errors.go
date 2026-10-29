package webgpu

import (
	"errors"
	"fmt"
)

var (
	// ErrNoWebGPU: this context has no navigator.gpu (no WebGPU, or not a
	// secure context, or not GOOS=js).
	ErrNoWebGPU = errors.New("webgpu: navigator.gpu is not available")
	// ErrNoAdapter: requestAdapter resolved to null.
	ErrNoAdapter = errors.New("webgpu: no adapter")
	// ErrFallbackAdapter: the only adapter is a fallback (software) one and
	// the caller did not accept it. Work on such an adapter is not GPU work.
	ErrFallbackAdapter = errors.New("webgpu: only a fallback (software) adapter")
	// ErrDestroyed: the object (or its device) was destroyed or released.
	ErrDestroyed = errors.New("webgpu: object destroyed")
	// ErrNoCanvasContext: OffscreenCanvas.getContext("webgpu") returned
	// null here (e.g. Firefox outside Windows); fall back to another
	// renderer.
	ErrNoCanvasContext = errors.New("webgpu: no WebGPU context on OffscreenCanvas")
)

// QuotaError reports a request a Quota refused.
type QuotaError struct {
	Resource string // "memory", "pipelines", "shader", "feature", "limit"
	Limit    int64
	Want     int64
	Name     string // the feature or limit, when Resource names one
}

func (e *QuotaError) Error() string {
	if e.Name != "" {
		return fmt.Sprintf("webgpu: quota: %s %q refused (allowed %d, requested %d)", e.Resource, e.Name, e.Limit, e.Want)
	}
	return fmt.Sprintf("webgpu: quota: %s: %d requested, %d allowed", e.Resource, e.Want, e.Limit)
}

// BoundsError reports an argument the package refused before it reached
// the browser: an out-of-range write, a misaligned offset, a missing usage.
// WebGPU itself would report most of these only asynchronously, as a
// validation error on the device; here they are an error at the call.
type BoundsError struct {
	Op  string
	Msg string
}

func (e *BoundsError) Error() string { return "webgpu: " + e.Op + ": " + e.Msg }

func bounds(op, format string, a ...any) error {
	return &BoundsError{Op: op, Msg: fmt.Sprintf(format, a...)}
}

// JSError is an exception the browser threw synchronously (a TypeError for
// a malformed descriptor, an OperationError, ...).
type JSError struct {
	Op      string
	Name    string
	Message string
}

func (e *JSError) Error() string {
	return fmt.Sprintf("webgpu: %s: %s: %s", e.Op, e.Name, e.Message)
}

// GPUError is a WebGPU error reported on the device: by PopErrorScope or
// to an uncaptured-error handler. Kind is "validation", "out-of-memory" or
// "internal".
type GPUError struct {
	Kind    string
	Message string
}

func (e *GPUError) Error() string { return "webgpu: " + e.Kind + " error: " + e.Message }

// DeviceLostInfo says why a device was lost: Reason "destroyed" (Destroy
// was called) or "unknown".
type DeviceLostInfo struct {
	Reason  string
	Message string
}
