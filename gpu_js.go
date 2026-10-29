//go:build js

package webgpu

import (
	"context"
	"syscall/js"
)

// GPU is navigator.gpu of the window or worker the program runs in.
type GPU struct{ v js.Value }

// Navigator returns navigator.gpu, or ErrNoWebGPU.
func Navigator() (*GPU, error) {
	nav := jsGlobal.Get("navigator")
	if nullish(nav) {
		return nil, ErrNoWebGPU
	}
	g := nav.Get("gpu")
	if nullish(g) {
		return nil, ErrNoWebGPU
	}
	return &GPU{v: g}, nil
}

// PreferredCanvasFormat is navigator.gpu.getPreferredCanvasFormat().
func (g *GPU) PreferredCanvasFormat() TextureFormat {
	return TextureFormat(g.v.Call("getPreferredCanvasFormat").String())
}

// Adapter is a GPUAdapter.
type Adapter struct {
	object
	info     AdapterInfo
	features map[FeatureName]bool
	limits   map[string]uint64
}

// RequestAdapter is navigator.gpu.requestAdapter(). It answers
// ErrNoAdapter when the browser offers none, and ErrFallbackAdapter for a
// fallback adapter unless opts.AllowFallback.
func (g *GPU) RequestAdapter(ctx context.Context, opts *RequestAdapterOptions) (*Adapter, error) {
	var p js.Value
	if err := try("requestAdapter", func() { p = g.v.Call("requestAdapter", js.ValueOf(opts.toJS())) }); err != nil {
		return nil, err
	}
	a, err := await(ctx, "requestAdapter", p)
	if err != nil {
		return nil, err
	}
	if nullish(a) {
		return nil, ErrNoAdapter
	}
	ad := &Adapter{object: object{v: a}, info: adapterInfo(a), features: features(a.Get("features")), limits: limits(a.Get("limits"))}
	if ad.info.Fallback && (opts == nil || !opts.AllowFallback) {
		return nil, ErrFallbackAdapter
	}
	return ad, nil
}

func adapterInfo(a js.Value) AdapterInfo {
	ai := AdapterInfo{}
	if info := a.Get("info"); !nullish(info) {
		ai.Vendor, ai.Architecture = str(info.Get("vendor")), str(info.Get("architecture"))
		ai.Device, ai.Description = str(info.Get("device")), str(info.Get("description"))
		ai.Fallback = info.Get("isFallbackAdapter").Truthy()
	}
	// Older browsers put the flag on the adapter.
	ai.Fallback = ai.Fallback || a.Get("isFallbackAdapter").Truthy()
	return ai
}

func features(set js.Value) map[FeatureName]bool {
	out := map[FeatureName]bool{}
	if nullish(set) {
		return out
	}
	it := set.Call("values")
	for {
		n := it.Call("next")
		if n.Get("done").Truthy() {
			return out
		}
		out[FeatureName(n.Get("value").String())] = true
	}
}

func limits(l js.Value) map[string]uint64 {
	out := map[string]uint64{}
	if nullish(l) {
		return out
	}
	for _, name := range LimitNames {
		if v := l.Get(name); v.Type() == js.TypeNumber {
			out[name] = uint64(v.Float())
		}
	}
	return out
}

// Info returns the adapter's GPUAdapterInfo.
func (a *Adapter) Info() AdapterInfo { return a.info }

// HasFeature reports whether the adapter supports a feature.
func (a *Adapter) HasFeature(f FeatureName) bool { return a.features[f] }

// Limit returns one of the adapter's limits and whether it is known.
func (a *Adapter) Limit(name string) (uint64, bool) {
	v, ok := a.limits[name]
	return v, ok
}

// RequestDevice is adapter.requestDevice(). The descriptor's Quota is
// checked first: a feature or limit it does not allow never reaches the
// browser.
func (a *Adapter) RequestDevice(ctx context.Context, desc *DeviceDescriptor) (*Device, error) {
	var q *Quota
	if desc != nil {
		q = desc.Quota
	}
	if err := q.checkDevice(desc); err != nil {
		return nil, err
	}
	var p js.Value
	if err := try("requestDevice", func() { p = a.v.Call("requestDevice", js.ValueOf(desc.toJS())) }); err != nil {
		return nil, err
	}
	dv, err := await(ctx, "requestDevice", p)
	if err != nil {
		return nil, err
	}
	label := ""
	if desc != nil {
		label = desc.Label
	}
	d := newDevice(dv, label, q)
	d.limits = limits(dv.Get("limits"))
	d.queue = &Queue{object: object{v: dv.Get("queue")}, dev: d}
	go func() {
		v, err := await(context.Background(), "lost", dv.Get("lost"))
		info := DeviceLostInfo{Reason: "unknown"}
		if err == nil && !nullish(v) {
			info = DeviceLostInfo{Reason: str(v.Get("reason")), Message: str(v.Get("message"))}
		}
		d.setLost(info)
	}()
	return d, nil
}
