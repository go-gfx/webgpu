package webgpu

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/go-gfx/webgpu/wire"
)

// Creation descriptors on the wire (package wire's layout: references,
// then the body). They let an object be created somewhere other than
// where its descriptor was written, e.g. by a virtio-gpu host for a
// guest: EncodeDescriptor names every referenced object by its replay id
// (the id the Recorder writes into streams), DecodeDescriptor rebuilds
// the descriptor with the receiver's objects.
//
// The body is the descriptor as JSON with its object fields left empty.
// References come in this order:
//
//	TextureView     the texture
//	PipelineLayout  each bind group layout
//	BindGroup       the layout, then one per entry (its buffer, sampler or view)
//	RenderPipeline  the layout (0: "auto"), the vertex module, the fragment module (0: none)
//	ComputePipeline the layout (0: "auto"), the module
//
// and none for the other kinds.

// TextureViewCreation is the descriptor of a texture view: the texture
// and Texture.CreateView's descriptor (nil: the default view).
type TextureViewCreation struct {
	Texture    *Texture
	Descriptor *TextureViewDescriptor
}

// ErrDescriptor is wrapped by every EncodeDescriptor and DecodeDescriptor
// error.
var ErrDescriptor = errors.New("webgpu: descriptor")

func descErr(format string, a ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrDescriptor}, a...)...)
}

func idOf(o *object) uint32 {
	id, _ := o.batchID()
	return id
}

// EncodeDescriptor returns the kind and wire bytes of a creation
// descriptor: BufferDescriptor, TextureDescriptor, TextureViewCreation,
// SamplerDescriptor, ShaderModuleDescriptor, BindGroupLayoutDescriptor,
// PipelineLayoutDescriptor, BindGroupDescriptor, RenderPipelineDescriptor
// or ComputePipelineDescriptor. Referenced objects must not be nil
// except where the descriptor allows it (a pipeline's layout, a render
// pipeline's fragment state).
func EncodeDescriptor(desc any) (wire.Kind, []byte, error) {
	var refs []uint32
	var kind wire.Kind
	var body any
	switch d := desc.(type) {
	case BufferDescriptor:
		kind, body = wire.KindBuffer, d
	case TextureDescriptor:
		kind, body = wire.KindTexture, d
	case SamplerDescriptor:
		kind, body = wire.KindSampler, d
	case ShaderModuleDescriptor:
		kind, body = wire.KindShaderModule, d
	case BindGroupLayoutDescriptor:
		kind, body = wire.KindBindGroupLayout, d
	case TextureViewCreation:
		if d.Texture == nil {
			return 0, nil, descErr("texture view of a nil texture")
		}
		kind, body, refs = wire.KindTextureView, d.Descriptor, []uint32{idOf(&d.Texture.object)}
	case PipelineLayoutDescriptor:
		kind = wire.KindPipelineLayout
		for i, l := range d.BindGroupLayouts {
			if l == nil {
				return 0, nil, descErr("pipeline layout: bind group layout %d is nil", i)
			}
			refs = append(refs, idOf(&l.object))
		}
		if d.BindGroupLayouts != nil { // nil and empty both survive the trip
			d.BindGroupLayouts = make([]*BindGroupLayout, len(d.BindGroupLayouts))
		}
		body = d
	case BindGroupDescriptor:
		kind = wire.KindBindGroup
		if d.Layout == nil {
			return 0, nil, descErr("bind group: nil layout")
		}
		refs = append(refs, idOf(&d.Layout.object))
		var es []BindGroupEntry
		if d.Entries != nil {
			es = make([]BindGroupEntry, len(d.Entries))
		}
		for i, e := range d.Entries {
			var o *object
			n := 0
			if e.Buffer != nil {
				o, n = &e.Buffer.object, n+1
			}
			if e.Sampler != nil {
				o, n = &e.Sampler.object, n+1
			}
			if e.TextureView != nil {
				o, n = &e.TextureView.object, n+1
			}
			if n != 1 {
				return 0, nil, descErr("bind group entry %d: %d resources set, want exactly 1", e.Binding, n)
			}
			refs = append(refs, idOf(o))
			e.Buffer, e.Sampler, e.TextureView = nil, nil, nil
			es[i] = e
		}
		d.Layout, d.Entries = nil, es
		body = d
	case RenderPipelineDescriptor:
		kind = wire.KindRenderPipeline
		if d.Vertex.Module == nil || d.Fragment != nil && d.Fragment.Module == nil {
			return 0, nil, descErr("render pipeline: nil shader module")
		}
		var layout, frag uint32
		if d.Layout != nil {
			layout = idOf(&d.Layout.object)
		}
		if d.Fragment != nil {
			frag = idOf(&d.Fragment.Module.object)
			f := *d.Fragment
			f.Module = nil
			d.Fragment = &f
		}
		refs = []uint32{layout, idOf(&d.Vertex.Module.object), frag}
		d.Layout, d.Vertex.Module = nil, nil
		body = d
	case ComputePipelineDescriptor:
		kind = wire.KindComputePipeline
		if d.Compute.Module == nil {
			return 0, nil, descErr("compute pipeline: nil shader module")
		}
		var layout uint32
		if d.Layout != nil {
			layout = idOf(&d.Layout.object)
		}
		refs = []uint32{layout, idOf(&d.Compute.Module.object)}
		d.Layout, d.Compute.Module = nil, nil
		body = d
	default:
		return 0, nil, descErr("%T is not a creation descriptor", desc)
	}
	js, err := json.Marshal(body)
	if err != nil {
		return 0, nil, descErr("%v", err)
	}
	return kind, wire.AppendDescriptor(nil, refs, js), nil
}

// Resolver returns the receiver's object for a reference id (a *Buffer,
// *Texture, *TextureView, *Sampler, *ShaderModule, *BindGroupLayout or
// *PipelineLayout). DecodeDescriptor checks the object's type.
type Resolver func(id uint32) (any, error)

// DecodeDescriptor rebuilds a creation descriptor of kind from its wire
// bytes, resolving references with resolve. It returns the same types
// EncodeDescriptor takes. The bytes come from another party: the body
// must name no unknown field, the reference count must match the body,
// every reference must resolve to an object of the right type, and the
// object fields of the body are discarded, never trusted.
func DecodeDescriptor(kind wire.Kind, b []byte, resolve Resolver) (any, error) {
	refs, body, err := wire.SplitDescriptor(b)
	if err != nil {
		return nil, descErr("%v", err)
	}
	r := refReader{refs: refs, resolve: resolve}
	var out any
	switch kind {
	case wire.KindBuffer:
		var d BufferDescriptor
		err = unmarshal(body, &d)
		out = d
	case wire.KindTexture:
		var d TextureDescriptor
		err = unmarshal(body, &d)
		out = d
	case wire.KindSampler:
		var d SamplerDescriptor
		err = unmarshal(body, &d)
		out = d
	case wire.KindShaderModule:
		var d ShaderModuleDescriptor
		err = unmarshal(body, &d)
		out = d
	case wire.KindBindGroupLayout:
		var d BindGroupLayoutDescriptor
		err = unmarshal(body, &d)
		out = d
	case wire.KindTextureView:
		var d *TextureViewDescriptor
		if err = unmarshal(body, &d); err == nil {
			var t *Texture
			t, err = ref[*Texture](&r, false)
			out = TextureViewCreation{Texture: t, Descriptor: d}
		}
	case wire.KindPipelineLayout:
		var d PipelineLayoutDescriptor
		if err = unmarshal(body, &d); err == nil {
			for i := range d.BindGroupLayouts {
				if d.BindGroupLayouts[i], err = ref[*BindGroupLayout](&r, false); err != nil {
					break
				}
			}
		}
		out = d
	case wire.KindBindGroup:
		var d BindGroupDescriptor
		if err = unmarshal(body, &d); err == nil {
			d.Layout, err = ref[*BindGroupLayout](&r, false)
			for i := range d.Entries {
				if err != nil {
					break
				}
				e := &d.Entries[i]
				e.Buffer, e.Sampler, e.TextureView = nil, nil, nil
				var o any
				if o, err = r.next(); err == nil {
					switch o := o.(type) {
					case *Buffer:
						e.Buffer = o
					case *Sampler:
						e.Sampler = o
					case *TextureView:
						e.TextureView = o
					default:
						err = descErr("bind group entry %d: a %T is not bindable", e.Binding, o)
					}
				}
			}
		}
		out = d
	case wire.KindRenderPipeline:
		var d RenderPipelineDescriptor
		if err = unmarshal(body, &d); err == nil {
			if d.Layout, err = ref[*PipelineLayout](&r, true); err == nil {
				d.Vertex.Module, err = ref[*ShaderModule](&r, false)
			}
			if err == nil {
				var m *ShaderModule
				if m, err = ref[*ShaderModule](&r, true); err == nil && (m == nil) != (d.Fragment == nil) {
					err = descErr("render pipeline: fragment module and fragment state disagree")
				} else if err == nil && m != nil {
					d.Fragment.Module = m
				}
			}
		}
		out = d
	case wire.KindComputePipeline:
		var d ComputePipelineDescriptor
		if err = unmarshal(body, &d); err == nil {
			if d.Layout, err = ref[*PipelineLayout](&r, true); err == nil {
				d.Compute.Module, err = ref[*ShaderModule](&r, false)
			}
		}
		out = d
	default:
		return nil, descErr("kind %d", kind)
	}
	if err == nil && r.i != len(refs) {
		err = descErr("%d references for %d objects", len(refs), r.i)
	}
	if err != nil {
		return nil, err
	}
	return out, nil
}

func unmarshal(body []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return descErr("%v", err)
	}
	if dec.More() {
		return descErr("data after the descriptor")
	}
	return nil
}

type refReader struct {
	refs    []uint32
	i       int
	resolve Resolver
}

// next resolves the next reference; a 0 reference is (nil, nil).
func (r *refReader) next() (any, error) {
	if r.i >= len(r.refs) {
		return nil, descErr("missing reference %d", r.i)
	}
	id := r.refs[r.i]
	r.i++
	if id == 0 {
		return nil, nil
	}
	o, err := r.resolve(id)
	if err != nil {
		return nil, descErr("reference %d: %v", id, err)
	}
	return o, nil
}

// ref resolves the next reference as a T; optional allows 0 (nil).
func ref[T any](r *refReader, optional bool) (T, error) {
	var zero T
	o, err := r.next()
	if err != nil {
		return zero, err
	}
	if o == nil {
		if optional {
			return zero, nil
		}
		return zero, descErr("reference %d is required", r.i-1)
	}
	t, ok := o.(T)
	if !ok {
		return zero, descErr("reference %d: a %T, want %T", r.i-1, o, zero)
	}
	return t, nil
}
