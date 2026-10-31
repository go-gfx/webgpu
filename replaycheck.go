package webgpu

import (
	"fmt"

	"github.com/go-gfx/webgpu/wire"
)

// replayable returns the object of a handle a stream may name.
func replayable(h any) (*object, error) {
	switch o := h.(type) {
	case *Buffer:
		return &o.object, nil
	case *Texture:
		return &o.object, nil
	case *TextureView:
		return &o.object, nil
	case *BindGroup:
		return &o.object, nil
	case *RenderPipeline:
		return &o.object, nil
	case *ComputePipeline:
		return &o.object, nil
	case *RenderBundle:
		return &o.object, nil
	}
	return nil, fmt.Errorf("webgpu: Replay: a %T cannot be named by a stream", h)
}

// ReplayID returns the id that streams and wire descriptors name h by
// (allocated on first use): the id a transport translates a foreign
// stream's or descriptor's ids to. h is any object handle of this
// package.
func ReplayID(h any) (uint32, error) {
	var o *object
	switch v := h.(type) {
	case *Sampler:
		o = &v.object
	case *ShaderModule:
		o = &v.object
	case *BindGroupLayout:
		o = &v.object
	case *PipelineLayout:
		o = &v.object
	default:
		var err error
		if o, err = replayable(h); err != nil {
			return 0, fmt.Errorf("webgpu: ReplayID: a %T is not an object handle", h)
		}
	}
	return idOf(o), nil
}

// checkReplay checks a foreign stream (Device.Replay): the references
// are objects a stream may name, the stream decodes with replay.js's
// checks, and every object it names is one of the references.
func checkReplay(words []uint32, data []byte, refs []any) ([]*object, error) {
	if uint64(len(data)) > maxData {
		return nil, bounds("Replay", "%d data bytes", len(data))
	}
	objs := make([]*object, len(refs))
	granted := make(map[uint32]bool, len(refs))
	for i, h := range refs {
		o, err := replayable(h)
		if err != nil {
			return nil, err
		}
		objs[i] = o
		granted[idOf(o)] = true
	}
	if _, err := wire.Decode(words, uint32(len(data)), func(id uint32) bool { return granted[id] }); err != nil {
		return nil, fmt.Errorf("webgpu: Replay: %w", err)
	}
	return objs, nil
}
