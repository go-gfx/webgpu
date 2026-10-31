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
