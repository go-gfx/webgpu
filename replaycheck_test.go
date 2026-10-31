package webgpu

import (
	"strings"
	"testing"

	"github.com/go-gfx/webgpu/wire"
)

func TestCheckReplay(t *testing.T) {
	src, dst := &Buffer{size: 16}, &Buffer{size: 16}
	refs := []any{src, dst, &TextureView{}, &BindGroup{}, &RenderPipeline{}, &ComputePipeline{}, &RenderBundle{}}
	s, d := idOf(&src.object), idOf(&dst.object)
	words := []uint32{uint32(wire.WriteBuffer), s, 0, 0, 0, 4, uint32(wire.CopyBufferToBuffer), s, 0, 0, d, 0, 0, 4, 0, uint32(wire.Submit)}
	objs, err := checkReplay(words, make([]byte, 4), refs)
	if err != nil || len(objs) != len(refs) || objs[0] != &src.object {
		t.Fatalf("check: %v", err)
	}
	// dst not granted: the stream may not name it.
	if _, err := checkReplay(words, make([]byte, 4), []any{src}); err == nil || !strings.Contains(err.Error(), "unknown object") {
		t.Errorf("ungranted object: %v", err)
	}
	for _, h := range []any{&Texture{}, &Sampler{}, &Device{}, 7} {
		if _, err := checkReplay(nil, nil, []any{h}); err == nil {
			t.Errorf("%T accepted as a stream object", h)
		}
	}
	old := maxData
	maxData = 3
	_, err = checkReplay(words, make([]byte, 4), refs)
	maxData = old
	if err == nil {
		t.Error("data past the limit")
	}
}
