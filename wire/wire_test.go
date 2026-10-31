package wire

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

// stream is one of every object-carrying command, ids 1..3, with a
// depth attachment and a resolve target; the Recorder's own tests (in
// package webgpu) check that what it writes decodes to what it recorded.
func stream() []uint32 {
	return []uint32{
		uint32(WriteBuffer), 1, 0, 0, 0, 4,
		uint32(CopyBufferToBuffer), 1, 0, 0, 2, 0, 0, 4, 0,
		uint32(CopyTextureToTexture), 2, 0, 1, 2, 0, 1, 3, 0, 0, 0, 0, 3, 4, 4, 1,
		uint32(BeginRenderPass), 2,
		3, 2, 2, 1, 0, 0, 0, 0, // color 0: view 3, resolve 2
		0, 0, 0, 0, 0, 0, 0, 0, // color 1 unused
		1, 1, 2, 1, 0, 0, 1, 1, 0, 0, // depth: view 1
		uint32(SetPipeline), 2,
		uint32(SetBindGroup), 0, 3, 2, 256, 512,
		uint32(SetBindGroup), 1, 0, 0, // unbind
		uint32(SetVertexBuffer), 0, 1, 0, 0, 64, 0,
		uint32(SetIndexBuffer), 1, 2, 0, 0, 64, 0,
		uint32(Draw), 3, 1, 0, 0,
		uint32(DrawIndirect), 1, 0, 0,
		uint32(ExecuteBundles), 2, 2, 3,
		uint32(EndPass),
		uint32(BeginComputePass),
		uint32(SetPipeline), 3,
		uint32(DispatchIndirect), 1, 16, 0,
		uint32(EndPass),
		uint32(Submit),
	}
}

func known(id uint32) bool { return id >= 1 && id <= 3 }

func TestDecode(t *testing.T) {
	w := stream()
	cmds, err := Decode(w, 4, known)
	if err != nil {
		t.Fatal(err)
	}
	var ops []Op
	n := 0
	for _, c := range cmds {
		ops = append(ops, c.Op)
		n += 1 + len(c.Args)
	}
	want := []Op{WriteBuffer, CopyBufferToBuffer, CopyTextureToTexture, BeginRenderPass, SetPipeline, SetBindGroup, SetBindGroup,
		SetVertexBuffer, SetIndexBuffer, Draw, DrawIndirect, ExecuteBundles, EndPass, BeginComputePass,
		SetPipeline, DispatchIndirect, EndPass, Submit}
	if !reflect.DeepEqual(ops, want) || n != len(w) {
		t.Fatalf("ops %v (%d of %d words)", ops, n, len(w))
	}
	if cmds, err := Decode(nil, 0, known); err != nil || cmds != nil {
		t.Fatalf("empty: %v %v", cmds, err)
	}
}

func TestDecodeRejects(t *testing.T) {
	rp := []uint32{20, 0, 0} // empty render pass
	for _, c := range []struct {
		name  string
		w     []uint32
		ndata uint32
		err   string
	}{
		{"bad opcode", []uint32{99}, 0, "bad opcode"},
		{"draw outside pass", []uint32{3, 1, 1, 0, 0}, 0, "out of place"},
		{"pipeline outside pass", []uint32{1, 1}, 0, "out of place"},
		{"dispatch in render", append(append([]uint32{}, rp...), 23, 1, 1, 1), 0, "out of place"},
		{"write in pass", append(append([]uint32{}, rp...), 30, 1, 0, 0, 0, 4), 4, "out of place"},
		{"truncated draw", append(append([]uint32{}, rp...), 3, 1), 0, "truncated"},
		{"truncated bind group", append(append([]uint32{}, rp...), 2, 0), 0, "truncated"},
		{"too many offsets", append(append([]uint32{}, rp...), 2, 0, 1, 70000), 0, "dynamic offsets"},
		{"offsets past end", append(append([]uint32{}, rp...), 2, 0, 1, 5, 1), 0, "truncated"},
		{"truncated bundles", append(append([]uint32{}, rp...), 11), 0, "truncated"},
		{"too many bundles", append(append([]uint32{}, rp...), 11, 70000), 0, "bundles"},
		{"truncated pass", []uint32{20}, 0, "truncated"},
		{"9 colors", []uint32{20, 9}, 0, "color attachments"},
		{"truncated colors", []uint32{20, 1, 1}, 0, "truncated"},
		{"bad depth flag", []uint32{20, 0, 2}, 0, "depth flag"},
		{"truncated depth", []uint32{20, 0, 1, 1}, 0, "truncated"},
		{"unknown view", []uint32{20, 1, 9, 0, 1, 1, 0, 0, 0, 0, 0}, 0, "unknown object 9"},
		{"unknown resolve", []uint32{20, 1, 1, 9, 1, 1, 0, 0, 0, 0, 0}, 0, "unknown object 9"},
		{"dirty unused slot", []uint32{20, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0}, 0, "not empty"},
		{"bad load op", []uint32{20, 1, 1, 0, 3, 1, 0, 0, 0, 0, 0}, 0, "load or store"},
		{"unknown depth view", []uint32{20, 0, 1, 9, 1, 1, 0, 0, 1, 1, 0, 0}, 0, "unknown object 9"},
		{"bad depth op", []uint32{20, 0, 1, 1, 3, 1, 0, 0, 1, 1, 0, 0}, 0, "depth-stencil ops"},
		{"bad read-only", []uint32{20, 0, 1, 1, 1, 1, 0, 2, 1, 1, 0, 0}, 0, "depth-stencil ops"},
		{"pipeline id 0", append(append([]uint32{}, rp...), 1, 0), 0, "unknown object 0"},
		{"bind group unknown", append(append([]uint32{}, rp...), 2, 0, 9, 0), 0, "unknown object 9"},
		{"vertex unknown", append(append([]uint32{}, rp...), 5, 0, 9, 0, 0, 0, 0), 0, "unknown object 9"},
		{"index unknown", append(append([]uint32{}, rp...), 6, 9, 1, 0, 0, 0, 0), 0, "unknown object 9"},
		{"index format", append(append([]uint32{}, rp...), 6, 1, 3, 0, 0, 0, 0), 0, "index format"},
		{"bundle unknown", append(append([]uint32{}, rp...), 11, 1, 9), 0, "unknown object 9"},
		{"indirect unknown", append(append([]uint32{}, rp...), 12, 9, 0, 0), 0, "unknown object 9"},
		{"write unknown", []uint32{30, 9, 0, 0, 0, 4}, 4, "unknown object 9"},
		{"write past data", []uint32{30, 1, 0, 0, 2, 4}, 4, "past 4 bytes"},
		{"write overflow", []uint32{30, 1, 0, 0, math.MaxUint32, 2}, 4, "past"},
		{"copy unknown src", []uint32{31, 9, 0, 0, 1, 0, 0, 4, 0}, 0, "unknown object 9"},
		{"copy unknown dst", []uint32{31, 1, 0, 0, 9, 0, 0, 4, 0}, 0, "unknown object 9"},
		{"pass left open", []uint32{22}, 0, "left open"},
		{"texture copy unknown src", []uint32{32, 9, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1}, 0, "unknown object 9"},
		{"texture copy unknown dst", []uint32{32, 1, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 1, 1, 1}, 0, "unknown object 9"},
		{"texture copy src aspect", []uint32{32, 1, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 1, 1, 1}, 0, "aspect"},
		{"texture copy dst aspect", []uint32{32, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 1, 1, 1}, 0, "aspect"},
		{"texture copy truncated", []uint32{32, 1, 0, 0}, 0, "truncated"},
		{"texture copy in pass", append(append([]uint32{}, rp...), 32, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1), 0, "out of place"},
	} {
		_, err := Decode(c.w, c.ndata, known)
		if err == nil || !strings.Contains(err.Error(), c.err) {
			t.Errorf("%s: %v, want %q", c.name, err, c.err)
		}
		// Remap reads with the same checks; it fails on the same streams.
		w := append([]uint32(nil), c.w...)
		if err := Remap(w, c.ndata, func(id uint32) (uint32, bool) { return id + 100, known(id) }); err == nil {
			t.Errorf("%s: Remap accepted it", c.name)
		}
	}
}

func TestRemap(t *testing.T) {
	w := stream()
	before := append([]uint32(nil), w...)
	seen := map[uint32]int{}
	err := Remap(w, 4, func(id uint32) (uint32, bool) {
		seen[id]++
		return id + 100, known(id)
	})
	if err != nil {
		t.Fatal(err)
	}
	// Every id moved, nothing else did: compare against Decode's view of
	// both streams, command by command.
	newKnown := func(id uint32) bool { return id >= 101 && id <= 103 }
	a, _ := Decode(before, 4, known)
	b, err := Decode(w, 4, newKnown)
	if err != nil {
		t.Fatalf("remapped stream does not decode against the new table: %v", err)
	}
	changed := 0
	for i := range w {
		if w[i] != before[i] {
			if w[i] != before[i]+100 {
				t.Fatalf("word %d: %d -> %d", i, before[i], w[i])
			}
			changed++
		}
	}
	// ids: write 1, copy 2, texture copy 2, colour 2, depth 1, pipeline 1,
	// group 1 (the unbind's 0 stays), vertex 1, index 1, indirect 1,
	// bundles 2, pipeline 1, dispatch 1 = 17.
	if changed != 17 || len(a) != len(b) {
		t.Fatalf("%d ids changed, %d/%d commands", changed, len(a), len(b))
	}
	if seen[0] != 0 {
		t.Fatal("f was called for a nullable 0")
	}
}

func TestRemapRefuses(t *testing.T) {
	grantNone := func(uint32) (uint32, bool) { return 0, false }
	if err := Remap(stream(), 4, grantNone); err == nil {
		t.Error("an ungranted id passed")
	}
	toZero := func(uint32) (uint32, bool) { return 0, true }
	if err := Remap(stream(), 4, toZero); err == nil {
		t.Error("a mapping to id 0 passed")
	}
	// A required id of 0 is refused without asking f.
	called := false
	err := Remap([]uint32{20, 0, 0, uint32(SetPipeline), 0, 21}, 0, func(id uint32) (uint32, bool) {
		called = true
		return 1, true
	})
	if err == nil || called {
		t.Errorf("pipeline 0: err %v, f called %v", err, called)
	}
}
