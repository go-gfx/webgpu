package webgpu

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/go-gfx/webgpu/wire"
)

// table maps replay ids to objects, as a receiver would.
type table map[uint32]any

func (tb table) add(o any, obj *object) { tb[idOf(obj)] = o }

func (tb table) resolve(id uint32) (any, error) {
	if o, ok := tb[id]; ok {
		return o, nil
	}
	return nil, errors.New("no such object")
}

type objects struct {
	buf  *Buffer
	tex  *Texture
	view *TextureView
	smp  *Sampler
	mod  *ShaderModule
	bgl  *BindGroupLayout
	pl   *PipelineLayout
	tb   table
}

func newObjects() objects {
	o := objects{buf: &Buffer{size: 64}, tex: &Texture{}, view: &TextureView{}, smp: &Sampler{},
		mod: &ShaderModule{}, bgl: &BindGroupLayout{}, pl: &PipelineLayout{}, tb: table{}}
	o.tb.add(o.buf, &o.buf.object)
	o.tb.add(o.tex, &o.tex.object)
	o.tb.add(o.view, &o.view.object)
	o.tb.add(o.smp, &o.smp.object)
	o.tb.add(o.mod, &o.mod.object)
	o.tb.add(o.bgl, &o.bgl.object)
	o.tb.add(o.pl, &o.pl.object)
	return o
}

func descriptors(o objects) []any {
	wm := ColorWriteAll
	return []any{
		BufferDescriptor{Label: "b", Size: 256, Usage: BufferUsageStorage | BufferUsageCopySrc},
		TextureDescriptor{Size: Extent3D{64, 32, 1}, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageCopyDst, ViewFormats: []TextureFormat{TextureFormatRGBA8Unorm}},
		TextureViewCreation{Texture: o.tex},
		TextureViewCreation{Texture: o.tex, Descriptor: &TextureViewDescriptor{BaseMipLevel: 1}},
		SamplerDescriptor{MagFilter: FilterModeLinear, LodMaxClamp: 8},
		ShaderModuleDescriptor{Code: "@compute @workgroup_size(1) fn main() {}"},
		BindGroupLayoutDescriptor{Entries: []BindGroupLayoutEntry{{Binding: 0, Visibility: ShaderStageCompute, Buffer: &BufferBindingLayout{Type: BufferBindingTypeStorage}}}},
		PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{o.bgl, o.bgl}},
		BindGroupDescriptor{Layout: o.bgl, Entries: []BindGroupEntry{
			{Binding: 0, Buffer: o.buf, Offset: 16, Size: 32}, {Binding: 1, Sampler: o.smp}, {Binding: 2, TextureView: o.view}}},
		RenderPipelineDescriptor{Layout: o.pl, Vertex: VertexState{Module: o.mod, EntryPoint: "vs", Constants: map[string]float64{"k": 2}},
			Fragment: &FragmentState{Module: o.mod, EntryPoint: "fs", Targets: []ColorTargetState{{Format: TextureFormatBGRA8Unorm, WriteMask: &wm}}}},
		RenderPipelineDescriptor{Vertex: VertexState{Module: o.mod}}, // auto layout, no fragment
		ComputePipelineDescriptor{Layout: o.pl, Compute: ProgrammableStage{Module: o.mod, EntryPoint: "main"}},
		ComputePipelineDescriptor{Compute: ProgrammableStage{Module: o.mod}},
	}
}

func TestDescriptorRoundTrip(t *testing.T) {
	o := newObjects()
	for _, d := range descriptors(o) {
		kind, b, err := EncodeDescriptor(d)
		if err != nil {
			t.Fatalf("%T: %v", d, err)
		}
		got, err := DecodeDescriptor(kind, b, o.tb.resolve)
		if err != nil {
			t.Fatalf("%T: decode: %v", d, err)
		}
		if !reflect.DeepEqual(got, d) {
			t.Errorf("%T round trip\n got %+v\nwant %+v", d, got, d)
		}
	}
}

func TestEncodeDescriptorRefuses(t *testing.T) {
	o := newObjects()
	for name, d := range map[string]any{
		"view of nil":     TextureViewCreation{},
		"nil bgl":         PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{nil}},
		"bg nil layout":   BindGroupDescriptor{},
		"bg empty entry":  BindGroupDescriptor{Layout: o.bgl, Entries: []BindGroupEntry{{}}},
		"bg two":          BindGroupDescriptor{Layout: o.bgl, Entries: []BindGroupEntry{{Buffer: o.buf, Sampler: o.smp}}},
		"rp nil vertex":   RenderPipelineDescriptor{},
		"rp nil fragment": RenderPipelineDescriptor{Vertex: VertexState{Module: o.mod}, Fragment: &FragmentState{}},
		"cp nil module":   ComputePipelineDescriptor{},
		"not a desc":      42,
		"NaN constant":    ComputePipelineDescriptor{Compute: ProgrammableStage{Module: o.mod, Constants: map[string]float64{"x": math.NaN()}}},
	} {
		if _, _, err := EncodeDescriptor(d); !errors.Is(err, ErrDescriptor) {
			t.Errorf("%s: %v", name, err)
		}
	}
}

func TestDecodeDescriptorRefuses(t *testing.T) {
	o := newObjects()
	enc := func(d any) []byte {
		_, b, err := EncodeDescriptor(d)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	raw := func(refs []uint32, body string) []byte { return wire.AppendDescriptor(nil, refs, []byte(body)) }
	bg := BindGroupDescriptor{Layout: o.bgl, Entries: []BindGroupEntry{{Binding: 0, Buffer: o.buf}}}
	mod, bgl, pl, buf := idOf(&o.mod.object), idOf(&o.bgl.object), idOf(&o.pl.object), idOf(&o.buf.object)
	cases := []struct {
		name string
		kind wire.Kind
		b    []byte
	}{
		{"split", wire.KindBuffer, []byte{1}},
		{"kind", 99, raw(nil, `{}`)},
		{"unknown field", wire.KindBuffer, raw(nil, `{"Nope":1}`)},
		{"trailing", wire.KindBuffer, raw(nil, `{} {}`)},
		{"not json", wire.KindTexture, raw(nil, `[`)},
		{"extra ref", wire.KindSampler, raw([]uint32{1}, `{}`)},
		{"missing ref", wire.KindTextureView, raw(nil, `null`)},
		{"required 0", wire.KindTextureView, raw([]uint32{0}, `null`)},
		{"unresolved", wire.KindTextureView, raw([]uint32{12345}, `null`)},
		{"wrong type", wire.KindTextureView, raw([]uint32{mod}, `null`)},
		{"pl short refs", wire.KindPipelineLayout, raw([]uint32{bgl}, `{"BindGroupLayouts":[null,null]}`)},
		{"bg entry type", wire.KindBindGroup, raw([]uint32{bgl, mod}, `{"Entries":[{"Binding":0}]}`)},
		{"bg entry missing", wire.KindBindGroup, raw([]uint32{bgl}, `{"Entries":[{"Binding":0}]}`)},
		{"bg bad layout", wire.KindBindGroup, raw([]uint32{buf, buf}, `{"Entries":[{"Binding":0}]}`)},
		{"rp fragment without module", wire.KindRenderPipeline, raw([]uint32{0, mod, 0}, `{"Fragment":{}}`)},
		{"rp module without fragment", wire.KindRenderPipeline, raw([]uint32{0, mod, mod}, `{}`)},
		{"rp bad layout", wire.KindRenderPipeline, raw([]uint32{mod, mod, 0}, `{}`)},
		{"rp bad vertex", wire.KindRenderPipeline, raw([]uint32{pl, pl, 0}, `{}`)},
		{"rp bad fragment", wire.KindRenderPipeline, raw([]uint32{pl, mod, pl}, `{"Fragment":{}}`)},
		{"cp bad layout", wire.KindComputePipeline, raw([]uint32{mod, mod}, `{}`)},
		{"cp no module", wire.KindComputePipeline, raw([]uint32{pl}, `{}`)},
		{"bg body", wire.KindBindGroup, raw([]uint32{bgl}, `{"Entries":7}`)},
		{"pl body", wire.KindPipelineLayout, raw(nil, `{"BindGroupLayouts":7}`)},
		{"rp body", wire.KindRenderPipeline, raw(nil, `7`)},
		{"cp body", wire.KindComputePipeline, raw(nil, `7`)},
	}
	for _, c := range cases {
		if _, err := DecodeDescriptor(c.kind, c.b, o.tb.resolve); !errors.Is(err, ErrDescriptor) {
			t.Errorf("%s: %v", c.name, err)
		}
	}
	// A body that smuggles objects in its object fields gets them
	// replaced by the references, never kept.
	b := enc(bg)
	refs, _, _ := wire.SplitDescriptor(b)
	smuggled := raw(refs, `{"Layout":{},"Entries":[{"Binding":0,"Sampler":{},"TextureView":{}}]}`)
	got, err := DecodeDescriptor(wire.KindBindGroup, smuggled, o.tb.resolve)
	if err != nil {
		t.Fatal(err)
	}
	d := got.(BindGroupDescriptor)
	if d.Layout != o.bgl || d.Entries[0].Buffer != o.buf || d.Entries[0].Sampler != nil || d.Entries[0].TextureView != nil {
		t.Errorf("smuggled objects kept: %+v", d)
	}
	if !strings.Contains(string(b), `"Binding":0`) {
		t.Errorf("body %q", b)
	}
}
