package webgpu

import (
	"reflect"
	"strings"
	"testing"
)

type m = map[string]any

func eq(t *testing.T, got, want any) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got  %#v\nwant %#v", got, want)
	}
}

func errHas(t *testing.T, err error, sub string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), sub) {
		t.Fatalf("got %v, want an error containing %q", err, sub)
	}
}

func TestSimpleDescriptors(t *testing.T) {
	eq(t, (*RequestAdapterOptions)(nil).toJS(), m{})
	eq(t, (&RequestAdapterOptions{PowerPreference: PowerPreferenceHighPerformance, ForceFallbackAdapter: true}).toJS(),
		m{"powerPreference": "high-performance", "forceFallbackAdapter": true})

	eq(t, (*DeviceDescriptor)(nil).toJS(), m{})
	eq(t, (&DeviceDescriptor{Label: "d", RequiredFeatures: []FeatureName{FeatureShaderF16}, RequiredLimits: map[string]uint64{"maxBindGroups": 4}}).toJS(),
		m{"label": "d", "requiredFeatures": []any{"shader-f16"}, "requiredLimits": m{"maxBindGroups": uint64(4)}})

	eq(t, BufferDescriptor{Label: "b", Size: 64, Usage: BufferUsageVertex | BufferUsageCopyDst, MappedAtCreation: true}.toJS(),
		m{"label": "b", "size": uint64(64), "usage": uint32(0x28), "mappedAtCreation": true})

	eq(t, TextureDescriptor{Label: "t", Size: Extent3D{Width: 8}, Format: "r8uint", Usage: TextureUsageCopyDst, ViewFormats: []TextureFormat{"r8uint"}}.toJS(),
		m{"label": "t", "size": m{"width": uint32(8), "height": uint32(1), "depthOrArrayLayers": uint32(1)},
			"mipLevelCount": uint32(1), "sampleCount": uint32(1), "dimension": "2d", "format": "r8uint", "usage": uint32(2),
			"viewFormats": []any{"r8uint"}})

	eq(t, (*TextureViewDescriptor)(nil).toJS(), m{})
	eq(t, (&TextureViewDescriptor{Label: "v", Format: "r8uint", Dimension: TextureViewDimension2DArray, Aspect: TextureAspectAll,
		BaseMipLevel: 1, MipLevelCount: 2, BaseArrayLayer: 3, ArrayLayerCount: 4}).toJS(),
		m{"label": "v", "format": "r8uint", "dimension": "2d-array", "aspect": "all", "baseMipLevel": uint32(1), "mipLevelCount": uint32(2),
			"baseArrayLayer": uint32(3), "arrayLayerCount": uint32(4)})

	eq(t, (*SamplerDescriptor)(nil).toJS(), m{})
	eq(t, (&SamplerDescriptor{Label: "s", AddressModeU: AddressModeRepeat, AddressModeV: AddressModeRepeat, AddressModeW: AddressModeClampToEdge,
		MagFilter: FilterModeLinear, MinFilter: FilterModeNearest, MipmapFilter: MipmapFilterModeLinear, LodMinClamp: 1, LodMaxClamp: 8,
		Compare: CompareFunctionLess, MaxAnisotropy: 4}).toJS(),
		m{"label": "s", "addressModeU": "repeat", "addressModeV": "repeat", "addressModeW": "clamp-to-edge", "magFilter": "linear",
			"minFilter": "nearest", "mipmapFilter": "linear", "lodMinClamp": float32(1), "lodMaxClamp": float32(8), "compare": "less",
			"maxAnisotropy": uint16(4)})

	eq(t, ShaderModuleDescriptor{Label: "w", Code: "x"}.toJS(), m{"label": "w", "code": "x"})
	eq(t, Origin3D{1, 2, 3}.toJS(), m{"x": uint32(1), "y": uint32(2), "z": uint32(3)})
	eq(t, Color{1, 0, 0, 1}.toJS(), m{"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0})
}

func TestBindGroupLayoutDescriptor(t *testing.T) {
	d := BindGroupLayoutDescriptor{Label: "l", Entries: []BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageVertex, Buffer: &BufferBindingLayout{Type: BufferBindingTypeUniform, HasDynamicOffset: true, MinBindingSize: 64}},
		{Binding: 1, Visibility: ShaderStageFragment, Sampler: &SamplerBindingLayout{Type: SamplerBindingTypeFiltering}},
		{Binding: 2, Visibility: ShaderStageFragment, Texture: &TextureBindingLayout{SampleType: TextureSampleTypeUint, ViewDimension: TextureViewDimension2D, Multisampled: true}},
		{Binding: 3, Visibility: ShaderStageCompute, StorageTexture: &StorageTextureBindingLayout{Access: StorageTextureAccessWriteOnly, Format: "rgba8unorm", ViewDimension: TextureViewDimension2D}},
	}}
	got, err := d.toJS()
	if err != nil {
		t.Fatal(err)
	}
	eq(t, got, m{"label": "l", "entries": []any{
		m{"binding": uint32(0), "visibility": uint32(1), "buffer": m{"type": "uniform", "hasDynamicOffset": true, "minBindingSize": uint64(64)}},
		m{"binding": uint32(1), "visibility": uint32(2), "sampler": m{"type": "filtering"}},
		m{"binding": uint32(2), "visibility": uint32(2), "texture": m{"sampleType": "uint", "viewDimension": "2d", "multisampled": true}},
		m{"binding": uint32(3), "visibility": uint32(4), "storageTexture": m{"access": "write-only", "format": "rgba8unorm", "viewDimension": "2d"}},
	}})
	_, err = BindGroupLayoutDescriptor{Entries: []BindGroupLayoutEntry{{Binding: 7}}}.toJS()
	errHas(t, err, "entry 7: 0 binding types")
}

func TestPipelineLayoutAndBindGroup(t *testing.T) {
	l := &BindGroupLayout{}
	got, err := PipelineLayoutDescriptor{Label: "p", BindGroupLayouts: []*BindGroupLayout{l}}.toJS()
	if err != nil {
		t.Fatal(err)
	}
	eq(t, got, m{"label": "p", "bindGroupLayouts": []any{l.ref()}})
	_, err = PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{nil}}.toJS()
	errHas(t, err, "layout 0 is nil")

	b := buf(256, BufferUsageUniform)
	s := &Sampler{}
	v := &TextureView{}
	got, err = BindGroupDescriptor{Label: "g", Layout: l, Entries: []BindGroupEntry{
		{Binding: 0, Buffer: b, Offset: 64, Size: 64},
		{Binding: 1, Buffer: b},
		{Binding: 2, Sampler: s},
		{Binding: 3, TextureView: v},
	}}.toJS()
	if err != nil {
		t.Fatal(err)
	}
	eq(t, got, m{"label": "g", "layout": l.ref(), "entries": []any{
		m{"binding": uint32(0), "resource": m{"buffer": b.ref(), "offset": uint64(64), "size": uint64(64)}},
		m{"binding": uint32(1), "resource": m{"buffer": b.ref()}},
		m{"binding": uint32(2), "resource": s.ref()},
		m{"binding": uint32(3), "resource": v.ref()},
	}})
	_, err = BindGroupDescriptor{Entries: nil}.toJS()
	errHas(t, err, "nil layout")
	_, err = BindGroupDescriptor{Layout: l, Entries: []BindGroupEntry{{Binding: 1, Sampler: s, TextureView: v}}}.toJS()
	errHas(t, err, "2 resources")
	_, err = BindGroupDescriptor{Layout: l, Entries: []BindGroupEntry{{Buffer: b, Offset: 250, Size: 16}}}.toJS()
	errHas(t, err, "outside")
	_, err = BindGroupDescriptor{Layout: l, Entries: []BindGroupEntry{{Buffer: b, Offset: 257}}}.toJS()
	errHas(t, err, "past the buffer")
}

func TestRenderPipelineDescriptor(t *testing.T) {
	mod := &ShaderModule{}
	mask := ColorWriteRed | ColorWriteAlpha
	rm, wm := uint32(0xff), uint32(0x0f)
	pl := &PipelineLayout{}
	d := RenderPipelineDescriptor{
		Label:  "r",
		Layout: pl,
		Vertex: VertexState{Module: mod, EntryPoint: "vs", Constants: map[string]float64{"k": 2},
			Buffers: []VertexBufferLayout{{ArrayStride: 16, StepMode: VertexStepModeInstance,
				Attributes: []VertexAttribute{{Format: VertexFormatFloat32x3, Offset: 0, ShaderLocation: 0}}}}},
		Primitive: PrimitiveState{Topology: PrimitiveTopologyTriangleStrip, StripIndexFormat: IndexFormatUint16, FrontFace: FrontFaceCW, CullMode: CullModeBack, UnclippedDepth: true},
		DepthStencil: &DepthStencilState{Format: TextureFormatDepth24PlusStencil8, DepthWriteEnabled: true, DepthCompare: CompareFunctionLessEqual,
			StencilFront:    StencilFaceState{Compare: CompareFunctionAlways, FailOp: StencilOperationKeep, DepthFailOp: StencilOperationZero, PassOp: StencilOperationReplace},
			StencilReadMask: &rm, StencilWriteMask: &wm, DepthBias: 2, DepthBiasSlopeScale: 1.5, DepthBiasClamp: 0.5},
		Multisample: MultisampleState{Count: 4, Mask: 0xf, AlphaToCoverageEnabled: true},
		Fragment: &FragmentState{Module: mod, EntryPoint: "fs", Targets: []ColorTargetState{
			{Format: "bgra8unorm", Blend: &BlendState{
				Color: BlendComponent{Operation: BlendOperationAdd, SrcFactor: BlendFactorSrcAlpha, DstFactor: BlendFactorOneMinusSrcAlpha},
				Alpha: BlendComponent{Operation: BlendOperationAdd, SrcFactor: BlendFactorOne, DstFactor: BlendFactorZero}}, WriteMask: &mask},
			{Format: "r8uint"},
		}},
	}
	got, err := d.toJS()
	if err != nil {
		t.Fatal(err)
	}
	eq(t, got, m{
		"label":  "r",
		"layout": pl.ref(),
		"vertex": m{"module": mod.ref(), "entryPoint": "vs", "constants": m{"k": 2.0}, "buffers": []any{
			m{"arrayStride": uint64(16), "stepMode": "instance", "attributes": []any{m{"format": "float32x3", "offset": uint64(0), "shaderLocation": uint32(0)}}}}},
		"primitive": m{"topology": "triangle-strip", "stripIndexFormat": "uint16", "frontFace": "cw", "cullMode": "back", "unclippedDepth": true},
		"depthStencil": m{"format": "depth24plus-stencil8", "depthWriteEnabled": true, "depthCompare": "less-equal",
			"stencilFront": m{"compare": "always", "failOp": "keep", "depthFailOp": "zero", "passOp": "replace"}, "stencilBack": m{},
			"stencilReadMask": uint32(0xff), "stencilWriteMask": uint32(0x0f), "depthBias": int32(2), "depthBiasSlopeScale": float32(1.5), "depthBiasClamp": float32(0.5)},
		"multisample": m{"count": uint32(4), "mask": uint32(0xf), "alphaToCoverageEnabled": true},
		"fragment": m{"module": mod.ref(), "entryPoint": "fs", "targets": []any{
			m{"format": "bgra8unorm", "blend": m{
				"color": m{"operation": "add", "srcFactor": "src-alpha", "dstFactor": "one-minus-src-alpha"},
				"alpha": m{"operation": "add", "srcFactor": "one", "dstFactor": "zero"}}, "writeMask": uint32(9)},
			m{"format": "r8uint"},
		}},
	})

	// Defaults: auto layout, no fragment, no depth.
	got, err = RenderPipelineDescriptor{Vertex: VertexState{Module: mod}}.toJS()
	if err != nil {
		t.Fatal(err)
	}
	eq(t, got, m{"layout": "auto", "vertex": m{"module": mod.ref()}, "primitive": m{}, "multisample": m{}})

	_, err = RenderPipelineDescriptor{}.toJS()
	errHas(t, err, "vertex state: nil shader module")
	_, err = RenderPipelineDescriptor{Vertex: VertexState{Module: mod}, Fragment: &FragmentState{}}.toJS()
	errHas(t, err, "fragment state: nil shader module")
}

func TestComputePipelineDescriptor(t *testing.T) {
	mod := &ShaderModule{}
	got, err := ComputePipelineDescriptor{Label: "c", Compute: ProgrammableStage{Module: mod, EntryPoint: "main"}}.toJS()
	if err != nil {
		t.Fatal(err)
	}
	eq(t, got, m{"label": "c", "layout": "auto", "compute": m{"module": mod.ref(), "entryPoint": "main"}})
	_, err = ComputePipelineDescriptor{}.toJS()
	errHas(t, err, "compute stage")
}

func TestRenderPassDescriptor(t *testing.T) {
	v, rv, dv := &TextureView{}, &TextureView{}, &TextureView{}
	slice := uint32(2)
	d := RenderPassDescriptor{
		Label: "p",
		ColorAttachments: []*RenderPassColorAttachment{
			{View: v, DepthSlice: &slice, ResolveTarget: rv, ClearValue: Color{0, 0, 0, 1}, LoadOp: LoadOpClear, StoreOp: StoreOpStore},
			nil,
		},
		DepthStencilAttachment: &RenderPassDepthStencilAttachment{View: dv, DepthClearValue: 1, DepthLoadOp: LoadOpClear, DepthStoreOp: StoreOpDiscard,
			StencilClearValue: 3, StencilLoadOp: LoadOpLoad, StencilStoreOp: StoreOpStore},
		MaxDrawCount: 1000,
	}
	got, err := d.toJS()
	if err != nil {
		t.Fatal(err)
	}
	eq(t, got, m{"label": "p", "colorAttachments": []any{
		m{"view": v.ref(), "depthSlice": uint32(2), "resolveTarget": rv.ref(), "clearValue": m{"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0}, "loadOp": "clear", "storeOp": "store"},
		nil,
	}, "depthStencilAttachment": m{"view": dv.ref(), "depthClearValue": float32(1), "depthLoadOp": "clear", "depthStoreOp": "discard",
		"stencilClearValue": uint32(3), "stencilLoadOp": "load", "stencilStoreOp": "store"},
		"maxDrawCount": uint64(1000)})

	got, err = RenderPassDescriptor{DepthStencilAttachment: &RenderPassDepthStencilAttachment{View: dv, DepthReadOnly: true, StencilReadOnly: true}}.toJS()
	if err != nil {
		t.Fatal(err)
	}
	eq(t, got, m{"colorAttachments": []any{}, "depthStencilAttachment": m{"view": dv.ref(), "depthReadOnly": true, "stencilReadOnly": true}})

	_, err = RenderPassDescriptor{ColorAttachments: []*RenderPassColorAttachment{{}}}.toJS()
	errHas(t, err, "color attachment 0 has no view")
	_, err = RenderPassDescriptor{DepthStencilAttachment: &RenderPassDepthStencilAttachment{}}.toJS()
	errHas(t, err, "depth-stencil attachment has no view")
}

func TestBundleAndCopyDescriptors(t *testing.T) {
	eq(t, RenderBundleEncoderDescriptor{Label: "rb", ColorFormats: []TextureFormat{"bgra8unorm", ""}, DepthStencilFormat: "depth24plus",
		SampleCount: 4, DepthReadOnly: true, StencilReadOnly: true}.toJS(),
		m{"label": "rb", "colorFormats": []any{"bgra8unorm", nil}, "depthStencilFormat": "depth24plus", "sampleCount": uint32(4),
			"depthReadOnly": true, "stencilReadOnly": true})

	b := buf(4, BufferUsageCopySrc)
	tx := tex(TextureDescriptor{Format: "r8uint"})
	eq(t, TexelCopyBufferInfo{Buffer: b, Layout: TexelCopyBufferLayout{Offset: 4, BytesPerRow: 256, RowsPerImage: 2}}.toJS(),
		m{"buffer": b.ref(), "offset": uint64(4), "bytesPerRow": uint32(256), "rowsPerImage": uint32(2)})
	eq(t, TexelCopyTextureInfo{Texture: tx, MipLevel: 1, Origin: Origin3D{X: 1}, Aspect: TextureAspectAll}.toJS(),
		m{"texture": tx.ref(), "mipLevel": uint32(1), "origin": m{"x": uint32(1), "y": uint32(0), "z": uint32(0)}, "aspect": "all"})
}

func TestLimitNames(t *testing.T) {
	seen := map[string]bool{}
	for _, n := range LimitNames {
		if seen[n] {
			t.Fatalf("%s twice", n)
		}
		seen[n] = true
	}
	if !seen["maxBufferSize"] || !seen["maxComputeWorkgroupsPerDimension"] {
		t.Fatal("limits missing")
	}
}
