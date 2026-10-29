package webgpu

import "fmt"

// Descriptors mirror the WebGPU IDL dictionaries. A zero field is left out
// of the dictionary, so the browser applies the specification's default;
// fields whose zero value is meaningful and differs from the default are
// pointers.

// Extent3D is a GPUExtent3DDict. DepthOrArrayLayers 0 means 1.
type Extent3D struct{ Width, Height, DepthOrArrayLayers uint32 }

func (e Extent3D) norm() Extent3D {
	if e.Height == 0 {
		e.Height = 1
	}
	if e.DepthOrArrayLayers == 0 {
		e.DepthOrArrayLayers = 1
	}
	return e
}

func (e Extent3D) toJS() map[string]any {
	e = e.norm()
	return map[string]any{"width": e.Width, "height": e.Height, "depthOrArrayLayers": e.DepthOrArrayLayers}
}

// Origin3D is a GPUOrigin3DDict.
type Origin3D struct{ X, Y, Z uint32 }

func (o Origin3D) toJS() map[string]any { return map[string]any{"x": o.X, "y": o.Y, "z": o.Z} }

// Color is a GPUColorDict.
type Color struct{ R, G, B, A float64 }

func (c Color) toJS() map[string]any { return map[string]any{"r": c.R, "g": c.G, "b": c.B, "a": c.A} }

func setStr[T ~string](d map[string]any, k string, v T) {
	if v != "" {
		d[k] = string(v)
	}
}

func setNum[T uint32 | uint64 | uint16 | int32 | float32 | float64](d map[string]any, k string, v T) {
	if v != 0 {
		d[k] = v
	}
}

func setBool(d map[string]any, k string, v bool) {
	if v {
		d[k] = true
	}
}

func constants(d map[string]any, c map[string]float64) {
	if len(c) == 0 {
		return
	}
	m := make(map[string]any, len(c))
	for k, v := range c {
		m[k] = v
	}
	d["constants"] = m
}

// RequestAdapterOptions is a GPURequestAdapterOptions. AllowFallback is
// ours: without it, a fallback (software) adapter is refused with
// ErrFallbackAdapter.
type RequestAdapterOptions struct {
	PowerPreference      PowerPreference
	ForceFallbackAdapter bool
	AllowFallback        bool
}

func (o *RequestAdapterOptions) toJS() map[string]any {
	d := map[string]any{}
	if o == nil {
		return d
	}
	setStr(d, "powerPreference", o.PowerPreference)
	setBool(d, "forceFallbackAdapter", o.ForceFallbackAdapter)
	return d
}

// DeviceDescriptor is a GPUDeviceDescriptor plus the Quota the device
// enforces.
type DeviceDescriptor struct {
	Label            string
	RequiredFeatures []FeatureName
	RequiredLimits   map[string]uint64
	Quota            *Quota
}

func (dd *DeviceDescriptor) toJS() map[string]any {
	d := map[string]any{}
	if dd == nil {
		return d
	}
	setStr(d, "label", dd.Label)
	if len(dd.RequiredFeatures) > 0 {
		fs := make([]any, len(dd.RequiredFeatures))
		for i, f := range dd.RequiredFeatures {
			fs[i] = string(f)
		}
		d["requiredFeatures"] = fs
	}
	if len(dd.RequiredLimits) > 0 {
		l := map[string]any{}
		for k, v := range dd.RequiredLimits {
			l[k] = v
		}
		d["requiredLimits"] = l
	}
	return d
}

// BufferDescriptor is a GPUBufferDescriptor.
type BufferDescriptor struct {
	Label            string
	Size             uint64
	Usage            BufferUsage
	MappedAtCreation bool
}

func (b BufferDescriptor) toJS() map[string]any {
	d := map[string]any{"size": b.Size, "usage": uint32(b.Usage)}
	setStr(d, "label", b.Label)
	setBool(d, "mappedAtCreation", b.MappedAtCreation)
	return d
}

// TextureDescriptor is a GPUTextureDescriptor. Zero MipLevelCount and
// SampleCount mean 1, zero Dimension "2d".
type TextureDescriptor struct {
	Label         string
	Size          Extent3D
	MipLevelCount uint32
	SampleCount   uint32
	Dimension     TextureDimension
	Format        TextureFormat
	Usage         TextureUsage
	ViewFormats   []TextureFormat
}

func (t TextureDescriptor) norm() TextureDescriptor {
	t.Size = t.Size.norm()
	if t.MipLevelCount == 0 {
		t.MipLevelCount = 1
	}
	if t.SampleCount == 0 {
		t.SampleCount = 1
	}
	if t.Dimension == "" {
		t.Dimension = TextureDimension2D
	}
	return t
}

func (t TextureDescriptor) toJS() map[string]any {
	t = t.norm()
	d := map[string]any{
		"size": t.Size.toJS(), "mipLevelCount": t.MipLevelCount, "sampleCount": t.SampleCount,
		"dimension": string(t.Dimension), "format": string(t.Format), "usage": uint32(t.Usage),
	}
	setStr(d, "label", t.Label)
	if len(t.ViewFormats) > 0 {
		vf := make([]any, len(t.ViewFormats))
		for i, f := range t.ViewFormats {
			vf[i] = string(f)
		}
		d["viewFormats"] = vf
	}
	return d
}

// TextureViewDescriptor is a GPUTextureViewDescriptor.
type TextureViewDescriptor struct {
	Label           string
	Format          TextureFormat
	Dimension       TextureViewDimension
	Aspect          TextureAspect
	BaseMipLevel    uint32
	MipLevelCount   uint32
	BaseArrayLayer  uint32
	ArrayLayerCount uint32
}

func (v *TextureViewDescriptor) toJS() map[string]any {
	d := map[string]any{}
	if v == nil {
		return d
	}
	setStr(d, "label", v.Label)
	setStr(d, "format", v.Format)
	setStr(d, "dimension", v.Dimension)
	setStr(d, "aspect", v.Aspect)
	setNum(d, "baseMipLevel", v.BaseMipLevel)
	setNum(d, "mipLevelCount", v.MipLevelCount)
	setNum(d, "baseArrayLayer", v.BaseArrayLayer)
	setNum(d, "arrayLayerCount", v.ArrayLayerCount)
	return d
}

// SamplerDescriptor is a GPUSamplerDescriptor. LodMaxClamp 0 means the
// default, 32.
type SamplerDescriptor struct {
	Label                                    string
	AddressModeU, AddressModeV, AddressModeW AddressMode
	MagFilter, MinFilter                     FilterMode
	MipmapFilter                             MipmapFilterMode
	LodMinClamp, LodMaxClamp                 float32
	Compare                                  CompareFunction
	MaxAnisotropy                            uint16
}

func (s *SamplerDescriptor) toJS() map[string]any {
	d := map[string]any{}
	if s == nil {
		return d
	}
	setStr(d, "label", s.Label)
	setStr(d, "addressModeU", s.AddressModeU)
	setStr(d, "addressModeV", s.AddressModeV)
	setStr(d, "addressModeW", s.AddressModeW)
	setStr(d, "magFilter", s.MagFilter)
	setStr(d, "minFilter", s.MinFilter)
	setStr(d, "mipmapFilter", s.MipmapFilter)
	setNum(d, "lodMinClamp", s.LodMinClamp)
	setNum(d, "lodMaxClamp", s.LodMaxClamp)
	setStr(d, "compare", s.Compare)
	setNum(d, "maxAnisotropy", s.MaxAnisotropy)
	return d
}

// ShaderModuleDescriptor is a GPUShaderModuleDescriptor (WGSL source).
type ShaderModuleDescriptor struct {
	Label string
	Code  string
}

func (s ShaderModuleDescriptor) toJS() map[string]any {
	d := map[string]any{"code": s.Code}
	setStr(d, "label", s.Label)
	return d
}

// BufferBindingLayout is a GPUBufferBindingLayout.
type BufferBindingLayout struct {
	Type             BufferBindingType
	HasDynamicOffset bool
	MinBindingSize   uint64
}

// SamplerBindingLayout is a GPUSamplerBindingLayout.
type SamplerBindingLayout struct{ Type SamplerBindingType }

// TextureBindingLayout is a GPUTextureBindingLayout.
type TextureBindingLayout struct {
	SampleType    TextureSampleType
	ViewDimension TextureViewDimension
	Multisampled  bool
}

// StorageTextureBindingLayout is a GPUStorageTextureBindingLayout.
type StorageTextureBindingLayout struct {
	Access        StorageTextureAccess
	Format        TextureFormat
	ViewDimension TextureViewDimension
}

// BindGroupLayoutEntry is a GPUBindGroupLayoutEntry: exactly one of
// Buffer, Sampler, Texture, StorageTexture is set.
type BindGroupLayoutEntry struct {
	Binding        uint32
	Visibility     ShaderStage
	Buffer         *BufferBindingLayout
	Sampler        *SamplerBindingLayout
	Texture        *TextureBindingLayout
	StorageTexture *StorageTextureBindingLayout
}

func (e BindGroupLayoutEntry) toJS() (map[string]any, error) {
	d := map[string]any{"binding": e.Binding, "visibility": uint32(e.Visibility)}
	n := 0
	if b := e.Buffer; b != nil {
		n++
		m := map[string]any{}
		setStr(m, "type", b.Type)
		setBool(m, "hasDynamicOffset", b.HasDynamicOffset)
		setNum(m, "minBindingSize", b.MinBindingSize)
		d["buffer"] = m
	}
	if s := e.Sampler; s != nil {
		n++
		m := map[string]any{}
		setStr(m, "type", s.Type)
		d["sampler"] = m
	}
	if t := e.Texture; t != nil {
		n++
		m := map[string]any{}
		setStr(m, "sampleType", t.SampleType)
		setStr(m, "viewDimension", t.ViewDimension)
		setBool(m, "multisampled", t.Multisampled)
		d["texture"] = m
	}
	if s := e.StorageTexture; s != nil {
		n++
		m := map[string]any{"format": string(s.Format)}
		setStr(m, "access", s.Access)
		setStr(m, "viewDimension", s.ViewDimension)
		d["storageTexture"] = m
	}
	if n != 1 {
		return nil, fmt.Errorf("webgpu: bind group layout entry %d: %d binding types set, want exactly 1", e.Binding, n)
	}
	return d, nil
}

// BindGroupLayoutDescriptor is a GPUBindGroupLayoutDescriptor.
type BindGroupLayoutDescriptor struct {
	Label   string
	Entries []BindGroupLayoutEntry
}

func (l BindGroupLayoutDescriptor) toJS() (map[string]any, error) {
	es := make([]any, len(l.Entries))
	for i, e := range l.Entries {
		m, err := e.toJS()
		if err != nil {
			return nil, err
		}
		es[i] = m
	}
	d := map[string]any{"entries": es}
	setStr(d, "label", l.Label)
	return d, nil
}

// PipelineLayoutDescriptor is a GPUPipelineLayoutDescriptor.
type PipelineLayoutDescriptor struct {
	Label            string
	BindGroupLayouts []*BindGroupLayout
}

func (p PipelineLayoutDescriptor) toJS() (map[string]any, error) {
	ls := make([]any, len(p.BindGroupLayouts))
	for i, l := range p.BindGroupLayouts {
		if l == nil {
			return nil, fmt.Errorf("webgpu: pipeline layout: bind group layout %d is nil", i)
		}
		ls[i] = l.ref()
	}
	d := map[string]any{"bindGroupLayouts": ls}
	setStr(d, "label", p.Label)
	return d, nil
}

// BindGroupEntry is a GPUBindGroupEntry: exactly one of Buffer, Sampler,
// TextureView is set. Size 0 binds the rest of the buffer from Offset.
type BindGroupEntry struct {
	Binding     uint32
	Buffer      *Buffer
	Offset      uint64
	Size        uint64
	Sampler     *Sampler
	TextureView *TextureView
}

func (e BindGroupEntry) toJS() (map[string]any, error) {
	d := map[string]any{"binding": e.Binding}
	n := 0
	if b := e.Buffer; b != nil {
		n++
		size := e.Size
		if size == 0 {
			if e.Offset > b.size {
				return nil, bounds("createBindGroup", "binding %d: offset %d past the buffer's %d bytes", e.Binding, e.Offset, b.size)
			}
		} else if err := checkRange("createBindGroup", b, e.Offset, size); err != nil {
			return nil, err
		}
		r := map[string]any{"buffer": b.ref()}
		setNum(r, "offset", e.Offset)
		setNum(r, "size", size)
		d["resource"] = r
	}
	if e.Sampler != nil {
		n++
		d["resource"] = e.Sampler.ref()
	}
	if e.TextureView != nil {
		n++
		d["resource"] = e.TextureView.ref()
	}
	if n != 1 {
		return nil, fmt.Errorf("webgpu: bind group entry %d: %d resources set, want exactly 1", e.Binding, n)
	}
	return d, nil
}

// BindGroupDescriptor is a GPUBindGroupDescriptor.
type BindGroupDescriptor struct {
	Label   string
	Layout  *BindGroupLayout
	Entries []BindGroupEntry
}

func (g BindGroupDescriptor) toJS() (map[string]any, error) {
	if g.Layout == nil {
		return nil, fmt.Errorf("webgpu: bind group: nil layout")
	}
	es := make([]any, len(g.Entries))
	for i, e := range g.Entries {
		m, err := e.toJS()
		if err != nil {
			return nil, err
		}
		es[i] = m
	}
	d := map[string]any{"layout": g.Layout.ref(), "entries": es}
	setStr(d, "label", g.Label)
	return d, nil
}

// VertexAttribute is a GPUVertexAttribute.
type VertexAttribute struct {
	Format         VertexFormat
	Offset         uint64
	ShaderLocation uint32
}

// VertexBufferLayout is a GPUVertexBufferLayout.
type VertexBufferLayout struct {
	ArrayStride uint64
	StepMode    VertexStepMode
	Attributes  []VertexAttribute
}

// VertexState is a GPUVertexState.
type VertexState struct {
	Module     *ShaderModule
	EntryPoint string
	Constants  map[string]float64
	Buffers    []VertexBufferLayout
}

func stage(op string, mod *ShaderModule, entry string, c map[string]float64) (map[string]any, error) {
	if mod == nil {
		return nil, fmt.Errorf("webgpu: %s: nil shader module", op)
	}
	d := map[string]any{"module": mod.ref()}
	setStr(d, "entryPoint", entry)
	constants(d, c)
	return d, nil
}

func (v VertexState) toJS() (map[string]any, error) {
	d, err := stage("vertex state", v.Module, v.EntryPoint, v.Constants)
	if err != nil {
		return nil, err
	}
	if len(v.Buffers) > 0 {
		bs := make([]any, len(v.Buffers))
		for i, b := range v.Buffers {
			as := make([]any, len(b.Attributes))
			for j, a := range b.Attributes {
				as[j] = map[string]any{"format": string(a.Format), "offset": a.Offset, "shaderLocation": a.ShaderLocation}
			}
			m := map[string]any{"arrayStride": b.ArrayStride, "attributes": as}
			setStr(m, "stepMode", b.StepMode)
			bs[i] = m
		}
		d["buffers"] = bs
	}
	return d, nil
}

// PrimitiveState is a GPUPrimitiveState.
type PrimitiveState struct {
	Topology         PrimitiveTopology
	StripIndexFormat IndexFormat
	FrontFace        FrontFace
	CullMode         CullMode
	UnclippedDepth   bool
}

func (p PrimitiveState) toJS() map[string]any {
	d := map[string]any{}
	setStr(d, "topology", p.Topology)
	setStr(d, "stripIndexFormat", p.StripIndexFormat)
	setStr(d, "frontFace", p.FrontFace)
	setStr(d, "cullMode", p.CullMode)
	setBool(d, "unclippedDepth", p.UnclippedDepth)
	return d
}

// StencilFaceState is a GPUStencilFaceState.
type StencilFaceState struct {
	Compare                     CompareFunction
	FailOp, DepthFailOp, PassOp StencilOperation
}

func (s StencilFaceState) toJS() map[string]any {
	d := map[string]any{}
	setStr(d, "compare", s.Compare)
	setStr(d, "failOp", s.FailOp)
	setStr(d, "depthFailOp", s.DepthFailOp)
	setStr(d, "passOp", s.PassOp)
	return d
}

// DepthStencilState is a GPUDepthStencilState. DepthWriteEnabled is
// always sent; nil masks mean the default 0xFFFFFFFF.
type DepthStencilState struct {
	Format              TextureFormat
	DepthWriteEnabled   bool
	DepthCompare        CompareFunction
	StencilFront        StencilFaceState
	StencilBack         StencilFaceState
	StencilReadMask     *uint32
	StencilWriteMask    *uint32
	DepthBias           int32
	DepthBiasSlopeScale float32
	DepthBiasClamp      float32
}

func (s DepthStencilState) toJS() map[string]any {
	d := map[string]any{"format": string(s.Format), "depthWriteEnabled": s.DepthWriteEnabled,
		"stencilFront": s.StencilFront.toJS(), "stencilBack": s.StencilBack.toJS()}
	setStr(d, "depthCompare", s.DepthCompare)
	if s.StencilReadMask != nil {
		d["stencilReadMask"] = *s.StencilReadMask
	}
	if s.StencilWriteMask != nil {
		d["stencilWriteMask"] = *s.StencilWriteMask
	}
	setNum(d, "depthBias", s.DepthBias)
	setNum(d, "depthBiasSlopeScale", s.DepthBiasSlopeScale)
	setNum(d, "depthBiasClamp", s.DepthBiasClamp)
	return d
}

// MultisampleState is a GPUMultisampleState. Zero Count means 1, zero
// Mask all samples.
type MultisampleState struct {
	Count                  uint32
	Mask                   uint32
	AlphaToCoverageEnabled bool
}

func (m MultisampleState) toJS() map[string]any {
	d := map[string]any{}
	setNum(d, "count", m.Count)
	setNum(d, "mask", m.Mask)
	setBool(d, "alphaToCoverageEnabled", m.AlphaToCoverageEnabled)
	return d
}

// BlendComponent is a GPUBlendComponent.
type BlendComponent struct {
	Operation            BlendOperation
	SrcFactor, DstFactor BlendFactor
}

func (b BlendComponent) toJS() map[string]any {
	d := map[string]any{}
	setStr(d, "operation", b.Operation)
	setStr(d, "srcFactor", b.SrcFactor)
	setStr(d, "dstFactor", b.DstFactor)
	return d
}

// BlendState is a GPUBlendState.
type BlendState struct{ Color, Alpha BlendComponent }

// ColorTargetState is a GPUColorTargetState; a nil WriteMask writes all
// channels.
type ColorTargetState struct {
	Format    TextureFormat
	Blend     *BlendState
	WriteMask *ColorWrite
}

// FragmentState is a GPUFragmentState.
type FragmentState struct {
	Module     *ShaderModule
	EntryPoint string
	Constants  map[string]float64
	Targets    []ColorTargetState
}

func (f FragmentState) toJS() (map[string]any, error) {
	d, err := stage("fragment state", f.Module, f.EntryPoint, f.Constants)
	if err != nil {
		return nil, err
	}
	ts := make([]any, len(f.Targets))
	for i, t := range f.Targets {
		m := map[string]any{"format": string(t.Format)}
		if t.Blend != nil {
			m["blend"] = map[string]any{"color": t.Blend.Color.toJS(), "alpha": t.Blend.Alpha.toJS()}
		}
		if t.WriteMask != nil {
			m["writeMask"] = uint32(*t.WriteMask)
		}
		ts[i] = m
	}
	d["targets"] = ts
	return d, nil
}

// RenderPipelineDescriptor is a GPURenderPipelineDescriptor. A nil Layout
// is "auto".
type RenderPipelineDescriptor struct {
	Label        string
	Layout       *PipelineLayout
	Vertex       VertexState
	Primitive    PrimitiveState
	DepthStencil *DepthStencilState
	Multisample  MultisampleState
	Fragment     *FragmentState
}

func layout(l *PipelineLayout) any {
	if l == nil {
		return "auto"
	}
	return l.ref()
}

func (r RenderPipelineDescriptor) toJS() (map[string]any, error) {
	v, err := r.Vertex.toJS()
	if err != nil {
		return nil, err
	}
	d := map[string]any{"layout": layout(r.Layout), "vertex": v, "primitive": r.Primitive.toJS(), "multisample": r.Multisample.toJS()}
	setStr(d, "label", r.Label)
	if r.DepthStencil != nil {
		d["depthStencil"] = r.DepthStencil.toJS()
	}
	if r.Fragment != nil {
		f, err := r.Fragment.toJS()
		if err != nil {
			return nil, err
		}
		d["fragment"] = f
	}
	return d, nil
}

// ProgrammableStage is a GPUProgrammableStage.
type ProgrammableStage struct {
	Module     *ShaderModule
	EntryPoint string
	Constants  map[string]float64
}

// ComputePipelineDescriptor is a GPUComputePipelineDescriptor. A nil
// Layout is "auto".
type ComputePipelineDescriptor struct {
	Label   string
	Layout  *PipelineLayout
	Compute ProgrammableStage
}

func (c ComputePipelineDescriptor) toJS() (map[string]any, error) {
	s, err := stage("compute stage", c.Compute.Module, c.Compute.EntryPoint, c.Compute.Constants)
	if err != nil {
		return nil, err
	}
	d := map[string]any{"layout": layout(c.Layout), "compute": s}
	setStr(d, "label", c.Label)
	return d, nil
}

// RenderPassColorAttachment is a GPURenderPassColorAttachment.
type RenderPassColorAttachment struct {
	View          *TextureView
	DepthSlice    *uint32
	ResolveTarget *TextureView
	ClearValue    Color
	LoadOp        LoadOp
	StoreOp       StoreOp
}

// RenderPassDepthStencilAttachment is a GPURenderPassDepthStencilAttachment.
// Load and store ops left empty are omitted (required by the browser for
// the aspects the format has, unless read-only).
type RenderPassDepthStencilAttachment struct {
	View              *TextureView
	DepthClearValue   float32
	DepthLoadOp       LoadOp
	DepthStoreOp      StoreOp
	DepthReadOnly     bool
	StencilClearValue uint32
	StencilLoadOp     LoadOp
	StencilStoreOp    StoreOp
	StencilReadOnly   bool
}

// RenderPassDescriptor is a GPURenderPassDescriptor. A nil entry in
// ColorAttachments is an unused slot.
type RenderPassDescriptor struct {
	Label                  string
	ColorAttachments       []*RenderPassColorAttachment
	DepthStencilAttachment *RenderPassDepthStencilAttachment
	MaxDrawCount           uint64
}

func (r RenderPassDescriptor) toJS() (map[string]any, error) {
	cs := make([]any, len(r.ColorAttachments))
	for i, c := range r.ColorAttachments {
		if c == nil {
			cs[i] = nil
			continue
		}
		if c.View == nil {
			return nil, fmt.Errorf("webgpu: render pass: color attachment %d has no view", i)
		}
		m := map[string]any{"view": c.View.ref(), "clearValue": c.ClearValue.toJS(), "loadOp": string(c.LoadOp), "storeOp": string(c.StoreOp)}
		if c.DepthSlice != nil {
			m["depthSlice"] = *c.DepthSlice
		}
		if c.ResolveTarget != nil {
			m["resolveTarget"] = c.ResolveTarget.ref()
		}
		cs[i] = m
	}
	d := map[string]any{"colorAttachments": cs}
	setStr(d, "label", r.Label)
	if ds := r.DepthStencilAttachment; ds != nil {
		if ds.View == nil {
			return nil, fmt.Errorf("webgpu: render pass: depth-stencil attachment has no view")
		}
		m := map[string]any{"view": ds.View.ref()}
		if ds.DepthLoadOp == LoadOpClear {
			m["depthClearValue"] = ds.DepthClearValue
		}
		setStr(m, "depthLoadOp", ds.DepthLoadOp)
		setStr(m, "depthStoreOp", ds.DepthStoreOp)
		setBool(m, "depthReadOnly", ds.DepthReadOnly)
		setNum(m, "stencilClearValue", ds.StencilClearValue)
		setStr(m, "stencilLoadOp", ds.StencilLoadOp)
		setStr(m, "stencilStoreOp", ds.StencilStoreOp)
		setBool(m, "stencilReadOnly", ds.StencilReadOnly)
		d["depthStencilAttachment"] = m
	}
	setNum(d, "maxDrawCount", r.MaxDrawCount)
	return d, nil
}

// RenderBundleEncoderDescriptor is a GPURenderBundleEncoderDescriptor.
type RenderBundleEncoderDescriptor struct {
	Label              string
	ColorFormats       []TextureFormat
	DepthStencilFormat TextureFormat
	SampleCount        uint32
	DepthReadOnly      bool
	StencilReadOnly    bool
}

func (r RenderBundleEncoderDescriptor) toJS() map[string]any {
	cf := make([]any, len(r.ColorFormats))
	for i, f := range r.ColorFormats {
		if f == "" {
			cf[i] = nil
		} else {
			cf[i] = string(f)
		}
	}
	d := map[string]any{"colorFormats": cf}
	setStr(d, "label", r.Label)
	setStr(d, "depthStencilFormat", r.DepthStencilFormat)
	setNum(d, "sampleCount", r.SampleCount)
	setBool(d, "depthReadOnly", r.DepthReadOnly)
	setBool(d, "stencilReadOnly", r.StencilReadOnly)
	return d
}

// TexelCopyBufferLayout is a GPUTexelCopyBufferLayout. Zero BytesPerRow
// and RowsPerImage are "not specified".
type TexelCopyBufferLayout struct {
	Offset       uint64
	BytesPerRow  uint32
	RowsPerImage uint32
}

func (l TexelCopyBufferLayout) toJS() map[string]any {
	d := map[string]any{}
	setNum(d, "offset", l.Offset)
	setNum(d, "bytesPerRow", l.BytesPerRow)
	setNum(d, "rowsPerImage", l.RowsPerImage)
	return d
}

// TexelCopyTextureInfo is a GPUTexelCopyTextureInfo.
type TexelCopyTextureInfo struct {
	Texture  *Texture
	MipLevel uint32
	Origin   Origin3D
	Aspect   TextureAspect
}

func (t TexelCopyTextureInfo) toJS() map[string]any {
	d := map[string]any{"texture": t.Texture.ref(), "mipLevel": t.MipLevel, "origin": t.Origin.toJS()}
	setStr(d, "aspect", t.Aspect)
	return d
}

// TexelCopyBufferInfo is a GPUTexelCopyBufferInfo.
type TexelCopyBufferInfo struct {
	Buffer *Buffer
	Layout TexelCopyBufferLayout
}

func (b TexelCopyBufferInfo) toJS() map[string]any {
	d := b.Layout.toJS()
	d["buffer"] = b.Buffer.ref()
	return d
}
