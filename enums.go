package webgpu

// Enumerations are strings, spelled as in the WebGPU IDL
// (https://www.w3.org/TR/webgpu/), so any value the specification adds can
// be used before this package names it. Constants are given for the
// common ones.

type (
	TextureFormat        string
	TextureDimension     string
	TextureViewDimension string
	TextureAspect        string
	AddressMode          string
	FilterMode           string
	MipmapFilterMode     string
	CompareFunction      string
	PrimitiveTopology    string
	IndexFormat          string
	FrontFace            string
	CullMode             string
	VertexFormat         string
	VertexStepMode       string
	LoadOp               string
	StoreOp              string
	BlendFactor          string
	BlendOperation       string
	StencilOperation     string
	BufferBindingType    string
	SamplerBindingType   string
	TextureSampleType    string
	StorageTextureAccess string
	FeatureName          string
	PowerPreference      string
	CanvasAlphaMode      string
	ErrorFilter          string
)

const (
	TextureFormatR8Unorm             TextureFormat = "r8unorm"
	TextureFormatR8Uint              TextureFormat = "r8uint"
	TextureFormatRG8Unorm            TextureFormat = "rg8unorm"
	TextureFormatR32Float            TextureFormat = "r32float"
	TextureFormatR32Uint             TextureFormat = "r32uint"
	TextureFormatRGBA8Unorm          TextureFormat = "rgba8unorm"
	TextureFormatRGBA8UnormSRGB      TextureFormat = "rgba8unorm-srgb"
	TextureFormatBGRA8Unorm          TextureFormat = "bgra8unorm"
	TextureFormatBGRA8UnormSRGB      TextureFormat = "bgra8unorm-srgb"
	TextureFormatRGBA16Float         TextureFormat = "rgba16float"
	TextureFormatRGBA32Float         TextureFormat = "rgba32float"
	TextureFormatStencil8            TextureFormat = "stencil8"
	TextureFormatDepth16Unorm        TextureFormat = "depth16unorm"
	TextureFormatDepth24Plus         TextureFormat = "depth24plus"
	TextureFormatDepth24PlusStencil8 TextureFormat = "depth24plus-stencil8"
	TextureFormatDepth32Float        TextureFormat = "depth32float"

	TextureDimension1D TextureDimension = "1d"
	TextureDimension2D TextureDimension = "2d"
	TextureDimension3D TextureDimension = "3d"

	TextureViewDimension1D        TextureViewDimension = "1d"
	TextureViewDimension2D        TextureViewDimension = "2d"
	TextureViewDimension2DArray   TextureViewDimension = "2d-array"
	TextureViewDimensionCube      TextureViewDimension = "cube"
	TextureViewDimensionCubeArray TextureViewDimension = "cube-array"
	TextureViewDimension3D        TextureViewDimension = "3d"

	TextureAspectAll         TextureAspect = "all"
	TextureAspectStencilOnly TextureAspect = "stencil-only"
	TextureAspectDepthOnly   TextureAspect = "depth-only"

	AddressModeClampToEdge  AddressMode = "clamp-to-edge"
	AddressModeRepeat       AddressMode = "repeat"
	AddressModeMirrorRepeat AddressMode = "mirror-repeat"

	FilterModeNearest FilterMode = "nearest"
	FilterModeLinear  FilterMode = "linear"

	MipmapFilterModeNearest MipmapFilterMode = "nearest"
	MipmapFilterModeLinear  MipmapFilterMode = "linear"

	CompareFunctionNever        CompareFunction = "never"
	CompareFunctionLess         CompareFunction = "less"
	CompareFunctionEqual        CompareFunction = "equal"
	CompareFunctionLessEqual    CompareFunction = "less-equal"
	CompareFunctionGreater      CompareFunction = "greater"
	CompareFunctionNotEqual     CompareFunction = "not-equal"
	CompareFunctionGreaterEqual CompareFunction = "greater-equal"
	CompareFunctionAlways       CompareFunction = "always"

	PrimitiveTopologyPointList     PrimitiveTopology = "point-list"
	PrimitiveTopologyLineList      PrimitiveTopology = "line-list"
	PrimitiveTopologyLineStrip     PrimitiveTopology = "line-strip"
	PrimitiveTopologyTriangleList  PrimitiveTopology = "triangle-list"
	PrimitiveTopologyTriangleStrip PrimitiveTopology = "triangle-strip"

	IndexFormatUint16 IndexFormat = "uint16"
	IndexFormatUint32 IndexFormat = "uint32"

	FrontFaceCCW FrontFace = "ccw"
	FrontFaceCW  FrontFace = "cw"

	CullModeNone  CullMode = "none"
	CullModeFront CullMode = "front"
	CullModeBack  CullMode = "back"

	VertexFormatUint8x4   VertexFormat = "uint8x4"
	VertexFormatUnorm8x4  VertexFormat = "unorm8x4"
	VertexFormatUint16x2  VertexFormat = "uint16x2"
	VertexFormatFloat16x2 VertexFormat = "float16x2"
	VertexFormatFloat16x4 VertexFormat = "float16x4"
	VertexFormatFloat32   VertexFormat = "float32"
	VertexFormatFloat32x2 VertexFormat = "float32x2"
	VertexFormatFloat32x3 VertexFormat = "float32x3"
	VertexFormatFloat32x4 VertexFormat = "float32x4"
	VertexFormatUint32    VertexFormat = "uint32"
	VertexFormatUint32x2  VertexFormat = "uint32x2"
	VertexFormatUint32x4  VertexFormat = "uint32x4"

	VertexStepModeVertex   VertexStepMode = "vertex"
	VertexStepModeInstance VertexStepMode = "instance"

	LoadOpLoad  LoadOp = "load"
	LoadOpClear LoadOp = "clear"

	StoreOpStore   StoreOp = "store"
	StoreOpDiscard StoreOp = "discard"

	BlendFactorZero             BlendFactor = "zero"
	BlendFactorOne              BlendFactor = "one"
	BlendFactorSrc              BlendFactor = "src"
	BlendFactorOneMinusSrc      BlendFactor = "one-minus-src"
	BlendFactorSrcAlpha         BlendFactor = "src-alpha"
	BlendFactorOneMinusSrcAlpha BlendFactor = "one-minus-src-alpha"
	BlendFactorDst              BlendFactor = "dst"
	BlendFactorOneMinusDst      BlendFactor = "one-minus-dst"
	BlendFactorDstAlpha         BlendFactor = "dst-alpha"
	BlendFactorOneMinusDstAlpha BlendFactor = "one-minus-dst-alpha"
	BlendFactorConstant         BlendFactor = "constant"

	BlendOperationAdd             BlendOperation = "add"
	BlendOperationSubtract        BlendOperation = "subtract"
	BlendOperationReverseSubtract BlendOperation = "reverse-subtract"
	BlendOperationMin             BlendOperation = "min"
	BlendOperationMax             BlendOperation = "max"

	StencilOperationKeep    StencilOperation = "keep"
	StencilOperationZero    StencilOperation = "zero"
	StencilOperationReplace StencilOperation = "replace"
	StencilOperationInvert  StencilOperation = "invert"

	BufferBindingTypeUniform         BufferBindingType = "uniform"
	BufferBindingTypeStorage         BufferBindingType = "storage"
	BufferBindingTypeReadOnlyStorage BufferBindingType = "read-only-storage"

	SamplerBindingTypeFiltering    SamplerBindingType = "filtering"
	SamplerBindingTypeNonFiltering SamplerBindingType = "non-filtering"
	SamplerBindingTypeComparison   SamplerBindingType = "comparison"

	TextureSampleTypeFloat         TextureSampleType    = "float"
	TextureSampleTypeUnfilterable  TextureSampleType    = "unfilterable-float"
	TextureSampleTypeDepth         TextureSampleType    = "depth"
	TextureSampleTypeSint          TextureSampleType    = "sint"
	TextureSampleTypeUint          TextureSampleType    = "uint"
	StorageTextureAccessWriteOnly  StorageTextureAccess = "write-only"
	StorageTextureAccessReadOnly   StorageTextureAccess = "read-only"
	StorageTextureAccessReadWrite  StorageTextureAccess = "read-write"
	FeatureTimestampQuery          FeatureName          = "timestamp-query"
	FeatureDepth32FloatStencil8    FeatureName          = "depth32float-stencil8"
	FeatureTextureCompressionBC    FeatureName          = "texture-compression-bc"
	FeatureIndirectFirstInstance   FeatureName          = "indirect-first-instance"
	FeatureShaderF16               FeatureName          = "shader-f16"
	FeatureFloat32Filterable       FeatureName          = "float32-filterable"
	PowerPreferenceLowPower        PowerPreference      = "low-power"
	PowerPreferenceHighPerformance PowerPreference      = "high-performance"
	CanvasAlphaModeOpaque          CanvasAlphaMode      = "opaque"
	CanvasAlphaModePremultiplied   CanvasAlphaMode      = "premultiplied"
	ErrorFilterValidation          ErrorFilter          = "validation"
	ErrorFilterOutOfMemory         ErrorFilter          = "out-of-memory"
	ErrorFilterInternal            ErrorFilter          = "internal"
)

// Flag sets, with the values of the IDL's GPUFlagsConstant namespaces.
type (
	BufferUsage  uint32
	TextureUsage uint32
	ShaderStage  uint32
	MapMode      uint32
	ColorWrite   uint32
)

const (
	BufferUsageMapRead      BufferUsage = 0x0001
	BufferUsageMapWrite     BufferUsage = 0x0002
	BufferUsageCopySrc      BufferUsage = 0x0004
	BufferUsageCopyDst      BufferUsage = 0x0008
	BufferUsageIndex        BufferUsage = 0x0010
	BufferUsageVertex       BufferUsage = 0x0020
	BufferUsageUniform      BufferUsage = 0x0040
	BufferUsageStorage      BufferUsage = 0x0080
	BufferUsageIndirect     BufferUsage = 0x0100
	BufferUsageQueryResolve BufferUsage = 0x0200

	TextureUsageCopySrc          TextureUsage = 0x01
	TextureUsageCopyDst          TextureUsage = 0x02
	TextureUsageTextureBinding   TextureUsage = 0x04
	TextureUsageStorageBinding   TextureUsage = 0x08
	TextureUsageRenderAttachment TextureUsage = 0x10

	ShaderStageVertex   ShaderStage = 0x1
	ShaderStageFragment ShaderStage = 0x2
	ShaderStageCompute  ShaderStage = 0x4

	MapModeRead  MapMode = 0x1
	MapModeWrite MapMode = 0x2

	ColorWriteRed   ColorWrite = 0x1
	ColorWriteGreen ColorWrite = 0x2
	ColorWriteBlue  ColorWrite = 0x4
	ColorWriteAlpha ColorWrite = 0x8
	ColorWriteAll   ColorWrite = 0xF
)
