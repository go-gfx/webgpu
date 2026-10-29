package webgpu

// LimitNames are the attributes of GPUSupportedLimits
// (https://www.w3.org/TR/webgpu/#gpusupportedlimits). The interface's
// attributes live on its prototype and cannot be enumerated, so they are
// read by name; a name the browser does not have is skipped.
var LimitNames = []string{
	"maxTextureDimension1D", "maxTextureDimension2D", "maxTextureDimension3D", "maxTextureArrayLayers",
	"maxBindGroups", "maxBindGroupsPlusVertexBuffers", "maxImmediateSize", "maxBindingsPerBindGroup",
	"maxDynamicUniformBuffersPerPipelineLayout", "maxDynamicStorageBuffersPerPipelineLayout",
	"maxSampledTexturesPerShaderStage", "maxSamplersPerShaderStage",
	"maxStorageBuffersPerShaderStage", "maxStorageBuffersInVertexStage", "maxStorageBuffersInFragmentStage",
	"maxStorageTexturesPerShaderStage", "maxStorageTexturesInVertexStage", "maxStorageTexturesInFragmentStage",
	"maxUniformBuffersPerShaderStage", "maxUniformBufferBindingSize", "maxStorageBufferBindingSize",
	"minUniformBufferOffsetAlignment", "minStorageBufferOffsetAlignment",
	"maxVertexBuffers", "maxBufferSize", "maxVertexAttributes", "maxVertexBufferArrayStride",
	"maxInterStageShaderVariables", "maxColorAttachments", "maxColorAttachmentBytesPerSample",
	"maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup",
	"maxComputeWorkgroupSizeX", "maxComputeWorkgroupSizeY", "maxComputeWorkgroupSizeZ",
	"maxComputeWorkgroupsPerDimension",
}

// AdapterInfo is a GPUAdapterInfo. Fallback reports a fallback (software)
// adapter, e.g. SwiftShader: work done on it is not GPU work.
type AdapterInfo struct {
	Vendor, Architecture, Device, Description string
	Fallback                                  bool
}
