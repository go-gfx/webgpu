package webgpu

import "sync"

// Quota bounds what devices may hold. An embedder that runs untrusted code
// through this package (the go-wasmcontainers playground does) gives each
// tenant a Quota and passes it in DeviceDescriptor.Quota; several devices
// may share one. A nil Quota is no limit.
//
// The package charges a Quota before it calls the browser and refunds on
// Destroy or Release, so a refused request never reaches WebGPU.
type Quota struct {
	// MemoryBytes bounds buffer and texture bytes alive (texture sizes are
	// computed by the package from the descriptor, mips and samples
	// included). 0 is no limit.
	MemoryBytes int64
	// Pipelines bounds render and compute pipelines alive. 0 is no limit.
	Pipelines int
	// ShaderBytes bounds one WGSL source. 0 is no limit.
	ShaderBytes int
	// Features lists the optional features a device may require. Any
	// other required feature is refused, timestamp-query included unless
	// listed.
	Features []FeatureName
	// Limits caps the limits a device may require; a required limit not
	// named here is refused.
	Limits map[string]uint64

	mu        sync.Mutex
	memory    int64
	pipelines int
}

// Usage reports what is charged now.
func (q *Quota) Usage() (memoryBytes int64, pipelines int) {
	if q == nil {
		return 0, 0
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.memory, q.pipelines
}

func (q *Quota) chargeMemory(n int64) error {
	if q == nil {
		return nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if n < 0 || (q.MemoryBytes > 0 && n > q.MemoryBytes-q.memory) {
		return &QuotaError{Resource: "memory", Limit: q.MemoryBytes, Want: q.memory + n}
	}
	q.memory += n
	return nil
}

func (q *Quota) refundMemory(n int64) {
	if q == nil {
		return
	}
	q.mu.Lock()
	q.memory -= n
	q.mu.Unlock()
}

func (q *Quota) chargePipeline() error {
	if q == nil {
		return nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.Pipelines > 0 && q.pipelines >= q.Pipelines {
		return &QuotaError{Resource: "pipelines", Limit: int64(q.Pipelines), Want: int64(q.pipelines + 1)}
	}
	q.pipelines++
	return nil
}

func (q *Quota) refundPipeline() {
	if q == nil {
		return
	}
	q.mu.Lock()
	q.pipelines--
	q.mu.Unlock()
}

func (q *Quota) checkShader(n int) error {
	if q == nil || q.ShaderBytes <= 0 || n <= q.ShaderBytes {
		return nil
	}
	return &QuotaError{Resource: "shader", Limit: int64(q.ShaderBytes), Want: int64(n)}
}

// checkDevice refuses required features and limits the quota does not
// allow.
func (q *Quota) checkDevice(d *DeviceDescriptor) error {
	if q == nil || d == nil {
		return nil
	}
	for _, f := range d.RequiredFeatures {
		ok := false
		for _, a := range q.Features {
			if a == f {
				ok = true
				break
			}
		}
		if !ok {
			return &QuotaError{Resource: "feature", Name: string(f)}
		}
	}
	for name, v := range d.RequiredLimits {
		max, ok := q.Limits[name]
		if !ok || v > max {
			return &QuotaError{Resource: "limit", Name: name, Limit: int64(max), Want: int64(v)}
		}
	}
	return nil
}
