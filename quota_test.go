package webgpu

import (
	"errors"
	"strings"
	"sync"
	"testing"
)

func TestQuotaNil(t *testing.T) {
	var q *Quota
	if err := q.chargeMemory(1 << 40); err != nil {
		t.Fatal(err)
	}
	q.refundMemory(1)
	if err := q.chargePipeline(); err != nil {
		t.Fatal(err)
	}
	q.refundPipeline()
	if err := q.checkShader(1 << 30); err != nil {
		t.Fatal(err)
	}
	if err := q.checkDevice(&DeviceDescriptor{RequiredFeatures: []FeatureName{"x"}}); err != nil {
		t.Fatal(err)
	}
	if m, p := q.Usage(); m != 0 || p != 0 {
		t.Fatal(m, p)
	}
}

func TestQuotaMemoryAndPipelines(t *testing.T) {
	q := &Quota{MemoryBytes: 100, Pipelines: 2, ShaderBytes: 10}
	if err := q.chargeMemory(60); err != nil {
		t.Fatal(err)
	}
	var qe *QuotaError
	if err := q.chargeMemory(41); !errors.As(err, &qe) || qe.Resource != "memory" || qe.Want != 101 {
		t.Fatalf("got %v", err)
	}
	if err := q.chargeMemory(-1); err == nil {
		t.Fatal("negative charge accepted")
	}
	q.refundMemory(60)
	if err := q.chargeMemory(100); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if err := q.chargePipeline(); err != nil {
			t.Fatal(err)
		}
	}
	if err := q.chargePipeline(); !errors.As(err, &qe) || qe.Resource != "pipelines" {
		t.Fatalf("got %v", err)
	}
	if m, p := q.Usage(); m != 100 || p != 2 {
		t.Fatal(m, p)
	}
	q.refundPipeline()
	if err := q.checkShader(10); err != nil {
		t.Fatal(err)
	}
	if err := q.checkShader(11); !errors.As(err, &qe) || qe.Resource != "shader" {
		t.Fatalf("got %v", err)
	}
	if !strings.Contains(qe.Error(), "11 requested, 10 allowed") {
		t.Fatal(qe.Error())
	}
	// Zero limits are no limit.
	z := &Quota{}
	if err := z.chargeMemory(1 << 50); err != nil {
		t.Fatal(err)
	}
	if err := z.chargePipeline(); err != nil {
		t.Fatal(err)
	}
}

func TestQuotaDevice(t *testing.T) {
	q := &Quota{Features: []FeatureName{FeatureTextureCompressionBC}, Limits: map[string]uint64{"maxBufferSize": 1 << 20}}
	if err := q.checkDevice(nil); err != nil {
		t.Fatal(err)
	}
	ok := &DeviceDescriptor{RequiredFeatures: []FeatureName{FeatureTextureCompressionBC}, RequiredLimits: map[string]uint64{"maxBufferSize": 1 << 20}}
	if err := q.checkDevice(ok); err != nil {
		t.Fatal(err)
	}
	var qe *QuotaError
	if err := q.checkDevice(&DeviceDescriptor{RequiredFeatures: []FeatureName{FeatureTimestampQuery}}); !errors.As(err, &qe) || qe.Name != "timestamp-query" {
		t.Fatalf("got %v", err)
	}
	if !strings.Contains(qe.Error(), `feature "timestamp-query" refused`) {
		t.Fatal(qe.Error())
	}
	if err := q.checkDevice(&DeviceDescriptor{RequiredLimits: map[string]uint64{"maxBufferSize": 1<<20 + 1}}); !errors.As(err, &qe) || qe.Resource != "limit" {
		t.Fatalf("got %v", err)
	}
	if err := q.checkDevice(&DeviceDescriptor{RequiredLimits: map[string]uint64{"maxBindGroups": 4}}); err == nil {
		t.Fatal("an unnamed limit was accepted")
	}
}

func TestQuotaConcurrent(t *testing.T) {
	q := &Quota{MemoryBytes: 1000}
	var wg sync.WaitGroup
	var mu sync.Mutex
	ok := 0
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if q.chargeMemory(100) == nil {
				mu.Lock()
				ok++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if ok != 10 {
		t.Fatalf("%d charges of 100 fit a quota of 1000", ok)
	}
}

func TestDeviceAccounting(t *testing.T) {
	q := &Quota{MemoryBytes: 100, Pipelines: 1}
	d := newDevice(jsValue{}, "dev", q)
	if err := d.charge(60, 1); err != nil {
		t.Fatal(err)
	}
	// A pipeline over the quota refunds the memory charged with it.
	if err := d.charge(10, 1); err == nil {
		t.Fatal("second pipeline accepted")
	}
	if m, p := q.Usage(); m != 60 || p != 1 {
		t.Fatal(m, p)
	}
	d.refund(60, 1)
	if m, p := q.Usage(); m != 0 || p != 0 {
		t.Fatal(m, p)
	}
	// Two pipelines at once: the first is refunded when the second fails.
	if err := d.charge(0, 2); err == nil {
		t.Fatal("two pipelines accepted")
	}
	if _, p := q.Usage(); p != 0 {
		t.Fatal(p)
	}
	if err := d.charge(200, 0); err == nil {
		t.Fatal("over memory accepted")
	}
	if err := d.charge(50, 1); err != nil {
		t.Fatal(err)
	}
	// Losing the device refunds everything once; later refunds are no-ops.
	d.setLost(DeviceLostInfo{Reason: "destroyed", Message: "bye"})
	d.setLost(DeviceLostInfo{Reason: "again"})
	select {
	case <-d.Lost():
	default:
		t.Fatal("Lost not closed")
	}
	if d.LostInfo().Reason != "destroyed" {
		t.Fatal(d.LostInfo())
	}
	if m, p := q.Usage(); m != 0 || p != 0 {
		t.Fatal(m, p)
	}
	d.refund(50, 1)
	d.releaseAll()
	if m, p := q.Usage(); m != 0 || p != 0 {
		t.Fatal(m, p)
	}
	if err := d.charge(1, 0); !errors.Is(err, ErrDestroyed) {
		t.Fatal(err)
	}
	if d.Quota() != q {
		t.Fatal("Quota")
	}
}

func TestOwned(t *testing.T) {
	d := newDevice(jsValue{}, "dev", nil)
	d.limits = map[string]uint64{"maxBufferSize": 256}
	if v, ok := d.Limit("maxBufferSize"); !ok || v != 256 {
		t.Fatal(v, ok)
	}
	b := &Buffer{size: 8, usage: BufferUsageVertex}
	b.dev = d
	b.label = "vb"
	if b.Size() != 8 || b.Usage() != BufferUsageVertex || b.Label() != "vb" {
		t.Fatal(b)
	}
	if b.isDead() {
		t.Fatal("dead at birth")
	}
	if !b.markDead() || b.markDead() {
		t.Fatal("markDead must succeed once")
	}
	if !b.isDead() {
		t.Fatal("not dead")
	}
	c := &Buffer{}
	c.dev = d
	d.setLost(DeviceLostInfo{})
	if !c.isDead() {
		t.Fatal("a lost device's buffer is alive")
	}
	tx := tex(TextureDescriptor{Size: Extent3D{Width: 4, Height: 2, DepthOrArrayLayers: 3}, Format: "r8uint"})
	if tx.Width() != 4 || tx.Height() != 2 || tx.DepthOrArrayLayers() != 3 || tx.Format() != "r8uint" || tx.Descriptor().MipLevelCount != 1 {
		t.Fatal(tx.desc)
	}
	tv := &TextureView{tex: tx}
	if tv.Texture() != tx {
		t.Fatal("view texture")
	}
}

func TestErrorStrings(t *testing.T) {
	for _, c := range []struct {
		err  error
		want string
	}{
		{&BoundsError{Op: "writeBuffer", Msg: "x"}, "webgpu: writeBuffer: x"},
		{&JSError{Op: "createBuffer", Name: "TypeError", Message: "bad"}, "webgpu: createBuffer: TypeError: bad"},
		{&GPUError{Kind: "validation", Message: "m"}, "webgpu: validation error: m"},
		{&QuotaError{Resource: "limit", Name: "maxBindGroups", Limit: 4, Want: 8}, `webgpu: quota: limit "maxBindGroups" refused (allowed 4, requested 8)`},
	} {
		if got := c.err.Error(); got != c.want {
			t.Errorf("got %q, want %q", got, c.want)
		}
	}
}
