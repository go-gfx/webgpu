//go:build js

package webgpu

import (
	"context"
	"testing"
)

// TestRecorderReadBufferBrowser: on a browser device a Recorder is its own
// host: Submit waits for the GPU and maps the buffer, so ReadBuffer returns
// what the recorded copy wrote.
func TestRecorderReadBufferBrowser(t *testing.T) {
	d := newDev(t, nil)
	if err := InstallReplay(context.Background(), ""); err != nil {
		t.Fatal(err)
	}
	src := must(d.CreateBuffer(BufferDescriptor{Size: 32, Usage: BufferUsageCopySrc | BufferUsageCopyDst}))(t)
	dst := must(d.CreateBuffer(BufferDescriptor{Size: 32, Usage: BufferUsageMapRead | BufferUsageCopyDst}))(t)
	data := make([]byte, 32)
	for i := range data {
		data[i] = byte(i * 3)
	}
	r := d.NewRecorder()
	r.WriteBuffer(src, 0, data)
	r.CopyBufferToBuffer(src, 0, dst, 0, 32)
	a := r.ReadBuffer(dst, 8, 16)
	if _, err := r.Submit(); err != nil {
		t.Fatal(err)
	}
	got, err := a.Bytes()
	if err != nil || string(got) != string(data[8:24]) {
		t.Fatalf("%v %v", got, err)
	}
	if rb := d.NewRecorder().ReadBuffer(src, 0, 4); rb.err == nil {
		t.Error("a buffer without MAP_READ was accepted")
	}
}
