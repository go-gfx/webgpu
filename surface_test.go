package webgpu

import (
	"errors"
	"testing"
)

func TestSurfaceSizeAndConfig(t *testing.T) {
	var nilQ *Quota
	if err := nilQ.checkSurfaceSize(4096, 4096); err != nil {
		t.Fatal(err)
	}
	var be *BoundsError
	if err := nilQ.checkSurfaceSize(0, 10); !errors.As(err, &be) {
		t.Fatal(err)
	}
	q := &Quota{SurfaceWidth: 1920, SurfaceHeight: 1080}
	if err := q.checkSurfaceSize(1920, 1080); err != nil {
		t.Fatal(err)
	}
	var qe *QuotaError
	if err := q.checkSurfaceSize(1921, 1080); !errors.As(err, &qe) || qe.Resource != "surface" {
		t.Fatal(err)
	}
	if err := q.checkSurfaceSize(1920, 1081); !errors.As(err, &qe) {
		t.Fatal(err)
	}
	if err := (&Quota{SurfaceWidth: 64}).checkSurfaceSize(64, 100000); err != nil {
		t.Fatal("an unset height limits nothing:", err)
	}
	eq(t, SurfaceDescriptor{}.config("dev", "bgra8unorm"), m{"device": "dev", "format": "bgra8unorm"})
	eq(t, SurfaceDescriptor{Format: "rgba8unorm", Usage: TextureUsageRenderAttachment | TextureUsageCopySrc, AlphaMode: CanvasAlphaModePremultiplied}.config("dev", "bgra8unorm"),
		m{"device": "dev", "format": "rgba8unorm", "usage": uint32(0x11), "alphaMode": "premultiplied"})
}
