package wire

import (
	"encoding/binary"
	"fmt"
)

// Kind is the kind of object a creation descriptor makes.
type Kind uint32

// Object kinds.
const (
	KindBuffer          Kind = 1
	KindTexture         Kind = 2
	KindTextureView     Kind = 3
	KindSampler         Kind = 4
	KindShaderModule    Kind = 5
	KindBindGroupLayout Kind = 6
	KindPipelineLayout  Kind = 7
	KindBindGroup       Kind = 8
	KindRenderPipeline  Kind = 9
	KindComputePipeline Kind = 10
)

// Valid reports whether k is a known kind.
func (k Kind) Valid() bool { return k >= KindBuffer && k <= KindComputePipeline }

// MaxRefs bounds the object references of one descriptor (a bind group:
// its layout and one per entry; WebGPU's maxBindingsPerBindGroup
// default is 1000).
const MaxRefs = 4096

// A creation descriptor on the wire is
//
//	nrefs u32 | refs[nrefs] u32 | body
//
// little-endian. refs are the ids of the objects the descriptor names,
// in an order its kind fixes (package webgpu's EncodeDescriptor), 0 for
// an optional reference left out; body is the descriptor itself with
// those references removed. Keeping every reference in the header lets a
// transport check and translate them (RemapDescriptor) without reading
// the body.

// AppendDescriptor appends a descriptor to dst.
func AppendDescriptor(dst []byte, refs []uint32, body []byte) []byte {
	dst = binary.LittleEndian.AppendUint32(dst, uint32(len(refs)))
	for _, r := range refs {
		dst = binary.LittleEndian.AppendUint32(dst, r)
	}
	return append(dst, body...)
}

// SplitDescriptor returns a descriptor's references and body (aliasing b).
func SplitDescriptor(b []byte) (refs []uint32, body []byte, err error) {
	n, err := descRefs(b)
	if err != nil {
		return nil, nil, err
	}
	refs = make([]uint32, n)
	for i := range refs {
		refs[i] = binary.LittleEndian.Uint32(b[4+4*i:])
	}
	return refs, b[4+4*n:], nil
}

// RemapDescriptor rewrites, in place, every nonzero reference of a
// descriptor through f, and refuses any reference f does not grant: a
// descriptor can only name objects the transport allows. 0 (an optional
// reference left out) stays 0.
func RemapDescriptor(b []byte, f func(uint32) (uint32, bool)) error {
	n, err := descRefs(b)
	if err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		p := b[4+4*i:]
		id := binary.LittleEndian.Uint32(p)
		if id == 0 {
			continue
		}
		to, ok := f(id)
		if !ok || to == 0 {
			return fmt.Errorf("descriptor: unknown object %d", id)
		}
		binary.LittleEndian.PutUint32(p, to)
	}
	return nil
}

func descRefs(b []byte) (int, error) {
	if len(b) < 4 {
		return 0, fmt.Errorf("descriptor: %d bytes", len(b))
	}
	n := binary.LittleEndian.Uint32(b)
	if n > MaxRefs || uint64(len(b)-4) < 4*uint64(n) {
		return 0, fmt.Errorf("descriptor: %d references in %d bytes", n, len(b))
	}
	return int(n), nil
}
