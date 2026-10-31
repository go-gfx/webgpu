// Package wire is the format of the batched command stream a
// webgpu.Recorder writes and replay.js executes: opcodes, argument
// layout, and the Go reader that mirrors replay.js.
//
// It is portable (no GOOS constraint) so that a stream can be carried
// outside the browser that recorded it — for example from a virtual
// machine's guest to a host over a virtio-gpu context — and checked and
// remapped there by the one implementation of the format, instead of a
// copy. The Recorder (package webgpu) is the only writer; replay.js and
// Decode are the readers, and the package tests keep them in agreement.
//
// A stream is a []uint32 of commands plus a data []byte that writeBuffer
// commands point into. Object arguments are ids: indices of a table the
// replayer holds (replay.js's object table, or a remote host's).
package wire

import "fmt"

// StreamVersion is the version of this format. It changes whenever an
// opcode's layout changes, so that a remote host and its guests can
// check they speak the same stream.
//
// 2 added CopyTextureToTexture.
const StreamVersion = 2

// Op is a command's opcode, the first word of a command.
type Op uint32

// Opcodes; after each, the words that follow it. (2) is a uint64 as
// low, high words.
const (
	SetPipeline         Op = 1  // id
	SetBindGroup        Op = 2  // index id n offsets[n] (id 0: unbind)
	Draw                Op = 3  // vertexCount instanceCount firstVertex firstInstance
	DrawIndexed         Op = 4  // indexCount instanceCount firstIndex baseVertex(int32) firstInstance
	SetVertexBuffer     Op = 5  // slot id offset(2) size(2)
	SetIndexBuffer      Op = 6  // id format offset(2) size(2)
	SetViewport         Op = 7  // x y w h minDepth maxDepth (float32 bits)
	SetScissorRect      Op = 8  // x y w h
	SetBlendConstant    Op = 9  // r g b a (float32 bits)
	SetStencilReference Op = 10 // ref
	ExecuteBundles      Op = 11 // n ids[n]
	DrawIndirect        Op = 12 // id offset(2)
	DrawIndexedIndirect Op = 13 // id offset(2)
	BeginRenderPass     Op = 20 // nColors, 8 words per color, depth flag, 9 depth words if 1
	EndPass             Op = 21
	BeginComputePass    Op = 22
	Dispatch            Op = 23 // x y z
	DispatchIndirect    Op = 24 // id offset(2)
	WriteBuffer         Op = 30 // id offset(2) dataOffset length
	CopyBufferToBuffer  Op = 31 // src srcOffset(2) dst dstOffset(2) size(2)
	// srcTexture mip x y z aspect, dstTexture mip x y z aspect, width
	// height depthOrArrayLayers. Aspect: 0 default, 1 all, 2 stencil-only,
	// 3 depth-only.
	CopyTextureToTexture Op = 32
	Submit               Op = 40
)

// Limits the readers enforce on counts inside one command.
const (
	MaxColorAttachments = 8
	MaxDynamicOffsets   = 1 << 16
	MaxBundles          = 1 << 16
)

// Cmd is one decoded command; Args aliases the stream.
type Cmd struct {
	Op   Op
	Args []uint32
}

// argc is the fixed argument count of each opcode; -1 is variable.
var argc = map[Op]int{
	SetPipeline: 1, SetBindGroup: -1, Draw: 4, DrawIndexed: 5, SetVertexBuffer: 6,
	SetIndexBuffer: 6, SetViewport: 6, SetScissorRect: 4, SetBlendConstant: 4,
	SetStencilReference: 1, ExecuteBundles: -1, DrawIndirect: 3, DrawIndexedIndirect: 3,
	BeginRenderPass: -1, EndPass: 0, BeginComputePass: 0, Dispatch: 3, DispatchIndirect: 3,
	WriteBuffer: 5, CopyBufferToBuffer: 8, CopyTextureToTexture: 15, Submit: 0,
}

// where each opcode may appear: 0 outside a pass, 1 render pass, 2
// compute pass; -1 render or compute.
var opPlace = map[Op]int{
	SetPipeline: -1, SetBindGroup: -1, Draw: 1, DrawIndexed: 1, SetVertexBuffer: 1,
	SetIndexBuffer: 1, SetViewport: 1, SetScissorRect: 1, SetBlendConstant: 1,
	SetStencilReference: 1, ExecuteBundles: 1, DrawIndirect: 1, DrawIndexedIndirect: 1,
	BeginRenderPass: 0, EndPass: -1, BeginComputePass: 0, Dispatch: 2, DispatchIndirect: 2,
	WriteBuffer: 0, CopyBufferToBuffer: 0, CopyTextureToTexture: 0, Submit: 0,
}

// Decode reads a stream with replay.js's checks and returns its commands.
// ndata is the length of the data stream; known(id) says whether a
// nonzero object id is in the replayer's table.
func Decode(w []uint32, ndata uint32, known func(uint32) bool) ([]Cmd, error) {
	var out []Cmd
	err := walk(w, ndata, func(pos int, nullable bool) error {
		id := w[pos]
		if id == 0 && nullable {
			return nil
		}
		if id == 0 || !known(id) {
			return fmt.Errorf("replay: unknown object %d", id)
		}
		return nil
	}, func(c Cmd) { out = append(out, c) })
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Remap rewrites, in place, every object id of a stream through f, which
// returns the replacement and whether the id is valid. Id 0 where the
// format allows "none" is left alone and not passed to f. Remap checks
// the stream exactly as Decode does; on error the words are partly
// rewritten and must be discarded.
//
// A transport uses it to translate ids from the namespace of the program
// that recorded the stream to the table of the replayer: an id f does not
// know is an error, so a stream can only name objects f grants.
func Remap(w []uint32, ndata uint32, f func(uint32) (uint32, bool)) error {
	return walk(w, ndata, func(pos int, nullable bool) error {
		id := w[pos]
		if id == 0 && nullable {
			return nil
		}
		to, ok := uint32(0), false
		if id != 0 {
			to, ok = f(id)
		}
		if !ok || to == 0 {
			return fmt.Errorf("replay: unknown object %d", id)
		}
		w[pos] = to
		return nil
	}, func(Cmd) {})
}

// walk is the one reader of the format: it checks structure, placement
// and data bounds, calls obj for every object-id position (pos indexes
// w), and emit for every command.
func walk(w []uint32, ndata uint32, obj func(pos int, nullable bool) error, emit func(Cmd)) error {
	pass := 0
	i := 0
	need := func(n int) error {
		if n < 0 || len(w)-i < n {
			return fmt.Errorf("replay: truncated at word %d", i)
		}
		return nil
	}
	for i < len(w) {
		start := i
		op := Op(w[i])
		i++
		n, ok := argc[op]
		if !ok {
			return fmt.Errorf("replay: bad opcode %d at word %d", op, start)
		}
		place := opPlace[op]
		if place == -1 && pass == 0 || place >= 0 && place != pass {
			return fmt.Errorf("replay: opcode %d out of place at word %d", op, start)
		}
		switch op {
		case SetBindGroup:
			if err := need(3); err != nil {
				return err
			}
			n = 3 + int(w[i+2])
			if w[i+2] > MaxDynamicOffsets {
				return fmt.Errorf("replay: %d dynamic offsets", w[i+2])
			}
		case ExecuteBundles:
			if err := need(1); err != nil {
				return err
			}
			if w[i] > MaxBundles {
				return fmt.Errorf("replay: %d bundles", w[i])
			}
			n = 1 + int(w[i])
		case BeginRenderPass:
			if err := need(1); err != nil {
				return err
			}
			nc := w[i]
			if nc > MaxColorAttachments {
				return fmt.Errorf("replay: %d color attachments", nc)
			}
			n = 1 + 8*int(nc) + 1
			if err := need(n); err != nil {
				return err
			}
			if w[i+n-1] == 1 {
				n += 9
			} else if w[i+n-1] != 0 {
				return fmt.Errorf("replay: bad depth flag at word %d", i+n-1)
			}
		}
		if err := need(n); err != nil {
			return err
		}
		at := i // position of a[0] in w
		a := w[i : i+n]
		i += n
		var err error
		switch op {
		case SetPipeline, DrawIndirect, DrawIndexedIndirect, DispatchIndirect:
			err = obj(at, false)
		case SetBindGroup:
			err = obj(at+1, true)
		case SetVertexBuffer:
			err = obj(at+1, false)
		case SetIndexBuffer:
			if err = obj(at, false); err == nil && a[1] != 1 && a[1] != 2 {
				err = fmt.Errorf("replay: index format %d", a[1])
			}
		case ExecuteBundles:
			for k := 1; k < len(a) && err == nil; k++ {
				err = obj(at+k, false)
			}
		case BeginRenderPass:
			err = renderPass(a, at, obj)
			pass = 1
		case BeginComputePass:
			pass = 2
		case EndPass:
			pass = 0
		case WriteBuffer:
			if err = obj(at, false); err == nil && uint64(a[3])+uint64(a[4]) > uint64(ndata) {
				err = fmt.Errorf("replay: data [%d, +%d) past %d bytes", a[3], a[4], ndata)
			}
		case CopyBufferToBuffer:
			if err = obj(at, false); err == nil {
				err = obj(at+3, false)
			}
		case CopyTextureToTexture:
			if err = obj(at, false); err == nil {
				err = obj(at+6, false)
			}
			if err == nil && (a[5] > 3 || a[11] > 3) {
				err = fmt.Errorf("replay: bad texture aspect")
			}
		}
		if err != nil {
			return err
		}
		emit(Cmd{Op: op, Args: a})
	}
	if pass != 0 {
		return fmt.Errorf("replay: pass left open")
	}
	return nil
}

// renderPass checks a beginRenderPass's attachments; a starts at w[at].
func renderPass(a []uint32, at int, obj func(int, bool) error) error {
	nc := int(a[0])
	for c := 0; c < nc; c++ {
		o := 1 + 8*c
		ca := a[o : o+8]
		if ca[0] == 0 { // an unused slot is all zeros
			for _, v := range ca {
				if v != 0 {
					return fmt.Errorf("replay: unused color slot %d is not empty", c)
				}
			}
			continue
		}
		if err := obj(at+o, false); err != nil {
			return err
		}
		if err := obj(at+o+1, true); err != nil {
			return err
		}
		if ca[2] > 2 || ca[3] > 2 {
			return fmt.Errorf("replay: bad load or store op")
		}
	}
	if a[1+8*nc] == 1 {
		o := 2 + 8*nc
		ds := a[o:]
		if err := obj(at+o, false); err != nil {
			return err
		}
		if ds[1] > 2 || ds[2] > 2 || ds[5] > 2 || ds[6] > 2 || ds[4] > 1 || ds[8] > 1 {
			return fmt.Errorf("replay: bad depth-stencil ops")
		}
	}
	return nil
}
