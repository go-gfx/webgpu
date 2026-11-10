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
// 2 added CopyTextureToTexture. 3 added WriteTexture. 4 added ReadBuffer.
// 5 added CopyTextureToBuffer.
const StreamVersion = 5

// CopyBytesPerRowAlignment is the multiple a CopyTextureToBuffer's
// bytesPerRow must be (WebGPU: encoder texel copies, 256).
const CopyBytesPerRowAlignment = 256

// MaxReadWindow bounds the read window of one submit (ReadBuffer's
// windowOffset + size), 64 MiB. A host grants a smaller one.
const MaxReadWindow = 64 << 20

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
	// texture mip x y z aspect dataOffset dataLength layoutOffset(2)
	// bytesPerRow rowsPerImage width height depthOrArrayLayers. The bytes
	// are data[dataOffset, dataOffset+dataLength); layoutOffset is
	// relative to dataOffset (queue.writeTexture's dataLayout.offset).
	// Aspect as in CopyTextureToTexture.
	WriteTexture Op = 33
	// id offset(2) size windowOffset. Read-back (GPUBuffer.mapAsync with
	// MAP_READ, then getMappedRange and unmap): once the stream's GPU work
	// has completed, the bytes [offset, offset+size) of buffer id are
	// copied into [windowOffset, windowOffset+size) of the submit's read
	// window, a byte range the caller of the replay owns. The buffer needs
	// MAP_READ; offset is a multiple of 8, size a nonzero multiple of 4
	// (the mapAsync rules of the WebGPU specification). replay.js ignores
	// the command: a host performs it after the replay, in stream order.
	ReadBuffer Op = 34
	// texture mip x y z aspect buffer offset(2) bytesPerRow rowsPerImage
	// width height depthOrArrayLayers: encoder.copyTextureToBuffer. Aspect as
	// in CopyTextureToTexture. bytesPerRow is a multiple of 256 (0 is one
	// block row at most). The texture needs COPY_SRC and the buffer COPY_DST;
	// a host checks the region against the texture's mip level and the layout
	// against the buffer's size (TextureCopyOf, webgpu.CheckCopyTextureToBuffer).
	CopyTextureToBuffer Op = 35
	Submit              Op = 40
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
	WriteBuffer: 5, CopyBufferToBuffer: 8, CopyTextureToTexture: 15, WriteTexture: 15, ReadBuffer: 5, CopyTextureToBuffer: 14, Submit: 0,
}

// where each opcode may appear: 0 outside a pass, 1 render pass, 2
// compute pass; -1 render or compute.
var opPlace = map[Op]int{
	SetPipeline: -1, SetBindGroup: -1, Draw: 1, DrawIndexed: 1, SetVertexBuffer: 1,
	SetIndexBuffer: 1, SetViewport: 1, SetScissorRect: 1, SetBlendConstant: 1,
	SetStencilReference: 1, ExecuteBundles: 1, DrawIndirect: 1, DrawIndexedIndirect: 1,
	BeginRenderPass: 0, EndPass: -1, BeginComputePass: 0, Dispatch: 2, DispatchIndirect: 2,
	WriteBuffer: 0, CopyBufferToBuffer: 0, CopyTextureToTexture: 0, WriteTexture: 0, ReadBuffer: 0, CopyTextureToBuffer: 0, Submit: 0,
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
	return walk(w, ndata, remapper(w, f), func(Cmd) {})
}

// RemapCmds is Remap that also returns every command, in order, with its
// ids translated (Args alias w).
func RemapCmds(w []uint32, ndata uint32, f func(uint32) (uint32, bool)) ([]Cmd, error) {
	var out []Cmd
	if err := walk(w, ndata, remapper(w, f), func(c Cmd) { out = append(out, c) }); err != nil {
		return nil, err
	}
	return out, nil
}

func remapper(w []uint32, f func(uint32) (uint32, bool)) func(int, bool) error {
	return func(pos int, nullable bool) error {
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
	}
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
		case WriteTexture:
			if err = obj(at, false); err == nil && a[5] > 3 {
				err = fmt.Errorf("replay: bad texture aspect")
			}
			if err == nil && uint64(a[6])+uint64(a[7]) > uint64(ndata) {
				err = fmt.Errorf("replay: data [%d, +%d) past %d bytes", a[6], a[7], ndata)
			}
		case ReadBuffer:
			if err = obj(at, false); err == nil {
				err = checkRead(a)
			}
		case CopyTextureToBuffer:
			if err = obj(at, false); err == nil {
				err = obj(at+6, false)
			}
			if err == nil && a[5] > 3 {
				err = fmt.Errorf("replay: bad texture aspect")
			}
			if err == nil && a[9]%CopyBytesPerRowAlignment != 0 {
				err = fmt.Errorf("replay: bytesPerRow %d is not a multiple of %d", a[9], CopyBytesPerRowAlignment)
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

// Read is a decoded ReadBuffer command.
type Read struct {
	ID           uint32 // the buffer (the replayer's id after Remap)
	Offset       uint64 // into the buffer, a multiple of 8
	Size         uint32 // a nonzero multiple of 4
	WindowOffset uint32 // into the read window
}

// ReadOf decodes c if it is a ReadBuffer command a reader returned.
func ReadOf(c Cmd) (Read, bool) {
	if c.Op != ReadBuffer || len(c.Args) != 5 {
		return Read{}, false
	}
	a := c.Args
	return Read{ID: a[0], Offset: uint64(a[1]) | uint64(a[2])<<32, Size: a[3], WindowOffset: a[4]}, true
}

// TextureCopy is a decoded CopyTextureToBuffer command.
type TextureCopy struct {
	Texture              uint32 // the replayer's id after Remap
	MipLevel             uint32
	OriginX, OriginY     uint32
	OriginZ              uint32
	Aspect               uint32 // 0 default, 1 all, 2 stencil-only, 3 depth-only
	Buffer               uint32
	Offset               uint64 // into the buffer
	BytesPerRow          uint32 // a multiple of 256
	RowsPerImage         uint32
	Width, Height, Depth uint32
}

// TextureCopyOf decodes c if it is a CopyTextureToBuffer command a reader
// returned.
func TextureCopyOf(c Cmd) (TextureCopy, bool) {
	if c.Op != CopyTextureToBuffer || len(c.Args) != 14 {
		return TextureCopy{}, false
	}
	a := c.Args
	return TextureCopy{Texture: a[0], MipLevel: a[1], OriginX: a[2], OriginY: a[3], OriginZ: a[4], Aspect: a[5],
		Buffer: a[6], Offset: u64(a[7], a[8]), BytesPerRow: a[9], RowsPerImage: a[10],
		Width: a[11], Height: a[12], Depth: a[13]}, true
}

// RemapReads is Remap that also returns the stream's ReadBuffer commands,
// in order, with their ids translated. A host checks them against its
// buffers' sizes and usage, and its window, before it replays anything.
func RemapReads(w []uint32, ndata uint32, f func(uint32) (uint32, bool)) ([]Read, error) {
	var reads []Read
	err := walk(w, ndata, remapper(w, f), func(c Cmd) {
		if r, ok := ReadOf(c); ok {
			reads = append(reads, r)
		}
	})
	if err != nil {
		return nil, err
	}
	return reads, nil
}

// checkRead checks a ReadBuffer's arguments (the mapAsync alignment rules
// and the window's bound); a is the command's argument words.
func checkRead(a []uint32) error {
	switch {
	case a[1]&7 != 0:
		return fmt.Errorf("replay: read offset %d is not a multiple of 8", uint64(a[1])|uint64(a[2])<<32)
	case a[3] == 0 || a[3]&3 != 0:
		return fmt.Errorf("replay: read size %d is not a nonzero multiple of 4", a[3])
	case uint64(a[4])+uint64(a[3]) > MaxReadWindow:
		return fmt.Errorf("replay: read window [%d, +%d) past %d bytes", a[4], a[3], MaxReadWindow)
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
