package webgpu

import "fmt"

// decode is the Go mirror of replay.js: it reads a recorded stream with
// the same checks and returns the commands. Tests compare it with the
// recorder (every stream written decodes to what was recorded) and with
// replay.js (both readers agree on every stream), and fuzz it; the
// browser never runs it.

type cmd struct {
	op   opcode
	args []uint32
}

// argc is the fixed argument count of each opcode; -1 is variable.
var argc = map[opcode]int{
	opSetPipeline: 1, opSetBindGroup: -1, opDraw: 4, opDrawIndexed: 5, opSetVertexBuffer: 6,
	opSetIndexBuffer: 6, opSetViewport: 6, opSetScissorRect: 4, opSetBlendConstant: 4,
	opSetStencilReference: 1, opExecuteBundles: -1, opDrawIndirect: 3, opDrawIndexedIndirect: 3,
	opBeginRenderPass: -1, opEndPass: 0, opBeginComputePass: 0, opDispatch: 3, opDispatchIndirect: 3,
	opWriteBuffer: 5, opCopyBufferToBuffer: 8, opSubmit: 0,
}

// where each opcode may appear: 0 outside a pass, 1 render pass, 2
// compute pass; -1 render or compute.
var opPlace = map[opcode]int{
	opSetPipeline: -1, opSetBindGroup: -1, opDraw: 1, opDrawIndexed: 1, opSetVertexBuffer: 1,
	opSetIndexBuffer: 1, opSetViewport: 1, opSetScissorRect: 1, opSetBlendConstant: 1,
	opSetStencilReference: 1, opExecuteBundles: 1, opDrawIndirect: 1, opDrawIndexedIndirect: 1,
	opBeginRenderPass: 0, opEndPass: -1, opBeginComputePass: 0, opDispatch: 2, opDispatchIndirect: 2,
	opWriteBuffer: 0, opCopyBufferToBuffer: 0, opSubmit: 0,
}

// decode reads words; ndata is the length of the data stream, known(id)
// says whether an object id is in the table.
func decode(w []uint32, ndata uint32, known func(uint32) bool) ([]cmd, error) {
	var out []cmd
	pass := 0
	i := 0
	need := func(n int) error {
		if n < 0 || len(w)-i < n {
			return fmt.Errorf("replay: truncated at word %d", i)
		}
		return nil
	}
	obj := func(id uint32, nullable bool) error {
		if id == 0 && nullable {
			return nil
		}
		if id == 0 || !known(id) {
			return fmt.Errorf("replay: unknown object %d", id)
		}
		return nil
	}
	for i < len(w) {
		start := i
		op := opcode(w[i])
		i++
		n, ok := argc[op]
		if !ok {
			return nil, fmt.Errorf("replay: bad opcode %d at word %d", op, start)
		}
		place := opPlace[op]
		if place == -1 && pass == 0 || place >= 0 && place != pass {
			return nil, fmt.Errorf("replay: opcode %d out of place at word %d", op, start)
		}
		switch op {
		case opSetBindGroup:
			if err := need(3); err != nil {
				return nil, err
			}
			n = 3 + int(w[i+2])
			if w[i+2] > 1<<16 {
				return nil, fmt.Errorf("replay: %d dynamic offsets", w[i+2])
			}
		case opExecuteBundles:
			if err := need(1); err != nil {
				return nil, err
			}
			if w[i] > 1<<16 {
				return nil, fmt.Errorf("replay: %d bundles", w[i])
			}
			n = 1 + int(w[i])
		case opBeginRenderPass:
			if err := need(1); err != nil {
				return nil, err
			}
			nc := w[i]
			if nc > 8 {
				return nil, fmt.Errorf("replay: %d color attachments", nc)
			}
			n = 1 + 8*int(nc) + 1
			if err := need(n); err != nil {
				return nil, err
			}
			if w[i+n-1] == 1 {
				n += 9
			} else if w[i+n-1] != 0 {
				return nil, fmt.Errorf("replay: bad depth flag at word %d", i+n-1)
			}
		}
		if err := need(n); err != nil {
			return nil, err
		}
		a := w[i : i+n]
		i += n
		switch op {
		case opSetPipeline, opDrawIndirect, opDrawIndexedIndirect, opDispatchIndirect:
			if err := obj(a[0], false); err != nil {
				return nil, err
			}
		case opSetBindGroup:
			if err := obj(a[1], true); err != nil {
				return nil, err
			}
		case opSetVertexBuffer:
			if err := obj(a[1], false); err != nil {
				return nil, err
			}
		case opSetIndexBuffer:
			if err := obj(a[0], false); err != nil {
				return nil, err
			}
			if a[1] != 1 && a[1] != 2 {
				return nil, fmt.Errorf("replay: index format %d", a[1])
			}
		case opExecuteBundles:
			for _, id := range a[1:] {
				if err := obj(id, false); err != nil {
					return nil, err
				}
			}
		case opBeginRenderPass:
			nc := int(a[0])
			for c := 0; c < nc; c++ {
				ca := a[1+8*c : 9+8*c]
				if ca[0] == 0 { // an unused slot is all zeros
					for _, v := range ca {
						if v != 0 {
							return nil, fmt.Errorf("replay: unused color slot %d is not empty", c)
						}
					}
					continue
				}
				if err := obj(ca[0], false); err != nil {
					return nil, err
				}
				if err := obj(ca[1], true); err != nil {
					return nil, err
				}
				if ca[2] > 2 || ca[3] > 2 {
					return nil, fmt.Errorf("replay: bad load or store op")
				}
			}
			if a[1+8*nc] == 1 {
				ds := a[2+8*nc:]
				if err := obj(ds[0], false); err != nil {
					return nil, err
				}
				if ds[1] > 2 || ds[2] > 2 || ds[5] > 2 || ds[6] > 2 || ds[4] > 1 || ds[8] > 1 {
					return nil, fmt.Errorf("replay: bad depth-stencil ops")
				}
			}
			pass = 1
		case opBeginComputePass:
			pass = 2
		case opEndPass:
			pass = 0
		case opWriteBuffer:
			if err := obj(a[0], false); err != nil {
				return nil, err
			}
			if uint64(a[3])+uint64(a[4]) > uint64(ndata) {
				return nil, fmt.Errorf("replay: data [%d, +%d) past %d bytes", a[3], a[4], ndata)
			}
		case opCopyBufferToBuffer:
			if err := obj(a[0], false); err != nil {
				return nil, err
			}
			if err := obj(a[3], false); err != nil {
				return nil, err
			}
		}
		out = append(out, cmd{op: op, args: a})
	}
	if pass != 0 {
		return nil, fmt.Errorf("replay: pass left open")
	}
	return out, nil
}
