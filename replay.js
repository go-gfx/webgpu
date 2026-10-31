// replay.js: the JavaScript half of github.com/go-gfx/webgpu's batched
// encoding. Go records a submit as uint32 words (and the bytes of its
// queue writes); Recorder.Submit copies both into arrays it owns and
// makes one call to replay(), which issues the WebGPU calls. Its Go mirror
// is package wire (wire.Decode): both read the same format with the same
// checks, and the package's tests hold them to agreeing.
//
// Load it as a same-origin script (importScripts in a worker, a <script>
// element in a window) before the Go program records; under a strict CSP
// (script-src 'self') serve it from the program's origin.
"use strict";
(() => {
  if (globalThis.goWebGPUReplay) return;
  const argc = {
    1: 1, 2: -1, 3: 4, 4: 5, 5: 6, 6: 6, 7: 6, 8: 4, 9: 4, 10: 1, 11: -1, 12: 3, 13: 3,
    20: -1, 21: 0, 22: 0, 23: 3, 24: 3, 30: 5, 31: 8, 40: 0,
  };
  // 0 outside a pass, 1 render, 2 compute, -1 either pass.
  const place = {
    1: -1, 2: -1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1,
    20: 0, 21: -1, 22: 0, 23: 2, 24: 2, 30: 0, 31: 0, 40: 0,
  };
  const loadOps = [undefined, "load", "clear"];
  const storeOps = [undefined, "store", "discard"];
  const indexFormats = [undefined, "uint16", "uint32"];
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  const fl = (bits) => { u32[0] = bits; return f32[0]; };
  const big = (lo, hi) => lo + hi * 4294967296;

  const R = {
    objs: [undefined],
    lastBuf: null, last32: null,
    put(id, v) { this.objs[id] = v; },
    drop(id) { this.objs[id] = undefined; },
    obj(id, nullable) {
      if (id === 0 && nullable) return null;
      const v = id === 0 ? undefined : this.objs[id];
      if (v === undefined) throw new Error("replay: unknown object " + id);
      return v;
    },
    // replay issues the first nwords words of cmd (a Uint8Array over a
    // whole ArrayBuffer, the recorder's own) as WebGPU calls; the first
    // ndata bytes of data are the queue writes' bytes.
    replay(device, queue, cmd, nwords, data, ndata) {
      if (cmd.buffer !== this.lastBuf) {
        this.lastBuf = cmd.buffer;
        this.last32 = new Uint32Array(cmd.buffer);
      }
      const w = this.last32;
      if (nwords > w.length || ndata > data.length) throw new Error("replay: lengths past the arrays");
      let i = 0, pass = 0, enc = null, pe = null;
      const need = (n) => { if (n < 0 || nwords - i < n) throw new Error("replay: truncated at word " + i); };
      while (i < nwords) {
        const start = i, op = w[i++];
        let n = argc[op];
        if (n === undefined) throw new Error("replay: bad opcode " + op + " at word " + start);
        const pl = place[op];
        if ((pl === -1 && pass === 0) || (pl >= 0 && pl !== pass)) throw new Error("replay: opcode " + op + " out of place at word " + start);
        if (op === 2) {
          need(3);
          if (w[i + 2] > 65536) throw new Error("replay: " + w[i + 2] + " dynamic offsets");
          n = 3 + w[i + 2];
        } else if (op === 11) {
          need(1);
          if (w[i] > 65536) throw new Error("replay: " + w[i] + " bundles");
          n = 1 + w[i];
        } else if (op === 20) {
          need(1);
          const nc = w[i];
          if (nc > 8) throw new Error("replay: " + nc + " color attachments");
          n = 1 + 8 * nc + 1;
          need(n);
          const flag = w[i + n - 1];
          if (flag === 1) n += 9;
          else if (flag !== 0) throw new Error("replay: bad depth flag at word " + (i + n - 1));
        }
        need(n);
        const a = i;
        i += n;
        switch (op) {
          case 1: pe.setPipeline(this.obj(w[a], false)); break;
          case 2: {
            const g = this.obj(w[a + 1], true), nd = w[a + 2];
            if (nd === 0) pe.setBindGroup(w[a], g);
            else pe.setBindGroup(w[a], g, w, a + 3, nd);
            break;
          }
          case 3: pe.draw(w[a], w[a + 1], w[a + 2], w[a + 3]); break;
          case 4: pe.drawIndexed(w[a], w[a + 1], w[a + 2], w[a + 3] | 0, w[a + 4]); break;
          case 5: pe.setVertexBuffer(w[a], this.obj(w[a + 1], false), big(w[a + 2], w[a + 3]), big(w[a + 4], w[a + 5])); break;
          case 6: {
            const f = indexFormats[w[a + 1]];
            if (f === undefined) throw new Error("replay: index format " + w[a + 1]);
            pe.setIndexBuffer(this.obj(w[a], false), f, big(w[a + 2], w[a + 3]), big(w[a + 4], w[a + 5]));
            break;
          }
          case 7: pe.setViewport(fl(w[a]), fl(w[a + 1]), fl(w[a + 2]), fl(w[a + 3]), fl(w[a + 4]), fl(w[a + 5])); break;
          case 8: pe.setScissorRect(w[a], w[a + 1], w[a + 2], w[a + 3]); break;
          case 9: pe.setBlendConstant({ r: fl(w[a]), g: fl(w[a + 1]), b: fl(w[a + 2]), a: fl(w[a + 3]) }); break;
          case 10: pe.setStencilReference(w[a]); break;
          case 11: {
            const bs = new Array(w[a]);
            for (let k = 0; k < bs.length; k++) bs[k] = this.obj(w[a + 1 + k], false);
            pe.executeBundles(bs);
            break;
          }
          case 12: pe.drawIndirect(this.obj(w[a], false), big(w[a + 1], w[a + 2])); break;
          case 13: pe.drawIndexedIndirect(this.obj(w[a], false), big(w[a + 1], w[a + 2])); break;
          case 20: {
            const nc = w[a], colors = new Array(nc);
            for (let c = 0; c < nc; c++) {
              const b = a + 1 + 8 * c;
              if (w[b] === 0) {
                for (let k = 0; k < 8; k++) if (w[b + k] !== 0) throw new Error("replay: unused color slot " + c + " is not empty");
                colors[c] = null;
                continue;
              }
              if (w[b + 2] > 2 || w[b + 3] > 2) throw new Error("replay: bad load or store op");
              const ca = {
                view: this.obj(w[b], false), loadOp: loadOps[w[b + 2]], storeOp: storeOps[w[b + 3]],
                clearValue: { r: fl(w[b + 4]), g: fl(w[b + 5]), b: fl(w[b + 6]), a: fl(w[b + 7]) },
              };
              const rt = this.obj(w[b + 1], true);
              if (rt !== null) ca.resolveTarget = rt;
              colors[c] = ca;
            }
            const desc = { colorAttachments: colors };
            if (w[a + 1 + 8 * nc] === 1) {
              const d = a + 2 + 8 * nc;
              if (w[d + 1] > 2 || w[d + 2] > 2 || w[d + 5] > 2 || w[d + 6] > 2 || w[d + 4] > 1 || w[d + 8] > 1) throw new Error("replay: bad depth-stencil ops");
              const ds = { view: this.obj(w[d], false) };
              if (w[d + 1]) ds.depthLoadOp = loadOps[w[d + 1]];
              if (w[d + 2]) ds.depthStoreOp = storeOps[w[d + 2]];
              if (w[d + 1] === 2) ds.depthClearValue = fl(w[d + 3]);
              if (w[d + 4]) ds.depthReadOnly = true;
              if (w[d + 5]) ds.stencilLoadOp = loadOps[w[d + 5]];
              if (w[d + 6]) ds.stencilStoreOp = storeOps[w[d + 6]];
              if (w[d + 7]) ds.stencilClearValue = w[d + 7];
              if (w[d + 8]) ds.stencilReadOnly = true;
              desc.depthStencilAttachment = ds;
            }
            if (enc === null) enc = device.createCommandEncoder();
            pe = enc.beginRenderPass(desc);
            pass = 1;
            break;
          }
          case 21: pe.end(); pe = null; pass = 0; break;
          case 22: if (enc === null) enc = device.createCommandEncoder(); pe = enc.beginComputePass(); pass = 2; break;
          case 23: pe.dispatchWorkgroups(w[a], w[a + 1], w[a + 2]); break;
          case 24: pe.dispatchWorkgroupsIndirect(this.obj(w[a], false), big(w[a + 1], w[a + 2])); break;
          case 30: {
            if (w[a + 3] + w[a + 4] > ndata) throw new Error("replay: data [" + w[a + 3] + ", +" + w[a + 4] + ") past " + ndata + " bytes");
            queue.writeBuffer(this.obj(w[a], false), big(w[a + 1], w[a + 2]), data, w[a + 3], w[a + 4]);
            break;
          }
          case 31: {
            if (enc === null) enc = device.createCommandEncoder();
            enc.copyBufferToBuffer(this.obj(w[a], false), big(w[a + 1], w[a + 2]), this.obj(w[a + 3], false), big(w[a + 4], w[a + 5]), big(w[a + 6], w[a + 7]));
            break;
          }
          case 40: {
            if (enc !== null) queue.submit([enc.finish()]);
            enc = null;
            break;
          }
        }
      }
      if (pass !== 0) throw new Error("replay: pass left open");
    },
  };
  globalThis.goWebGPUReplay = R;
})();
