package webgpu

import _ "embed"

// ReplayJS is replay.js, the JavaScript half of batched encoding
// (Recorder). A page with a strict Content-Security-Policy (script-src
// 'self') must serve it from its own origin, as it serves wasm_exec.js,
// and pass its URL to InstallReplay; ReplayJSName is the conventional
// file name.
//
//go:embed replay.js
var ReplayJS string

// ReplayJSName is the file name ReplayJS is conventionally served under.
const ReplayJSName = "webgpu-replay.js"
