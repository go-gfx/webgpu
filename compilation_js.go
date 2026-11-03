//go:build js

package webgpu

import (
	"context"
	"fmt"
)

// CompilationMessage is one message of GPUShaderModule.getCompilationInfo().
type CompilationMessage struct {
	Type    string // "error", "warning" or "info"
	Message string
	Line    int // 1-based, 0 if unknown
	Column  int // 1-based, 0 if unknown
}

func (m CompilationMessage) String() string {
	if m.Line > 0 {
		return fmt.Sprintf("%s %d:%d: %s", m.Type, m.Line, m.Column, m.Message)
	}
	return m.Type + ": " + m.Message
}

// CompilationMessages is module.getCompilationInfo(): what the browser's WGSL
// compiler said, with positions. Some browsers (Firefox) give only a generic
// validation error at createShaderModule and keep the compiler's text here.
// It waits on a promise: not from a js.FuncOf callback.
func (m *ShaderModule) CompilationMessages(ctx context.Context) ([]CompilationMessage, error) {
	var p = m.v.Call("getCompilationInfo")
	v, err := await(ctx, "getCompilationInfo", p)
	if err != nil {
		return nil, err
	}
	list := v.Get("messages")
	n := list.Length()
	out := make([]CompilationMessage, 0, n)
	for i := 0; i < n; i++ {
		x := list.Index(i)
		out = append(out, CompilationMessage{Type: str(x.Get("type")), Message: str(x.Get("message")), Line: x.Get("lineNum").Int(), Column: x.Get("linePos").Int()})
	}
	return out, nil
}
