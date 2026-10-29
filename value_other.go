//go:build !js

package webgpu

// jsValue is empty outside the browser: the descriptor and validation code
// compiles and is tested everywhere, the calls exist only under GOOS=js.
type jsValue struct{}

type staging struct{}

// ref stands for the browser object in a descriptor; outside the browser
// it is the handle itself, which lets tests check what would be sent.
func (o *object) ref() any { return o }
