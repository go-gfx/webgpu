// Command wasmbrowser runs a GOOS=js test binary (or any js/wasm program)
// in a dedicated worker of a real browser, through Playwright, and exits
// with the program's exit code. It is meant for go test's -exec flag:
//
//	GOOS=js GOARCH=wasm go test -exec="go -C internal/cmd/wasmbrowser run ." ./...
//
// The worker is where WebGPU programs of this module run in production
// (an engine worker, a game client worker). Standard output and error are
// forwarded as they are written. Environment variables named WEBGPU_* and
// GODEBUG reach the program; WEBGPU_BROWSER_CHANNEL picks the Chromium
// channel ("chromium", the default, is the full browser in new headless
// mode: the headless shell offers no GPU adapter), WEBGPU_BROWSER_ARGS adds
// browser arguments, WEBGPU_BROWSER_TIMEOUT bounds the run (default 20m).
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/playwright-community/playwright-go"
	"github.com/spf13/cobra"
)

var osExit = os.Exit

func main() {
	code, err := run(os.Args[1:], os.Stdout, os.Stderr)
	if err != nil {
		fmt.Fprintln(os.Stderr, "wasmbrowser:", err)
		if code == 0 {
			code = 2
		}
	}
	osExit(code)
}

// run parses the command line and runs the program; it returns the
// program's exit code.
func run(args []string, stdout, stderr io.Writer) (int, error) {
	code := 0
	cmd := RootCmd(&code)
	cmd.SetArgs(args)
	cmd.SetOut(stdout)
	cmd.SetErr(stderr)
	err := cmd.Execute()
	return code, err
}

// RootCmd is the command; the program's exit code lands in *code.
func RootCmd(code *int) *cobra.Command {
	cmd := &cobra.Command{
		Use:          "wasmbrowser program.wasm [program arguments]",
		Short:        "Run a js/wasm program in a browser worker",
		SilenceUsage: true,
		Args:         cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			c, err := runInBrowser(args[0], args[1:], os.Environ(), cmd.OutOrStdout(), cmd.ErrOrStderr())
			*code = c
			return err
		},
	}
	// Everything after the program is the program's.
	cmd.Flags().SetInterspersed(false)
	return cmd
}

const page = `<!doctype html><meta charset=utf-8><title>wasmbrowser</title><script>
const w = new Worker("worker.js");
w.onmessage = (e) => { if ("exit" in e.data) window.__wbExit(e.data.exit); else window.__wbOut(e.data.fd, e.data.text); };
w.onerror = (e) => { window.__wbOut(2, "worker error: " + e.message + "\n"); window.__wbExit(2); };
</script>`

// workerJS defines the fs stub wasm_exec.js would otherwise define, with
// writes posted to the page, then runs the program.
const workerJS = `"use strict";
const dec = new TextDecoder("utf-8");
const enosys = () => { const e = new Error("not implemented"); e.code = "ENOSYS"; return e; };
const nosys = (...a) => a[a.length - 1](enosys());
globalThis.fs = {
  constants: { O_WRONLY: -1, O_RDWR: -1, O_CREAT: -1, O_TRUNC: -1, O_APPEND: -1, O_EXCL: -1, O_DIRECTORY: -1 },
  writeSync(fd, buf) { postMessage({ fd, text: dec.decode(buf) }); return buf.length; },
  write(fd, buf, offset, length, position, callback) {
    if (offset !== 0 || length !== buf.length || position !== null) { callback(enosys()); return; }
    callback(null, this.writeSync(fd, buf));
  },
  chmod: nosys, chown: nosys, close: nosys, fchmod: nosys, fchown: nosys, fstat: nosys,
  fsync(fd, callback) { callback(null); }, ftruncate: nosys, lchown: nosys, link: nosys, lstat: nosys,
  mkdir: nosys, open: nosys, read: nosys, readdir: nosys, readlink: nosys, rename: nosys, rmdir: nosys,
  stat: nosys, symlink: nosys, truncate: nosys, unlink: nosys, utimes: nosys,
};
importScripts("wasm_exec.js");
const go = new Go();
globalThis.goRunner = go; // lets a benchmark reach the instance's memory
go.argv = ARGV;
go.env = ENV;
go.exit = (code) => postMessage({ exit: code });
WebAssembly.instantiateStreaming(fetch("program.wasm"), go.importObject)
  .then((r) => go.run(r.instance))
  .catch((e) => { postMessage({ fd: 2, text: "wasmbrowser: " + e + "\n" }); postMessage({ exit: 2 }); });
`

// toInt reads a number Playwright decoded from JSON.
func toInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	}
	return 2
}

// programEnv keeps the variables the program may see.
func programEnv(environ []string) map[string]string {
	env := map[string]string{}
	for _, kv := range environ {
		k, v, ok := strings.Cut(kv, "=")
		if ok && (strings.HasPrefix(k, "WEBGPU_") || k == "GODEBUG") {
			env[k] = v
		}
	}
	return env
}

func getenv(environ []string, key, def string) string {
	for _, kv := range environ {
		if k, v, ok := strings.Cut(kv, "="); ok && k == key {
			return v
		}
	}
	return def
}

// site builds the files the browser loads.
func site(program string, args []string, environ []string, wasmExec []byte) (map[string][]byte, error) {
	bin, err := os.ReadFile(program)
	if err != nil {
		return nil, err
	}
	argv, err := json.Marshal(append([]string{filepath.Base(program)}, args...))
	if err != nil {
		return nil, err
	}
	env, err := json.Marshal(programEnv(environ))
	if err != nil {
		return nil, err
	}
	w := strings.Replace(workerJS, "ARGV", string(argv), 1)
	w = strings.Replace(w, "ENV", string(env), 1)
	return map[string][]byte{
		"index.html": []byte(page), "worker.js": []byte(w), "wasm_exec.js": wasmExec, "program.wasm": bin,
	}, nil
}

// handler serves the site, cross-origin isolated (fine-grained timers,
// SharedArrayBuffer), with the right types.
func handler(files map[string][]byte) http.Handler {
	types := map[string]string{".html": "text/html; charset=utf-8", ".js": "text/javascript", ".wasm": "application/wasm"}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(r.URL.Path, "/")
		if name == "" {
			name = "index.html"
		}
		b, ok := files[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", types[filepath.Ext(name)])
		w.Header().Set("Cross-Origin-Opener-Policy", "same-origin")
		w.Header().Set("Cross-Origin-Embedder-Policy", "require-corp")
		w.Write(b)
	})
}

// wasmExecJS reads wasm_exec.js from the GOROOT of the go command the
// test runs under, so it matches the toolchain that built the program.
func wasmExecJS() ([]byte, error) {
	out, err := exec.Command("go", "env", "GOROOT").Output()
	if err != nil {
		return nil, fmt.Errorf("go env GOROOT: %w", err)
	}
	return os.ReadFile(filepath.Join(strings.TrimSpace(string(out)), "lib", "wasm", "wasm_exec.js"))
}

func runInBrowser(program string, args, environ []string, stdout, stderr io.Writer) (int, error) {
	we, err := wasmExecJS()
	if err != nil {
		return 2, err
	}
	files, err := site(program, args, environ, we)
	if err != nil {
		return 2, err
	}
	timeout, err := time.ParseDuration(getenv(environ, "WEBGPU_BROWSER_TIMEOUT", "20m"))
	if err != nil {
		return 2, err
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 2, err
	}
	srv := &http.Server{Handler: handler(files)}
	go srv.Serve(ln)
	defer srv.Close()

	pw, err := playwright.Run()
	if err != nil {
		return 2, err
	}
	defer pw.Stop()
	opts := playwright.BrowserTypeLaunchOptions{Headless: playwright.Bool(true)}
	if ch := getenv(environ, "WEBGPU_BROWSER_CHANNEL", "chromium"); ch != "" {
		opts.Channel = playwright.String(ch)
	}
	if a := getenv(environ, "WEBGPU_BROWSER_ARGS", ""); a != "" {
		opts.Args = strings.Fields(a)
	}
	b, err := pw.Chromium.Launch(opts)
	if err != nil {
		return 2, err
	}
	defer b.Close()
	p, err := b.NewPage()
	if err != nil {
		return 2, err
	}
	exit := make(chan int, 1)
	if err := p.ExposeFunction("__wbOut", func(a ...any) any {
		fd := toInt(a[0])
		text, _ := a[1].(string)
		if fd == 2 {
			io.WriteString(stderr, text)
		} else {
			io.WriteString(stdout, text)
		}
		return nil
	}); err != nil {
		return 2, err
	}
	if err := p.ExposeFunction("__wbExit", func(a ...any) any {
		c := toInt(a[0])
		select {
		case exit <- c:
		default:
		}
		return nil
	}); err != nil {
		return 2, err
	}
	// localhost is a secure context, which WebGPU requires.
	if _, err := p.Goto(fmt.Sprintf("http://localhost:%d/", ln.Addr().(*net.TCPAddr).Port)); err != nil {
		return 2, err
	}
	select {
	case c := <-exit:
		return c, nil
	case <-time.After(timeout):
		return 2, errors.New("timed out after " + timeout.String())
	}
}
