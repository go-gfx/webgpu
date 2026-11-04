package argmax

import (
	"strings"
	"testing"

	"github.com/go-gfx/wgsl"
)

func TestSourcesParse(t *testing.T) {
	for _, wg := range []int{64, 32, 16} {
		for _, vocab := range []int{49152, 151936, 100} {
			a, b := Sources(wg, vocab)
			for i, src := range []string{a, b} {
				if strings.Contains(src, "{{") {
					t.Fatalf("stage %d: unsubstituted placeholder", i+1)
				}
				if _, err := wgsl.Parse(src); err != nil {
					t.Errorf("wg %d vocab %d stage %d: %v", wg, vocab, i+1, err)
				}
			}
			if got := strings.Count(a, "@binding("); got != len(Access1) {
				t.Errorf("stage 1: %d bindings", got)
			}
			if got := strings.Count(b, "@binding("); got != len(Access2) {
				t.Errorf("stage 2: %d bindings", got)
			}
		}
	}
}
