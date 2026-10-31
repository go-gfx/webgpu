package wire

import (
	"reflect"
	"testing"
)

func TestDescriptor(t *testing.T) {
	b := AppendDescriptor(nil, []uint32{7, 0, 9}, []byte(`{"x":1}`))
	refs, body, err := SplitDescriptor(b)
	if err != nil || !reflect.DeepEqual(refs, []uint32{7, 0, 9}) || string(body) != `{"x":1}` {
		t.Fatalf("split %v %q %v", refs, body, err)
	}
	if err := RemapDescriptor(b, func(id uint32) (uint32, bool) { return id + 100, true }); err != nil {
		t.Fatal(err)
	}
	refs, _, _ = SplitDescriptor(b)
	if !reflect.DeepEqual(refs, []uint32{107, 0, 109}) {
		t.Fatalf("remapped %v", refs)
	}
	if err := RemapDescriptor(b, func(uint32) (uint32, bool) { return 0, false }); err == nil {
		t.Error("ungranted reference passed")
	}
	if err := RemapDescriptor(b, func(uint32) (uint32, bool) { return 0, true }); err == nil {
		t.Error("mapping to 0 passed")
	}
	for name, bad := range map[string][]byte{
		"short":     {1, 0},
		"too many":  AppendDescriptor(nil, make([]uint32, MaxRefs+1), nil),
		"truncated": {2, 0, 0, 0, 1, 0, 0, 0},
	} {
		if _, _, err := SplitDescriptor(bad); err == nil {
			t.Errorf("%s: split accepted", name)
		}
		if err := RemapDescriptor(bad, nil); err == nil {
			t.Errorf("%s: remap accepted", name)
		}
	}
	for k := Kind(0); k <= 11; k++ {
		if k.Valid() != (k >= 1 && k <= 10) {
			t.Errorf("kind %d valid=%v", k, k.Valid())
		}
	}
}
