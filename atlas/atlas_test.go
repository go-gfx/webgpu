package atlas

import (
	"errors"
	"testing"
)

func TestShelf(t *testing.T) {
	s := NewShelf(100, 50)
	for i, c := range []struct {
		w, h int
		want Rect
	}{
		{60, 10, Rect{0, 0, 60, 10}},
		{40, 20, Rect{60, 0, 40, 20}}, // fills the row; the shelf is 20 tall
		{30, 5, Rect{0, 20, 30, 5}},   // next shelf under the tallest
		{70, 8, Rect{30, 20, 70, 8}},
		{100, 22, Rect{0, 28, 100, 22}},
	} {
		got, err := s.Place(c.w, c.h)
		if err != nil || got != c.want {
			t.Fatalf("case %d: %v %v, want %v", i, got, err, c.want)
		}
	}
	if s.Width() != 100 || s.Height() != 50 {
		t.Fatalf("size %dx%d", s.Width(), s.Height())
	}
	for _, c := range [][2]int{{1, 1}, {0, 5}, {5, 0}, {101, 1}} {
		if _, err := s.Place(c[0], c[1]); !errors.Is(err, ErrFull) {
			t.Errorf("Place%v = %v, want ErrFull", c, err)
		}
	}
	// A refusal changes nothing: a smaller rectangle still fits elsewhere.
	s2 := NewShelf(10, 10)
	s2.Place(10, 6)
	if _, err := s2.Place(10, 6); !errors.Is(err, ErrFull) {
		t.Fatal("second shelf past the height")
	}
	if r, err := s2.Place(10, 4); err != nil || r != (Rect{0, 6, 10, 4}) {
		t.Fatalf("after a refusal: %v %v", r, err)
	}
}
