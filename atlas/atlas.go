// Package atlas packs rectangles into a texture atlas: the placement
// half of "many small images in one texture" (textures, lightmaps, glyphs,
// sprites), which a WebGPU program does to bind one texture instead of
// hundreds. It has no GOOS constraint and no dependency; copying the
// pixels in (and into the GPU texture) is the caller's.
package atlas

import "errors"

// ErrFull is returned by [Shelf.Place] when the rectangle does not fit.
var ErrFull = errors.New("atlas: full")

// Rect is a placed region of the atlas, in texels.
type Rect struct{ X, Y, W, H uint32 }

// Shelf is a shelf packer: rectangles go left to right in rows (shelves),
// each as tall as its tallest rectangle; a rectangle that does not fit the
// row starts the next one. It is simple, fast and tight enough for images
// of similar height.
type Shelf struct {
	width, maxHeight int
	x, y, rowH       int
	height           int
}

// NewShelf returns a packer for an atlas width texels wide that may grow
// to maxHeight texels (the device's maxTextureDimension2D bounds it).
func NewShelf(width, maxHeight int) *Shelf { return &Shelf{width: width, maxHeight: maxHeight} }

// Place reserves a w x h rectangle. It returns [ErrFull] for an empty or
// too wide rectangle and when the atlas would pass its maximum height.
func (s *Shelf) Place(w, h int) (Rect, error) {
	if w <= 0 || h <= 0 || w > s.width {
		return Rect{}, ErrFull
	}
	x, y, rowH := s.x, s.y, s.rowH
	if x+w > s.width {
		y, x, rowH = y+rowH, 0, 0
	}
	if y+h > s.maxHeight {
		return Rect{}, ErrFull
	}
	s.x, s.y, s.rowH = x+w, y, max(rowH, h)
	s.height = max(s.height, y+s.rowH)
	return Rect{uint32(x), uint32(y), uint32(w), uint32(h)}, nil
}

// Width is the atlas width.
func (s *Shelf) Width() int { return s.width }

// Height is the height used so far: the texture to create is Width x Height.
func (s *Shelf) Height() int { return s.height }
