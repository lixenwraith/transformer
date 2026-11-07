package transformer

import (
	"errors"
	"fmt"
)

// Head is one named slice of the output vector
type Head struct {
	Name string
	Size int
}

// Conv is one convolution over the frames as an image, frames by features: Out
// channels from the last's (the Config's at the first), a kernel of Time frames
// by Freq features centred on each, the edges zero, GELU, then the features
// max-pooled by Pool, a remainder dropped
type Conv struct {
	Out, Time, Freq, Pool int
}

// Tower is a stack of Layers attention blocks of its own over the shared input,
// with its own final norm and output layer, computing the next Outputs heads
type Tower struct {
	Layers, Outputs int
}

// Config is a model's shape: features in, the channels of them a frame lays end
// to end (none: one), the convolutions before the input layer, the width, the
// attention heads and layers, the feed-forward width, the frames attended to
// either side, the towers (none: one of Layers computing every head), and the
// output heads. Cells, when given, is the model in place of all between.
type Config struct {
	Inputs   int
	Channels int
	Convs    []Conv
	Dim      int
	Heads    int
	Layers   int
	FFN      int
	Window   int
	Towers   []Tower
	Outputs  []Head
	Cells    *Cells
}

// Cells is a convolutional net over the frames as an image, the same weights at
// every cell along the features: the first Heads outputs are per cell, each Read
// cells times its channels, read from around its cell, and the rest per frame,
// from the features pooled to regions and the Scalars. doc/format.md states each
// step; every count but Depth is one or more, every kernel odd.
type Cells struct {
	Bins                      int // features of each channel the convolutions read, from its first
	Reduce                    int // channels of the 1×1 convolution over the input's
	Wide, WideTime, WideFreq  int // channels and kernel of the convolution over the bins
	Cell, CellTime, CellFreq  int // of the convolution to cells, cell j centred on bin Stride·j
	Stride                    int
	Depth, NearTime, NearFreq int   // convolutions over the cells in turn, each added back, and their kernel
	Long                      []int // then one of 3 frames by NearFreq cells for each, its frames this far apart, each added back
	First, Read               int   // the first cell the cell heads read, and how many
	Heads                     int   // how many of Outputs, from the first, are per cell
	Regions                   int   // runs of bins each wide channel averages to for the frame path
	Scalars                   []int // inputs of a frame, across its channels, the frame path reads as they are
	Frame, FrameTime          int   // channels and frames of the frame path's convolution,
	AgainTime, Dilation       int   // then its second's frames, Dilation apart, added back
}

// cells is how many cells the convolution to cells makes of the bins
func (c *Cells) cells() int { return (c.Bins+c.CellFreq/2-c.CellFreq)/c.Stride + 1 }

// pooled is what the frame path's first convolution reads a frame as
func (c *Cells) pooled() int { return c.Wide*c.Regions + len(c.Scalars) }

// channels is the cell heads' channels at each cell, their sizes over Read
func (c *Cells) channels(heads []Head) int {
	n := 0
	for _, h := range heads[:c.Heads] {
		n += h.Size / c.Read
	}
	return n
}

// validateCells checks the sizes of a cell net
func (c Config) validateCells() error {
	n := c.Cells
	odd := func(k ...int) bool {
		for _, v := range k {
			if v < 1 || v%2 == 0 {
				return false
			}
		}
		return true
	}
	switch {
	case len(c.Convs) > 0 || c.Dim != 0 || c.Heads != 0 || c.Layers != 0 || c.FFN != 0 || c.Window != 0 || len(c.Towers) > 0:
		return errors.New("transformer: a cell net takes no transformer sizes")
	case n.Bins < 1 || n.Bins > c.Inputs || n.Reduce < 1 || n.Wide < 1 || n.Cell < 1 || n.Stride < 1 || n.Depth < 0 || n.Read < 1 || n.Regions < 1 || n.Frame < 1 || n.Dilation < 1:
		return fmt.Errorf("transformer: cell net %+v: a count under one, or more bins than inputs", *n)
	case !odd(n.WideTime, n.WideFreq, n.CellTime, n.CellFreq, n.NearTime, n.NearFreq, n.FrameTime, n.AgainTime):
		return errors.New("transformer: a cell net's kernel is not odd")
	case n.Bins <= n.CellFreq/2:
		return fmt.Errorf("transformer: %d bins make no cell of %d", n.Bins, n.CellFreq)
	case n.Heads < 1 || n.Heads >= len(c.Outputs):
		return fmt.Errorf("transformer: %d cell heads of %d, want a frame head too", n.Heads, len(c.Outputs))
	}
	if n.First < 0 || n.First+n.Read > n.cells() {
		return fmt.Errorf("transformer: the cell heads read %d cells from %d of %d", n.Read, n.First, n.cells())
	}
	for _, h := range c.Outputs[:n.Heads] {
		if h.Size < 1 || h.Size%n.Read != 0 {
			return fmt.Errorf("transformer: cell head %q of %d is no multiple of %d cells", h.Name, h.Size, n.Read)
		}
	}
	for _, d := range n.Long {
		if d < 1 {
			return fmt.Errorf("transformer: frames %d apart in a long convolution", d)
		}
	}
	for _, s := range n.Scalars {
		if s < 0 || s >= c.Inputs*c.channels() {
			return fmt.Errorf("transformer: scalar %d past a frame's %d inputs", s, c.Inputs*c.channels())
		}
	}
	return nil
}

func (c Config) channels() int { return max(c.Channels, 1) }

// towers is the Towers, or the one tower of Layers a Config without them has
func (c Config) towers() []Tower {
	if len(c.Towers) > 0 {
		return c.Towers
	}
	return []Tower{{Layers: c.Layers, Outputs: len(c.Outputs)}}
}

// outputSize is the output vector's length, the heads laid end to end
func (c Config) outputSize() int {
	n := 0
	for _, h := range c.Outputs {
		n += h.Size
	}
	return n
}

// features is what the input layer takes a frame as: the inputs by the channels,
// or the last convolution's channels by the features its pooling leaves
func (c Config) features() int {
	n, ch := c.Inputs, c.channels()
	for _, cv := range c.Convs {
		n, ch = n/max(cv.Pool, 1), cv.Out
	}
	return n * ch
}

// kind is the row of kinds a Config's model is
func (c Config) kind() int {
	if c.Cells != nil {
		return kindCells
	}
	return kindTransformer
}

func (c Config) validate() error {
	if c.Inputs < 1 || c.Channels < 0 {
		return errors.New("transformer: a size is not positive")
	}
	if err := kinds[c.kind()].validate(c); err != nil {
		return err
	}
	if c.outputSize() < 1 {
		return errors.New("transformer: no output head")
	}
	seen := map[string]bool{}
	for _, h := range c.Outputs {
		if h.Name == "" || h.Size < 1 || seen[h.Name] {
			return fmt.Errorf("transformer: output head %q of %d", h.Name, h.Size)
		}
		seen[h.Name] = true
	}
	return nil
}

// validateTransformer checks the sizes of a transformer
func (c Config) validateTransformer() error {
	for i, cv := range c.Convs {
		if cv.Out < 1 || cv.Time < 1 || cv.Time%2 == 0 || cv.Freq < 1 || cv.Freq%2 == 0 || cv.Pool < 1 {
			return fmt.Errorf("transformer: convolution %d is %+v, want channels, odd kernels and a pool of one or more", i, cv)
		}
	}
	heads := 0
	for _, t := range c.Towers {
		if t.Layers < 0 || t.Outputs < 1 {
			return fmt.Errorf("transformer: tower %+v", t)
		}
		heads += t.Outputs
	}
	switch {
	case len(c.Towers) > 0 && heads != len(c.Outputs):
		return fmt.Errorf("transformer: the towers compute %d heads of %d", heads, len(c.Outputs))
	case c.features() < 1 || c.Dim < 1 || c.Heads < 1 || c.Layers < 0 || c.FFN < 1 || c.Window < 0:
		return errors.New("transformer: a size is not positive")
	case c.Dim%c.Heads != 0:
		return fmt.Errorf("transformer: width %d is not a multiple of %d heads", c.Dim, c.Heads)
	}
	return nil
}
