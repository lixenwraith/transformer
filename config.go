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
// channels from the last's (one at the first), a kernel of Time frames by Freq
// features centred on each, the edges zero, GELU, then the features max-pooled
// by Pool, a remainder dropped
type Conv struct {
	Out, Time, Freq, Pool int
}

// Tower is a stack of Layers attention blocks of its own over the shared input,
// with its own final norm and output layer, computing the next Outputs heads
type Tower struct {
	Layers, Outputs int
}

// Config is a model's shape: features in, the convolutions before the input
// layer, the width, the attention heads and layers, the feed-forward width, the
// frames attended to either side, the towers (none: one of Layers computing
// every head), and the output heads.
type Config struct {
	Inputs  int
	Convs   []Conv
	Dim     int
	Heads   int
	Layers  int
	FFN     int
	Window  int
	Towers  []Tower
	Outputs []Head
}

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

// features is what the input layer takes a frame as: the inputs, or the last
// convolution's channels by the features its pooling leaves
func (c Config) features() int {
	n, ch := c.Inputs, 1
	for _, cv := range c.Convs {
		n, ch = n/max(cv.Pool, 1), cv.Out
	}
	return n * ch
}

func (c Config) validate() error {
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
	case c.Inputs < 1 || c.features() < 1 || c.Dim < 1 || c.Heads < 1 || c.Layers < 0 || c.FFN < 1 || c.Window < 0:
		return errors.New("transformer: a size is not positive")
	case c.Dim%c.Heads != 0:
		return fmt.Errorf("transformer: width %d is not a multiple of %d heads", c.Dim, c.Heads)
	case c.outputSize() < 1:
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
