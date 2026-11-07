// Package transformer runs small models over sequences of frames in Go, for
// inference, in float32: transformers, convolutions over the frames as an image,
// pre-norm attention blocks with ALiBi biases and exact-GELU feed-forwards, and
// cell nets, convolutions alone; each into named output heads. Load reads the
// versioned weight file doc/format.md defines, from any trainer that writes it.
package transformer

import (
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
	"sync/atomic"
)

// linear is y = W x + b, W held as Out rows of In
type linear struct {
	In, Out int
	W, B    []float32
}

func newLinear(in, out int) linear {
	return linear{In: in, Out: out, W: make([]float32, out*in), B: make([]float32, out)}
}

// apply writes x (frames × In) through the layer into y (frames × Out), a
// weight row at a time for a block of frames
func (l *linear) apply(x []float32, frames int, y []float32) {
	across(frames, frameBlock, func(lo, hi int) {
		for o := range l.Out {
			w := l.W[o*l.In : (o+1)*l.In]
			for t := lo; t < hi; t++ {
				y[t*l.Out+o] = dot(l.B[o], w, x[t*l.In:(t+1)*l.In])
			}
		}
	})
}

// dotGeneric is s plus a·b summed in order, the reference the vector kernels
// match to float32 rounding and the dot of a target without them
func dotGeneric(s float32, a, b []float32) float32 {
	b = b[:len(a)]
	for i, v := range a {
		s += v * b[i]
	}
	return s
}

// axpyGeneric adds w·x into y, as axpy does on a target without vector kernels
func axpyGeneric(w float32, x, y []float32) {
	y = y[:len(x)]
	for i, v := range x {
		y[i] += w * v
	}
}

// frameBlock is the frames a transformer's layer takes at once: few enough that a
// slow core's last block barely delays the rest, enough that a weight row read
// once serves them
const frameBlock = 8

// across hands 0 to n-1 in blocks to up to GOMAXPROCS goroutines, each taking
// the next block as it finishes one, and to none under two blocks. f must write
// only its block's outputs, so the bits never depend on the split.
func across(n, block int, f func(lo, hi int)) {
	procs := min(runtime.GOMAXPROCS(0), (n+block-1)/block)
	if procs < 2 {
		for lo := 0; lo < n; lo += block {
			f(lo, min(lo+block, n))
		}
		return
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	for range procs {
		wg.Go(func() {
			for {
				lo := int(next.Add(int64(block))) - block
				if lo >= n {
					return
				}
				f(lo, min(lo+block, n))
			}
		})
	}
	wg.Wait()
}

// layerNorm normalizes each frame over the width, then scales and shifts it
type layerNorm struct {
	G, B []float32
}

func newLayerNorm(dim int) layerNorm {
	ln := layerNorm{G: make([]float32, dim), B: make([]float32, dim)}
	for i := range ln.G {
		ln.G[i] = 1
	}
	return ln
}

const layerNormEps = 1e-5

func (ln *layerNorm) apply(x []float32, frames int, y []float32) {
	dim := len(ln.G)
	for t := range frames {
		xt, yt := x[t*dim:(t+1)*dim], y[t*dim:(t+1)*dim]
		var mean float64
		for _, v := range xt {
			mean += float64(v)
		}
		mean /= float64(dim)
		var variance float64
		for _, v := range xt {
			d := float64(v) - mean
			variance += d * d
		}
		inv := 1 / math.Sqrt(variance/float64(dim)+layerNormEps)
		for i, v := range xt {
			yt[i] = float32((float64(v)-mean)*inv)*ln.G[i] + ln.B[i]
		}
	}
}

// gelu is the exact Gaussian error linear unit, not its tanh approximation
func gelu(v float32) float32 {
	x := float64(v)
	return float32(0.5 * x * (1 + math.Erf(x/math.Sqrt2)))
}

type layer struct {
	ln1, ln2   layerNorm
	q, k, v, o linear
	ff1, ff2   linear
}

// conv is a Conv's weights, out by in channels by Time by Freq, and its biases
type conv struct {
	Conv
	in   int
	W, B []float32
}

// apply convolves x, frames by in channels by f features, into frames by Out by
// f/Pool, the GELU taken before the pool
func (c *conv) apply(x []float32, frames, f int) []float32 {
	y := make([]float32, frames*c.Out*f)
	across(frames, frameBlock, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			for o := range c.Out {
				yo := y[(t*c.Out+o)*f : (t*c.Out+o+1)*f]
				for k := range yo {
					yo[k] = c.B[o]
				}
				for i := range c.in {
					for dt := range c.Time {
						tt := t + dt - c.Time/2
						if tt < 0 || tt >= frames {
							continue
						}
						xi := x[(tt*c.in+i)*f : (tt*c.in+i+1)*f]
						for dk := range c.Freq {
							w, d := c.W[((o*c.in+i)*c.Time+dt)*c.Freq+dk], dk-c.Freq/2
							if from, to := max(-d, 0), min(f, f-d); from < to {
								axpy(w, xi[from+d:to+d], yo[from:to])
							}
						}
					}
				}
				for k := range yo {
					yo[k] = gelu(yo[k])
				}
			}
		}
	})
	if c.Pool == 1 {
		return y
	}
	g := f / c.Pool
	z := make([]float32, frames*c.Out*g)
	for r := range frames * c.Out {
		for k := range g {
			m := y[r*f+k*c.Pool]
			for j := 1; j < c.Pool; j++ {
				m = max(m, y[r*f+k*c.Pool+j])
			}
			z[r*g+k] = m
		}
	}
	return z
}

// tower is a Tower's blocks, final norm and output layer
type tower struct {
	layers []layer
	ln     layerNorm
	out    linear
}

// Model is one model with its weights
type Model struct {
	cfg Config
	net net
}

// net is a model kind's layers
type net interface {
	tensors() []tensor                       // every weight, in the order the file holds them
	randomize(rng *rand.Rand)                // small random weights, scaled to each layer's inputs
	forward(frames [][]float32, y []float32) // y: a frame's outputs after another's
}

const (
	kindTransformer = iota
	kindCells
)

// kinds is what each kind of model brings, indexed by Config.kind: its checks,
// its layers of zero weights, and its header in the file after the channels
var kinds = [...]struct {
	validate func(Config) error
	build    func(Config) net
	save     func(put func(...int), cfg Config)
	load     func(get func() (int, error), version int, cfg *Config) error
}{
	kindTransformer: {Config.validateTransformer, newTransformer, saveTransformer, loadTransformer},
	kindCells:       {Config.validateCells, newCellNet, saveCells, loadCells},
}

// newModel makes a model of zero weights, the shape Load fills
func newModel(cfg Config) (*Model, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return &Model{cfg: cfg, net: kinds[cfg.kind()].build(cfg)}, nil
}

// transformerNet is a transformer's layers
type transformerNet struct {
	cfg    Config
	convs  []conv
	in     linear
	towers []tower
	slopes []float32 // one relative-position slope per head
}

func newTransformer(cfg Config) net {
	m := &transformerNet{cfg: cfg, in: newLinear(cfg.features(), cfg.Dim)}
	ch := cfg.channels()
	for _, cv := range cfg.Convs {
		m.convs = append(m.convs, conv{Conv: cv, in: ch, W: make([]float32, cv.Out*ch*cv.Time*cv.Freq), B: make([]float32, cv.Out)})
		ch = cv.Out
	}
	head := 0
	for _, t := range cfg.towers() {
		size := 0
		for _, h := range cfg.Outputs[head : head+t.Outputs] {
			size += h.Size
		}
		head += t.Outputs
		tw := tower{ln: newLayerNorm(cfg.Dim), out: newLinear(cfg.Dim, size)}
		for range t.Layers {
			tw.layers = append(tw.layers, layer{
				ln1: newLayerNorm(cfg.Dim), ln2: newLayerNorm(cfg.Dim),
				q: newLinear(cfg.Dim, cfg.Dim), k: newLinear(cfg.Dim, cfg.Dim), v: newLinear(cfg.Dim, cfg.Dim), o: newLinear(cfg.Dim, cfg.Dim),
				ff1: newLinear(cfg.Dim, cfg.FFN), ff2: newLinear(cfg.FFN, cfg.Dim),
			})
		}
		m.towers = append(m.towers, tw)
	}
	// ALiBi: head h leans on nearer frames by 2^(-8h/H) per frame of distance
	m.slopes = make([]float32, cfg.Heads)
	for h := range m.slopes {
		m.slopes[h] = float32(math.Pow(2, -8*float64(h+1)/float64(cfg.Heads)))
	}
	return m
}

// NewRandom makes a model of small random weights under seed, for tests and
// for a trainer's shape check.
func NewRandom(cfg Config, seed uint64) (*Model, error) {
	m, err := newModel(cfg)
	if err != nil {
		return nil, err
	}
	m.net.randomize(rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15)))
	return m, nil
}

func (m *transformerNet) randomize(rng *rand.Rand) {
	for _, l := range m.linears() {
		scale := float32(1 / math.Sqrt(float64(l.In)))
		for i := range l.W {
			l.W[i] = float32(rng.NormFloat64()) * scale
		}
	}
	for _, c := range m.convs {
		scale := float32(1 / math.Sqrt(float64(c.in*c.Time*c.Freq)))
		for i := range c.W {
			c.W[i] = float32(rng.NormFloat64()) * scale
		}
	}
}

func (m *Model) Config() Config { return m.cfg }

// Parameters counts the weights
func (m *Model) Parameters() int {
	n := 0
	for _, t := range m.net.tensors() {
		n += len(t.data)
	}
	return n
}

func (m *transformerNet) linears() []*linear {
	ls := []*linear{&m.in}
	for ti := range m.towers {
		tw := &m.towers[ti]
		for i := range tw.layers {
			l := &tw.layers[i]
			ls = append(ls, &l.q, &l.k, &l.v, &l.o, &l.ff1, &l.ff2)
		}
		ls = append(ls, &tw.out)
	}
	return ls
}

// Forward runs frames, each its channels of Inputs features end to end, through
// the model and returns one output vector per frame, the heads laid end to end as
// Config.Outputs.
func (m *Model) Forward(frames [][]float32) [][]float32 {
	T, size := len(frames), m.cfg.outputSize()
	if T == 0 {
		return nil
	}
	y := make([]float32, T*size)
	m.net.forward(frames, y)
	out := make([][]float32, T)
	for t := range out {
		out[t] = y[t*size : (t+1)*size]
	}
	return out
}

func (m *transformerNet) forward(frames [][]float32, y []float32) {
	T, cfg := len(frames), m.cfg
	per := cfg.Inputs * cfg.channels()
	x := make([]float32, T*per)
	for t, f := range frames {
		copy(x[t*per:], f)
	}
	for i, f := 0, cfg.Inputs; i < len(m.convs); i, f = i+1, f/m.convs[i].Pool {
		x = m.convs[i].apply(x, T, f)
	}
	in := make([]float32, T*cfg.Dim)
	m.in.apply(x, T, in)
	h, n := make([]float32, T*cfg.Dim), make([]float32, T*cfg.Dim)
	q, k, v, att := make([]float32, T*cfg.Dim), make([]float32, T*cfg.Dim), make([]float32, T*cfg.Dim), make([]float32, T*cfg.Dim)
	o := make([]float32, T*cfg.Dim)
	ff := make([]float32, T*cfg.FFN)
	size := cfg.outputSize()
	at := 0 // where this tower's heads begin in a frame's output
	for ti := range m.towers {
		tw := &m.towers[ti]
		copy(h, in)
		for i := range tw.layers {
			l := &tw.layers[i]
			l.ln1.apply(h, T, n)
			l.q.apply(n, T, q)
			l.k.apply(n, T, k)
			l.v.apply(n, T, v)
			m.attend(q, k, v, T, att)
			l.o.apply(att, T, o)
			for j := range h {
				h[j] += o[j]
			}
			l.ln2.apply(h, T, n)
			l.ff1.apply(n, T, ff)
			across(T, frameBlock, func(lo, hi int) {
				for j := lo * cfg.FFN; j < hi*cfg.FFN; j++ {
					ff[j] = gelu(ff[j])
				}
			})
			l.ff2.apply(ff, T, o)
			for j := range h {
				h[j] += o[j]
			}
		}
		tw.ln.apply(h, T, n)
		yt := make([]float32, T*tw.out.Out)
		tw.out.apply(n, T, yt)
		for t := range T {
			copy(y[t*size+at:], yt[t*tw.out.Out:(t+1)*tw.out.Out])
		}
		at += tw.out.Out
	}
}

// attend is multi-head attention over the window: each frame weighs the frames
// within Window of it by their keys' match to its query, less the head's slope
// per frame of distance, and sums their values.
func (m *transformerNet) attend(q, k, v []float32, T int, out []float32) {
	cfg := m.cfg
	dh := cfg.Dim / cfg.Heads
	scale := float32(1 / math.Sqrt(float64(dh)))
	across(T, frameBlock, func(from, to int) {
		scores := make([]float32, 2*cfg.Window+1)
		for i := from; i < to; i++ {
			lo, hi := max(i-cfg.Window, 0), min(i+cfg.Window, T-1)
			for h := range cfg.Heads {
				off := h * dh
				qi := q[i*cfg.Dim+off : i*cfg.Dim+off+dh]
				var peak float32 = -math.MaxFloat32
				for j := lo; j <= hi; j++ {
					s := dot(0, qi, k[j*cfg.Dim+off:j*cfg.Dim+off+dh])
					d := i - j
					if d < 0 {
						d = -d
					}
					s = s*scale - m.slopes[h]*float32(d)
					scores[j-lo] = s
					peak = max(peak, s)
				}
				var sum float32
				for j := lo; j <= hi; j++ {
					e := float32(math.Exp(float64(scores[j-lo] - peak)))
					scores[j-lo] = e
					sum += e
				}
				oi := out[i*cfg.Dim+off : i*cfg.Dim+off+dh]
				clear(oi)
				for j := lo; j <= hi; j++ {
					axpy(scores[j-lo]/sum, v[j*cfg.Dim+off:j*cfg.Dim+off+dh], oi)
				}
			}
		}
	})
}

// Slice is the part of a Forward output that one head wrote, nil for a head
// the model has not.
func (m *Model) Slice(out []float32, head string) []float32 {
	at := 0
	for _, h := range m.cfg.Outputs {
		if h.Name == head {
			return out[at : at+h.Size]
		}
		at += h.Size
	}
	return nil
}
