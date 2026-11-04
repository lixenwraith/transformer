// Package transformer runs small transformer models over sequences of frames in
// pure Go, for inference: convolutions over the frames as an image, a linear
// input, pre-norm attention blocks with ALiBi biases over a window of frames,
// exact-GELU feed-forwards and named output heads, in float32. Load reads the
// versioned weight file doc/format.md defines, from any trainer that writes it.
package transformer

import (
	"math"
	"math/rand/v2"
)

// linear is y = W x + b, W held as Out rows of In
type linear struct {
	In, Out int
	W, B    []float32
}

func newLinear(in, out int) linear {
	return linear{In: in, Out: out, W: make([]float32, out*in), B: make([]float32, out)}
}

// apply writes x (frames × In) through the layer into y (frames × Out)
func (l *linear) apply(x []float32, frames int, y []float32) {
	for t := range frames {
		xt, yt := x[t*l.In:(t+1)*l.In], y[t*l.Out:(t+1)*l.Out]
		for o := range l.Out {
			w := l.W[o*l.In : (o+1)*l.In]
			s := l.B[o]
			for i, v := range xt {
				s += w[i] * v
			}
			yt[o] = s
		}
	}
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
	for t := range frames {
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
						for k := max(-d, 0); k < min(f, f-d); k++ {
							yo[k] += w * xi[k+d]
						}
					}
				}
			}
			for k := range yo {
				yo[k] = gelu(yo[k])
			}
		}
	}
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

// Model is one transformer with its weights
type Model struct {
	cfg    Config
	convs  []conv
	in     linear
	towers []tower
	slopes []float32 // one relative-position slope per head
}

// newModel makes a model of zero weights, the shape Load fills
func newModel(cfg Config) (*Model, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	m := &Model{cfg: cfg, in: newLinear(cfg.features(), cfg.Dim)}
	ch := 1
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
	return m, nil
}

// NewRandom makes a model of small random weights under seed, for tests and
// for a trainer's shape check.
func NewRandom(cfg Config, seed uint64) (*Model, error) {
	m, err := newModel(cfg)
	if err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
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
	return m, nil
}

func (m *Model) Config() Config { return m.cfg }

// Parameters counts the weights
func (m *Model) Parameters() int {
	n := 0
	for _, t := range m.tensors() {
		n += len(t.data)
	}
	return n
}

func (m *Model) linears() []*linear {
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

// Forward runs frames, each of Inputs features, through the model and returns
// one output vector per frame, the heads laid end to end as Config.Outputs.
func (m *Model) Forward(frames [][]float32) [][]float32 {
	T, cfg := len(frames), m.cfg
	if T == 0 {
		return nil
	}
	x := make([]float32, T*cfg.Inputs)
	for t, f := range frames {
		copy(x[t*cfg.Inputs:], f)
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
	y := make([]float32, T*size)
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
			for j := range ff {
				ff[j] = gelu(ff[j])
			}
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
	out := make([][]float32, T)
	for t := range out {
		out[t] = y[t*size : (t+1)*size]
	}
	return out
}

// attend is multi-head attention over the window: each frame weighs the frames
// within Window of it by their keys' match to its query, less the head's slope
// per frame of distance, and sums their values.
func (m *Model) attend(q, k, v []float32, T int, out []float32) {
	cfg := m.cfg
	dh := cfg.Dim / cfg.Heads
	scale := float32(1 / math.Sqrt(float64(dh)))
	scores := make([]float32, 2*cfg.Window+1)
	for h := range cfg.Heads {
		off := h * dh
		for i := range T {
			lo, hi := max(i-cfg.Window, 0), min(i+cfg.Window, T-1)
			qi := q[i*cfg.Dim+off : i*cfg.Dim+off+dh]
			var peak float32 = -math.MaxFloat32
			for j := lo; j <= hi; j++ {
				kj := k[j*cfg.Dim+off : j*cfg.Dim+off+dh]
				var s float32
				for d := range dh {
					s += qi[d] * kj[d]
				}
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
				w := scores[j-lo] / sum
				vj := v[j*cfg.Dim+off : j*cfg.Dim+off+dh]
				for d := range dh {
					oi[d] += w * vj[d]
				}
			}
		}
	}
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
