package transformer

import (
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
)

// cellNet is a cell net's layers, each convolution's weights as PyTorch holds them
type cellNet struct {
	cfg                                Config
	reduce, wide, cells, near, cellout conv // out × in × time × freq
	frame1, frame2                     conv // out × in × time
	frameout                           linear
}

func newCellNet(cfg Config) net {
	c := cfg.Cells
	k := func(out, in, time, freq int) conv {
		return conv{Conv: Conv{Out: out, Time: time, Freq: freq, Pool: 1}, in: in, W: make([]float32, out*in*time*freq), B: make([]float32, out)}
	}
	return &cellNet{cfg: cfg,
		reduce:   k(c.Reduce, cfg.channels(), 1, 1),
		wide:     k(c.Wide, c.Reduce, c.WideTime, c.WideFreq),
		cells:    k(c.Cell, c.Wide, c.CellTime, c.CellFreq),
		near:     k(c.Cell, c.Cell, c.NearTime, c.NearFreq),
		cellout:  k(c.Heads, c.Cell, 1, 1),
		frame1:   k(c.Frame, c.pooled(), c.FrameTime, 1),
		frame2:   k(c.Frame, c.Frame, c.AgainTime, 1),
		frameout: newLinear(c.Frame, cfg.outputSize()-c.Heads*cfg.Outputs[0].Size),
	}
}

func (n *cellNet) tensors() []tensor {
	var ts []tensor
	for _, l := range []struct {
		name string
		c    *conv
		dims int
	}{{"reduce", &n.reduce, 4}, {"wide", &n.wide, 4}, {"cells", &n.cells, 4}, {"near", &n.near, 4}, {"cellout", &n.cellout, 4},
		{"frame1", &n.frame1, 3}, {"frame2", &n.frame2, 3}} {
		dims := []int{l.c.Out, l.c.in, l.c.Time, l.c.Freq}[:l.dims]
		ts = append(ts, tensor{l.name + ".weight", dims, l.c.W}, tensor{l.name + ".bias", []int{l.c.Out}, l.c.B})
	}
	l := &n.frameout
	return append(ts, tensor{"frameout.weight", []int{l.Out, l.In}, l.W}, tensor{"frameout.bias", []int{l.Out}, l.B})
}

func (n *cellNet) randomize(rng *rand.Rand) {
	for _, t := range n.tensors() {
		if len(t.dims) < 2 {
			continue
		}
		fan := len(t.data) / t.dims[0]
		for i := range t.data {
			t.data[i] = float32(rng.NormFloat64() / math.Sqrt(float64(fan)))
		}
	}
}

// cellBlock is the most frames a goroutine runs at once: a block computes again
// the frames around it that its outputs reach, so larger ones waste less, and
// the blocks come in a multiple of the cores, so the last round keeps them busy
const cellBlock = 128

// arranged is a cell net's kernels as its passes read them: each output's frame
// by frame, a frame's features by channel, so a channel-innermost row is one dot
type arranged struct{ wide, cells, near, frame1, frame2 []float32 }

func (c *conv) arranged() []float32 {
	w := make([]float32, len(c.W))
	for o := range c.Out {
		for i := range c.in {
			for t := range c.Time {
				for f := range c.Freq {
					w[((o*c.Time+t)*c.Freq+f)*c.in+i] = c.W[((o*c.in+i)*c.Time+t)*c.Freq+f]
				}
			}
		}
	}
	return w
}

// reach is how many frames either side of a block each layer is computed over,
// what the layers after it read; pads are the zero features either side of a row
type reach struct{ reduced, wide, cells, pooled, frame int }

func (c *Cells) reach() reach {
	var r reach
	r.cells = c.NearTime / 2
	r.frame = c.Dilation * (c.AgainTime / 2)
	r.pooled = r.frame + c.FrameTime/2
	r.wide = max(r.cells+c.CellTime/2, r.pooled)
	r.reduced = r.wide + c.WideTime/2
	return r
}

// cellScratch is a goroutine's layers over a block, each a frame's row after another's
type cellScratch struct {
	reduced, wide, cells, pooled, frame []float32
	sum, patch, v                       []float32
}

func (n *cellNet) scratch(block int) *cellScratch {
	c, r := n.cfg.Cells, n.cfg.Cells.reach()
	rows := func(h, width int) []float32 { return make([]float32, (block+2*h)*width) }
	return &cellScratch{
		reduced: rows(r.reduced, (c.Bins+c.WideFreq/2*2)*c.Reduce),
		wide:    rows(r.wide, (c.CellFreq/2+c.Bins)*c.Wide),
		cells:   rows(r.cells, (c.cells()+c.NearFreq/2*2)*c.Cell),
		pooled:  rows(r.pooled, c.pooled()),
		frame:   rows(r.frame, c.Frame),
		sum:     make([]float32, c.Bins),
		patch:   make([]float32, max(c.CellTime*c.CellFreq*c.Wide, c.NearTime*c.NearFreq*c.Cell)),
		v:       make([]float32, max(c.Cell, c.Frame)),
	}
}

func (n *cellNet) forward(frames [][]float32, y []float32) {
	w := &arranged{n.wide.arranged(), n.cells.arranged(), n.near.arranged(), n.frame1.arranged(), n.frame2.arranged()}
	T, procs := len(frames), runtime.GOMAXPROCS(0)
	blocks := procs * ((T + procs*cellBlock - 1) / (procs * cellBlock))
	block := (T + blocks - 1) / blocks
	pool := sync.Pool{New: func() any { return n.scratch(block) }}
	across(T, block, func(a, b int) {
		s := pool.Get().(*cellScratch)
		n.block(frames, a, b, w, s, y)
		pool.Put(s)
	})
}

// block writes frames a to b's outputs into y, each layer over the frames the
// next reads, the frames past the sequence's ends zero
func (n *cellNet) block(frames [][]float32, a, b int, w *arranged, s *cellScratch, y []float32) {
	c, r, T := n.cfg.Cells, n.cfg.Cells.reach(), len(frames)
	inside := func(t int) bool { return t >= 0 && t < T }
	over := func(h int) (int, int) { return max(a-h, 0), min(b+h, T) }
	padW, padC, padN := c.WideFreq/2, c.CellFreq/2, c.NearFreq/2
	rw, ww, cw, pw := (c.Bins+2*padW)*c.Reduce, (padC+c.Bins)*c.Wide, (c.cells()+2*padN)*c.Cell, c.pooled()
	row := func(buf []float32, h, width, t int) []float32 { return buf[(t-a+h)*width:][:width] }
	acc := s.sum

	for t, end := over(r.reduced); t < end; t++ {
		out, x := row(s.reduced, r.reduced, rw, t), frames[t]
		for o := range c.Reduce {
			for k := range acc {
				acc[k] = n.reduce.B[o]
			}
			for i := range n.reduce.in {
				axpy(n.reduce.W[o*n.reduce.in+i], x[i*n.cfg.Inputs:][:c.Bins], acc)
			}
			for k, v := range acc {
				out[(padW+k)*c.Reduce+o] = gelu(v)
			}
		}
	}

	span := c.WideFreq * c.Reduce
	for t, end := over(r.wide); t < end; t++ {
		out := row(s.wide, r.wide, ww, t)
		for o := range c.Wide {
			for k := range acc {
				acc[k] = n.wide.B[o]
			}
			for dt := range c.WideTime {
				if tt := t + dt - c.WideTime/2; inside(tt) {
					in, kw := row(s.reduced, r.reduced, rw, tt), w.wide[(o*c.WideTime+dt)*span:][:span]
					for k := range acc {
						acc[k] = dot(acc[k], kw, in[k*c.Reduce:][:span])
					}
				}
			}
			for k, v := range acc {
				out[(padC+k)*c.Wide+o] = gelu(v)
			}
		}
	}

	span = c.CellFreq * c.Wide
	for t, end := over(r.cells); t < end; t++ {
		out, p := row(s.cells, r.cells, cw, t), s.patch[:c.CellTime*span]
		for j := range c.cells() {
			for dt := range c.CellTime {
				seg := p[dt*span:][:span]
				if tt := t + dt - c.CellTime/2; inside(tt) {
					copy(seg, row(s.wide, r.wide, ww, tt)[c.Stride*j*c.Wide:])
				} else {
					clear(seg)
				}
			}
			for o := range c.Cell {
				out[(padN+j)*c.Cell+o] = gelu(dot(n.cells.B[o], w.cells[o*len(p):][:len(p)], p))
			}
		}
	}

	size, read := n.cfg.outputSize(), n.cfg.Outputs[0].Size
	span = c.NearFreq * c.Cell
	for t := a; t < b; t++ {
		at, p, v := row(s.cells, r.cells, cw, t), s.patch[:c.NearTime*span], s.v[:c.Cell]
		for j := c.First; j < c.First+read; j++ {
			for dt := range c.NearTime {
				seg := p[dt*span:][:span]
				if tt := t + dt - c.NearTime/2; inside(tt) {
					copy(seg, row(s.cells, r.cells, cw, tt)[j*c.Cell:])
				} else {
					clear(seg)
				}
			}
			for o := range c.Cell {
				v[o] = gelu(dot(n.near.B[o], w.near[o*len(p):][:len(p)], p) + at[(padN+j)*c.Cell+o])
			}
			for h := range c.Heads {
				y[t*size+h*read+j-c.First] = dot(n.cellout.B[h], n.cellout.W[h*c.Cell:][:c.Cell], v)
			}
		}
	}

	for t, end := over(r.pooled); t < end; t++ {
		out, in := row(s.pooled, r.pooled, pw, t), row(s.wide, r.wide, ww, t)[padC*c.Wide:]
		for ch := range c.Wide {
			for g := range c.Regions {
				from, to := g*c.Bins/c.Regions, ((g+1)*c.Bins+c.Regions-1)/c.Regions
				var sum float32
				for k := from; k < to; k++ {
					sum += in[k*c.Wide+ch]
				}
				out[ch*c.Regions+g] = sum / float32(to-from)
			}
		}
		for i, sc := range c.Scalars {
			out[c.Wide*c.Regions+i] = frames[t][sc]
		}
	}

	for t, end := over(r.frame); t < end; t++ {
		out := row(s.frame, r.frame, c.Frame, t)
		for o := range c.Frame {
			v := n.frame1.B[o]
			for dt := range c.FrameTime {
				if tt := t + dt - c.FrameTime/2; inside(tt) {
					v = dot(v, w.frame1[(o*c.FrameTime+dt)*pw:][:pw], row(s.pooled, r.pooled, pw, tt))
				}
			}
			out[o] = gelu(v)
		}
	}

	at := c.Heads * read
	for t := a; t < b; t++ {
		v, in := s.v[:c.Frame], row(s.frame, r.frame, c.Frame, t)
		for o := range c.Frame {
			u := n.frame2.B[o]
			for dt := range c.AgainTime {
				if tt := t + (dt-c.AgainTime/2)*c.Dilation; inside(tt) {
					u = dot(u, w.frame2[(o*c.AgainTime+dt)*c.Frame:][:c.Frame], row(s.frame, r.frame, c.Frame, tt))
				}
			}
			v[o] = gelu(u + in[o])
		}
		l := &n.frameout
		for o := range l.Out {
			y[t*size+at+o] = dot(l.B[o], l.W[o*l.In:][:l.In], v)
		}
	}
}
