package transformer

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
)

// cellNet is a cell net's layers, each convolution's weights as PyTorch holds them
type cellNet struct {
	cfg                 Config
	reduce, wide, cells conv
	near                []conv
	cellout             conv
	frame1, frame2      conv // over the frames alone, Freq 1
	frameout            conv // a linear layer, Time and Freq 1
}

func newCellNet(cfg Config) net {
	c := cfg.Cells
	k := func(out, in, time, freq int) conv {
		return conv{Conv: Conv{Out: out, Time: time, Freq: freq, Pool: 1}, in: in, W: make([]float32, out*in*time*freq), B: make([]float32, out)}
	}
	n := &cellNet{cfg: cfg,
		reduce:   k(c.Reduce, cfg.channels(), 1, 1),
		wide:     k(c.Wide, c.Reduce, c.WideTime, c.WideFreq),
		cells:    k(c.Cell, c.Wide, c.CellTime, c.CellFreq),
		cellout:  k(c.channels(cfg.Outputs), c.Cell, 1, 1),
		frame1:   k(c.Frame, c.pooled(), c.FrameTime, 1),
		frame2:   k(c.Frame, c.Frame, c.AgainTime, 1),
		frameout: k(cfg.outputSize()-c.channels(cfg.Outputs)*c.Read, c.Frame, 1, 1),
	}
	for range c.Depth {
		n.near = append(n.near, k(c.Cell, c.Cell, c.NearTime, c.NearFreq))
	}
	return n
}

func (n *cellNet) tensors() []tensor {
	type layer struct {
		name string
		c    *conv
		dims int
	}
	ls := []layer{{"reduce", &n.reduce, 4}, {"wide", &n.wide, 4}, {"cells", &n.cells, 4}}
	for i := range n.near {
		ls = append(ls, layer{fmt.Sprintf("near.%d", i), &n.near[i], 4})
	}
	ls = append(ls, layer{"cellout", &n.cellout, 4}, layer{"frame1", &n.frame1, 3}, layer{"frame2", &n.frame2, 3}, layer{"frameout", &n.frameout, 2})
	var ts []tensor
	for _, l := range ls {
		dims := []int{l.c.Out, l.c.in, l.c.Time, l.c.Freq}[:l.dims]
		ts = append(ts, tensor{l.name + ".weight", dims, l.c.W}, tensor{l.name + ".bias", []int{l.c.Out}, l.c.B})
	}
	return ts
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

// rows is a layer over a block of frames, a frame's row after another's: each
// row its features from first, each feature's channels together, between pad
// zero features either side, row 0 the frame base's
type rows struct {
	buf            []float32
	base, first    int
	feats, ch, pad int
}

func newRows(frames, first, feats, ch, pad int) *rows {
	return &rows{buf: make([]float32, frames*(feats+2*pad)*ch), first: first, feats: feats, ch: ch, pad: pad}
}

func (r *rows) row(t int) []float32 {
	w := (r.feats + 2*r.pad) * r.ch
	return r.buf[(t-r.base)*w:][:w]
}

// from is where feature j's channels start in a row, j counted from feature 0
func (r *rows) from(j int) int { return (j - r.first + r.pad) * r.ch }

// arranged is a convolution's weights as run reads them: each output's kernel
// frame by frame, feature by feature, the channels innermost
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

// run writes the convolution of in into out's features at frames from to to,
// output j centred on input feature stride·j and its frames dil apart, the
// frames past the sequence's T zero, then res added when given, then the GELU
// when act; w is arranged, and patch holds a kernel's inputs
func (c *conv) run(w []float32, in, out, res *rows, from, to, T, stride, dil int, act bool, patch []float32) {
	span := c.Freq * in.ch
	p := patch[:c.Time*span]
	for t := from; t < to; t++ {
		row := out.row(t)
		for j := out.first; j < out.first+out.feats; j++ {
			for dt := range c.Time {
				seg := p[dt*span:][:span]
				if tt := t + (dt-c.Time/2)*dil; tt >= 0 && tt < T {
					copy(seg, in.row(tt)[in.from(stride*j-c.Freq/2):])
				} else {
					clear(seg)
				}
			}
			at := row[out.from(j):][:c.Out]
			for o := range at {
				v := dot(c.B[o], w[o*len(p):][:len(p)], p)
				if res != nil {
					v += res.row(t)[res.from(j)+o]
				}
				if act {
					v = gelu(v)
				}
				at[o] = v
			}
		}
	}
}

// reach is how many frames either side of a block each layer is computed over:
// what the layers after it read
type reach struct{ reduced, wide, cells, pooled, frame int }

func (c *Cells) reach() reach {
	var r reach
	r.cells = c.Depth * (c.NearTime / 2)
	r.frame = c.Dilation * (c.AgainTime / 2)
	r.pooled = r.frame + c.FrameTime/2
	r.wide = max(r.cells+c.CellTime/2, r.pooled)
	r.reduced = r.wide + c.WideTime/2
	return r
}

// cellScratch is a goroutine's layers over a block
type cellScratch struct {
	input, reduced, wide, heads, pooled, frame, again, out *rows
	cells                                                  [2]*rows // a near convolution reads one and writes the other
	patch                                                  []float32
}

func (n *cellNet) scratch(block int) *cellScratch {
	c, r := n.cfg.Cells, n.cfg.Cells.reach()
	cells := func() *rows { return newRows(block+2*r.cells, 0, c.cells(), c.Cell, c.NearFreq/2) }
	patch := 0
	for _, l := range append([]conv{n.reduce, n.wide, n.cells, n.cellout, n.frame1, n.frame2, n.frameout}, n.near...) {
		patch = max(patch, l.Time*l.Freq*l.in)
	}
	return &cellScratch{
		input:   newRows(block+2*r.reduced, 0, c.Bins, n.cfg.channels(), 0),
		reduced: newRows(block+2*r.reduced, 0, c.Bins, c.Reduce, c.WideFreq/2),
		wide:    newRows(block+2*r.wide, 0, c.Bins, c.Wide, c.CellFreq/2),
		cells:   [2]*rows{cells(), cells()},
		heads:   newRows(block, c.First, c.Read, n.cellout.Out, 0),
		pooled:  newRows(block+2*r.pooled, 0, 1, c.pooled(), 0),
		frame:   newRows(block+2*r.frame, 0, 1, c.Frame, 0),
		again:   newRows(block, 0, 1, c.Frame, 0),
		out:     newRows(block, 0, 1, n.frameout.Out, 0),
		patch:   make([]float32, patch),
	}
}

func (n *cellNet) forward(frames [][]float32, y []float32) {
	w := map[*conv][]float32{}
	for _, l := range append([]*conv{&n.reduce, &n.wide, &n.cells, &n.cellout, &n.frame1, &n.frame2, &n.frameout}, n.nears()...) {
		w[l] = l.arranged()
	}
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

func (n *cellNet) nears() []*conv {
	ls := make([]*conv, len(n.near))
	for i := range n.near {
		ls[i] = &n.near[i]
	}
	return ls
}

// block writes frames a to b's outputs into y, each layer over the frames the
// next reads
func (n *cellNet) block(frames [][]float32, a, b int, w map[*conv][]float32, s *cellScratch, y []float32) {
	c, r, T := n.cfg.Cells, n.cfg.Cells.reach(), len(frames)
	over := func(h int) (int, int) { return max(a-h, 0), min(b+h, T) }
	for l, h := range map[*rows]int{s.input: r.reduced, s.reduced: r.reduced, s.wide: r.wide, s.cells[0]: r.cells, s.cells[1]: r.cells,
		s.heads: 0, s.pooled: r.pooled, s.frame: r.frame, s.again: 0, s.out: 0} {
		l.base = a - h
	}

	from, to := over(r.reduced)
	for t := from; t < to; t++ {
		row, x := s.input.row(t), frames[t]
		for i := range s.input.ch {
			for k, v := range x[i*n.cfg.Inputs:][:c.Bins] {
				row[k*s.input.ch+i] = v
			}
		}
	}
	n.reduce.run(w[&n.reduce], s.input, s.reduced, nil, from, to, T, 1, 1, true, s.patch)
	from, to = over(r.wide)
	n.wide.run(w[&n.wide], s.reduced, s.wide, nil, from, to, T, 1, 1, true, s.patch)
	from, to = over(r.cells)
	n.cells.run(w[&n.cells], s.wide, s.cells[0], nil, from, to, T, c.Stride, 1, true, s.patch)
	for i := range n.near {
		in, out := s.cells[i%2], s.cells[(i+1)%2]
		from, to = over(r.cells - (i+1)*(c.NearTime/2))
		n.near[i].run(w[&n.near[i]], in, out, in, from, to, T, 1, 1, true, s.patch)
	}
	n.cellout.run(w[&n.cellout], s.cells[len(n.near)%2], s.heads, nil, a, b, T, 1, 1, false, s.patch)

	from, to = over(r.pooled)
	for t := from; t < to; t++ {
		in, out := s.wide.row(t)[s.wide.from(0):], s.pooled.row(t)
		for ch := range c.Wide {
			for g := range c.Regions {
				lo, hi := g*c.Bins/c.Regions, ((g+1)*c.Bins+c.Regions-1)/c.Regions
				var sum float32
				for k := lo; k < hi; k++ {
					sum += in[k*c.Wide+ch]
				}
				out[ch*c.Regions+g] = sum / float32(hi-lo)
			}
		}
		for i, sc := range c.Scalars {
			out[c.Wide*c.Regions+i] = frames[t][sc]
		}
	}
	from, to = over(r.frame)
	n.frame1.run(w[&n.frame1], s.pooled, s.frame, nil, from, to, T, 1, 1, true, s.patch)
	n.frame2.run(w[&n.frame2], s.frame, s.again, s.frame, a, b, T, 1, c.Dilation, true, s.patch)
	n.frameout.run(w[&n.frameout], s.again, s.out, nil, a, b, T, 1, 1, false, s.patch)

	size, at, ch := n.cfg.outputSize(), 0, 0
	for _, h := range n.cfg.Outputs[:c.Heads] {
		for t := a; t < b; t++ {
			row, yt := s.heads.row(t), y[t*size+at:][:h.Size]
			for k := range h.Size / c.Read {
				for j := range c.Read {
					yt[k*c.Read+j] = row[j*s.heads.ch+ch+k]
				}
			}
		}
		at, ch = at+h.Size, ch+h.Size/c.Read
	}
	for t := a; t < b; t++ {
		copy(y[t*size+at:], s.out.row(t))
	}
}
