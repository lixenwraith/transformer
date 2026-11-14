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
	near, long          []conv
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
	for range c.Long {
		n.long = append(n.long, k(c.Cell, c.Cell, longTime, c.NearFreq))
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
	for i := range n.long {
		ls = append(ls, layer{fmt.Sprintf("long.%d", i), &n.long[i], 4})
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

// longTime is a long convolution's frames
const longTime = 3

// cellBlock is the fewest frames a goroutine may run at once, and blockReach how
// many of its reach it runs at least: a block computes again the frames around it
// that its outputs reach, so larger ones waste less, and the blocks come in a
// multiple of the cores, so the last round keeps them busy; a pass takes
// cellChunk frames through a layer at a time
const (
	cellBlock  = 128
	blockReach = 8
	cellChunk  = 16
)

// rows is a layer over the frames a pass still reads, a frame's row after
// another's from frame base on, made up to made: each row its features from
// first, each feature's channels together, between pad zero features either
// side. It trails the input by lag frames, and later layers read keep before made.
type rows struct {
	buf            []float32
	base, made     int
	lag, keep      int
	reach          int // frames of it before a frame that the frame's output reads
	first          int
	feats, ch, pad int
	width          int // a row's floats
}

// newRows holds keep frames and the chunk more a pass makes at a time
func newRows(lag, keep, chunk, first, feats, ch, pad int) *rows {
	w := (feats + 2*pad) * ch
	return &rows{buf: make([]float32, (keep+chunk)*w), lag: lag, keep: keep, first: first, feats: feats, ch: ch, pad: pad, width: w}
}

func (r *rows) row(t int) []float32 { return r.buf[(t-r.base)*r.width:][:r.width] }

// span is frames a to b's rows
func (r *rows) span(a, b int) []float32 { return r.buf[(a-r.base)*r.width : (b-r.base)*r.width] }

// room lets go of the frames no layer reads again, so that those up to upto fit
func (r *rows) room(upto int) {
	if (upto-r.base)*r.width > len(r.buf) {
		from := max(r.made-r.keep, r.base)
		copy(r.buf, r.span(from, r.made))
		r.base = from
	}
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
				at[o] = v
			}
		}
		if act {
			geluRow(row[out.from(out.first):out.from(out.first+out.feats)])
		}
	}
}

// reach is how many frames either side of its own a frame's output reads: the
// convolutions over the cells or the frame path's, whichever reach further, and
// the wide one's
func (n *cellNet) reach() int {
	c := n.cfg.Cells
	cells := c.CellTime / 2
	for i, l := range n.overCells() {
		cells += n.apart(i) * (l.Time / 2)
	}
	return max(cells, c.FrameTime/2+c.Dilation*(c.AgainTime/2)) + c.WideTime/2
}

// overCells is the convolutions over the cells, in the order they run
func (n *cellNet) overCells() []*conv {
	var ls []*conv
	for i := range n.near {
		ls = append(ls, &n.near[i])
	}
	for i := range n.long {
		ls = append(ls, &n.long[i])
	}
	return ls
}

// apart is how far apart the frames of overCells' i-th convolution lie
func (n *cellNet) apart(i int) int {
	if i < len(n.near) {
		return 1
	}
	return n.cfg.Cells.Long[i-len(n.near)]
}

// arrange is every convolution's weights as run reads them
func (n *cellNet) arrange() map[*conv][]float32 {
	w := map[*conv][]float32{}
	for _, l := range append([]*conv{&n.reduce, &n.wide, &n.cells, &n.cellout, &n.frame1, &n.frame2, &n.frameout}, n.overCells()...) {
		w[l] = l.arranged()
	}
	return w
}

// cellPass is a cell net over a sequence as its frames come, a chunk of them
// through a layer at a time, each layer over the frames its inputs reach
type cellPass struct {
	n                                                               *cellNet
	w                                                               map[*conv][]float32
	over                                                            []*conv
	lag                                                             int // frames an output trails the input by
	in, T                                                           int // frames taken, counted on past the end, and the sequence's once it ends
	done                                                            int // frames whose outputs are made
	from, to                                                        int // the frames whose outputs are wanted
	input, scalars, reduced, wide, heads, pooled, frame, again, out *rows
	cells                                                           []*rows // the cells, then as each convolution over them leaves them
	patch                                                           []float32
}

func (n *cellNet) pass(int) pass { return n.newPass(n.arrange()) }

// newPass lays out each layer's rows: how far it trails the input, the frames
// before its last that the layers after it read, and how far back an output reads it
func (n *cellNet) newPass(w map[*conv][]float32) *cellPass {
	c, over := n.cfg.Cells, n.overCells()
	wt, ct, ft, at := c.WideTime/2, c.CellTime/2, c.FrameTime/2, c.Dilation*(c.AgainTime/2)
	p := &cellPass{n: n, w: w, over: over, lag: n.reach(), T: math.MaxInt, to: math.MaxInt}
	mk := func(lag, keep, reach, first, feats, ch, pad int) *rows {
		r := newRows(lag, keep, cellChunk, first, feats, ch, pad)
		r.reach = reach
		return r
	}
	p.input = mk(0, 0, p.lag, 0, c.Bins, n.cfg.channels(), 0)
	p.scalars = mk(0, wt, p.lag, 0, 1, len(c.Scalars), 0)
	p.reduced = mk(0, 2*wt, p.lag, 0, c.Bins, c.Reduce, c.WideFreq/2)
	p.wide = mk(wt, 2*ct, p.lag-wt, 0, c.Bins, c.Wide, c.CellFreq/2)
	lag, last := wt+ct, wt+ct // the cells' lag, and their last's
	for i, l := range over {
		last += n.apart(i) * (l.Time / 2)
	}
	for i := 0; i <= len(over); i++ {
		h := 0
		if i < len(over) {
			h = n.apart(i) * (over[i].Time / 2)
		}
		p.cells = append(p.cells, mk(lag, 2*h, last-lag, 0, c.cells(), c.Cell, c.NearFreq/2))
		lag += h
	}
	p.heads = mk(last, p.lag-last, 0, c.First, c.Read, n.cellout.Out, 0)
	p.pooled = mk(wt, 2*ft, ft+at, 0, 1, c.pooled(), 0)
	p.frame = mk(wt+ft, 2*at, at, 0, 1, c.Frame, 0)
	p.again = mk(wt+ft+at, 0, 0, 0, 1, c.Frame, 0)
	p.out = mk(wt+ft+at, p.lag-(wt+ft+at), 0, 0, 1, n.frameout.Out, 0)
	patch := 0
	for l := range w {
		patch = max(patch, l.Time*l.Freq*l.in)
	}
	p.patch = make([]float32, patch)
	return p
}

// restart begins a sequence anew, the outputs of frames from to to wanted and
// each layer made from the first frame they read
func (p *cellPass) restart(from, to int) {
	p.in, p.T, p.done, p.from, p.to = 0, math.MaxInt, 0, from, to
	for _, r := range append([]*rows{p.input, p.scalars, p.reduced, p.wide, p.heads, p.pooled, p.frame, p.again, p.out}, p.cells...) {
		r.base = max(from-r.reach, 0)
		r.made = r.base
	}
}

func (n *cellNet) forward(frames [][]float32, y []float32) {
	w, r := n.arrange(), n.reach()
	T, procs, per := len(frames), runtime.GOMAXPROCS(0), n.cfg.outputSize()
	block := T // one core runs each frame once, no block's reach again
	if procs > 1 {
		size := max(cellBlock, blockReach*r)
		blocks := procs * ((T + procs*size - 1) / (procs * size))
		block = (T + blocks - 1) / blocks
	}
	pool := sync.Pool{New: func() any { return n.newPass(w) }}
	across(T, block, func(a, b int) {
		p := pool.Get().(*cellPass)
		lo, hi := max(a-r, 0), min(b+r, T)
		p.restart(a-lo, b-lo)
		made := p.write(frames[lo:hi], y[a*per:b*per])
		if hi == T {
			p.close(y[(a+made)*per : b*per])
		}
		pool.Put(p)
	})
}

func (p *cellPass) write(frames [][]float32, y []float32) int {
	c, made := p.n.cfg.Cells, 0
	for len(frames) > 0 {
		k := min(len(frames), cellChunk)
		p.input.room(p.in + k)
		p.scalars.room(p.in + k)
		for t, x := range frames[:k] {
			row, sc := p.input.row(p.in+t), p.scalars.row(p.in+t)
			for i := range p.input.ch {
				for j, v := range x[i*p.n.cfg.Inputs:][:c.Bins] {
					row[j*p.input.ch+i] = v
				}
			}
			for i, s := range c.Scalars {
				sc[i] = x[s]
			}
		}
		p.in += k
		p.input.made, p.scalars.made, frames = p.in, p.in, frames[k:]
		made += p.advance(y[made*p.n.cfg.outputSize():])
	}
	return made
}

func (p *cellPass) close(y []float32) int {
	made := 0
	for p.T = p.in; p.done < p.T; {
		p.in += cellChunk
		made += p.advance(y[made*p.n.cfg.outputSize():])
	}
	return made
}

// upto is how many frames a layer lag behind the input has the frames to make
func (p *cellPass) upto(lag int) int { return max(min(p.in-lag, p.T), 0) }

// advance runs each layer over the frames its inputs now reach and writes the
// wanted outputs they complete into y; it returns how many
func (p *cellPass) advance(y []float32) int {
	n, c := p.n, p.n.cfg.Cells
	run := func(l *conv, in, out, res *rows, stride, dil int, act bool) {
		if to := p.upto(out.lag); to > out.made {
			out.room(to)
			l.run(p.w[l], in, out, res, out.made, to, p.T, stride, dil, act, p.patch)
			out.made = to
		}
	}
	run(&n.reduce, p.input, p.reduced, nil, 1, 1, true)
	run(&n.wide, p.reduced, p.wide, nil, 1, 1, true)
	run(&n.cells, p.wide, p.cells[0], nil, c.Stride, 1, true)
	for i, l := range p.over {
		run(l, p.cells[i], p.cells[i+1], p.cells[i], 1, n.apart(i), true)
	}
	run(&n.cellout, p.cells[len(p.cells)-1], p.heads, nil, 1, 1, false)

	if to := p.upto(p.pooled.lag); to > p.pooled.made {
		p.pooled.room(to)
		first, bins := c.pool()
		for t := p.pooled.made; t < to; t++ {
			in, out := p.wide.row(t)[p.wide.from(first):], p.pooled.row(t)
			for ch := range c.Wide {
				for g := range c.Regions {
					lo, hi := g*bins/c.Regions, ((g+1)*bins+c.Regions-1)/c.Regions
					var sum float32
					for k := lo; k < hi; k++ {
						sum += in[k*c.Wide+ch]
					}
					out[ch*c.Regions+g] = sum / float32(hi-lo)
				}
			}
			copy(out[c.Wide*c.Regions:], p.scalars.row(t))
		}
		p.pooled.made = to
	}
	run(&n.frame1, p.pooled, p.frame, nil, 1, 1, true)
	run(&n.frame2, p.frame, p.again, p.frame, 1, c.Dilation, true)
	run(&n.frameout, p.again, p.out, nil, 1, 1, false)

	size, made := n.cfg.outputSize(), 0
	for t := p.done; t < p.upto(p.lag); t, p.done = t+1, t+1 {
		if t < p.from || t >= p.to {
			continue
		}
		row, yt, ch := p.heads.row(t), y[made*size:][:size], 0
		for _, h := range n.cfg.Outputs[:c.Heads] {
			for k := range h.Size / c.Read {
				for j := range c.Read {
					yt[k*c.Read+j] = row[j*p.heads.ch+ch+k]
				}
			}
			yt, ch = yt[h.Size:], ch+h.Size/c.Read
		}
		copy(yt, p.out.row(t))
		made++
	}
	return made
}
