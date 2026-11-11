package transformer

import "math"

// Stream is a sequence a model takes as its frames come, each frame through
// each layer once: Write returns the outputs its frames complete and Close the
// rest. Any number may run over one model, each on one goroutine at a time.
type Stream struct {
	m       *Model
	p       pass
	in, out int // frames taken and outputs given
}

// pass is a kind's layers over a sequence as its frames come: write puts the
// outputs its frames complete into y, a frame's after another's, and close those
// of the frames left, the sequence ending; each returns how many
type pass interface {
	write(frames [][]float32, y []float32) int
	close(y []float32) int
}

// Reach is how many frames before a frame and after it its output reads
func (m *Model) Reach() int { return m.net.reach() }

// Stream starts a sequence whose outputs wait for ahead frames after their own.
// At Reach or more they are Forward's. Under it a transformer's blocks share
// what its convolutions leave of ahead, each attending that few frames past a
// frame, the first blocks taking the remainder; a cell net waits its Reach.
func (m *Model) Stream(ahead int) *Stream { return &Stream{m: m, p: m.net.pass(ahead)} }

// Write takes the sequence's next frames and returns the outputs they complete,
// in order from the last returned.
func (s *Stream) Write(frames [][]float32) [][]float32 {
	s.in += len(frames)
	y := make([]float32, len(frames)*s.m.cfg.outputSize())
	return s.give(y, s.p.write(frames, y))
}

// Close ends the sequence and returns the outputs of the frames left, which
// read nothing past the end, as Forward's last do. It takes no frames after.
func (s *Stream) Close() [][]float32 {
	y := make([]float32, (s.in-s.out)*s.m.cfg.outputSize())
	return s.give(y, s.p.close(y))
}

func (s *Stream) give(y []float32, n int) [][]float32 {
	s.out += n
	size := s.m.cfg.outputSize()
	out := make([][]float32, n)
	for t := range out {
		out[t] = y[t*size : (t+1)*size : (t+1)*size]
	}
	return out
}

// passChunk is the most frames a transformer's pass takes through a layer at
// a time: enough to hand its cores a block each
const passChunk = 64

// transformerPass is a transformer over a sequence as its frames come, a chunk
// of them through a layer at a time, each layer over the frames its inputs reach
type transformerPass struct {
	m             *transformerNet
	in, T         int     // frames taken, counted on past the end, and the sequence's once it ends
	done, lag     int     // outputs made, and the frames an output trails the input by
	x             []*rows // the frames, then as each convolution leaves them
	h             *rows   // as the input layer leaves them
	towers        []towerPass
	n, att, o, ff []float32 // a chunk's norms, attention, sums and feed-forward
}

// towerPass is a tower's blocks and its outputs, kept until every tower's are made
type towerPass struct {
	blocks []blockPass
	y      *rows
}

// blockPass is a block's queries over the frames still waiting for those ahead
// of them, its keys and values over those its attention still reads, and its
// outputs
type blockPass struct {
	ahead      int // frames its attention reads past a frame
	q, k, v, h *rows
}

func (m *transformerNet) reach() int {
	r := 0
	for _, tw := range m.towers {
		r = max(r, len(tw.layers)*m.cfg.Window)
	}
	for _, c := range m.convs {
		r += c.Time / 2
	}
	return r
}

// pass lays out each layer's rows: how far it trails the input, and the frames
// before its last that the layers after it read
func (m *transformerNet) pass(ahead int) pass {
	cfg := m.cfg
	p := &transformerPass{m: m, T: math.MaxInt, towers: make([]towerPass, len(m.towers)),
		n: make([]float32, passChunk*cfg.Dim), att: make([]float32, passChunk*cfg.Dim), o: make([]float32, passChunk*cfg.Dim), ff: make([]float32, passChunk*cfg.FFN)}
	flat := func(lag, keep, width int) *rows { return newRows(lag, keep, passChunk, 0, 1, width, 0) }
	lag, width := 0, cfg.Inputs*cfg.channels()
	for i, f := 0, cfg.Inputs; i < len(m.convs); i, f = i+1, f/m.convs[i].Pool {
		c := &m.convs[i]
		p.x = append(p.x, flat(lag, 2*(c.Time/2), width))
		lag, width = lag+c.Time/2, c.Out*(f/c.Pool)
	}
	p.x = append(p.x, flat(lag, 0, width))
	first := 0 // the most frames a tower's first block waits for
	for ti, tw := range m.towers {
		tp, at := &p.towers[ti], lag
		spare := max(ahead-lag, 0)
		for i := range tw.layers {
			a := spare / len(tw.layers)
			if i < spare%len(tw.layers) {
				a++
			}
			a = min(a, cfg.Window)
			if i == 0 {
				first = max(first, a)
			} else {
				tp.blocks[i-1].h.keep = a
			}
			tp.blocks = append(tp.blocks, blockPass{ahead: a, q: flat(at, a, cfg.Dim), k: flat(at, cfg.Window+a, cfg.Dim), v: flat(at, cfg.Window+a, cfg.Dim), h: &rows{lag: at + a}})
			at += a
		}
		tp.y = &rows{lag: at}
		p.lag = max(p.lag, at)
	}
	p.h = flat(lag, first, cfg.Dim)
	for ti := range p.towers {
		tp := &p.towers[ti]
		for i := range tp.blocks {
			b := &tp.blocks[i]
			b.h = flat(b.h.lag, b.h.keep, cfg.Dim)
		}
		tp.y = flat(tp.y.lag, p.lag-tp.y.lag, m.towers[ti].out.Out)
	}
	return p
}

func (p *transformerPass) write(frames [][]float32, y []float32) int {
	x, made := p.x[0], 0
	for len(frames) > 0 {
		k := min(len(frames), passChunk)
		x.room(p.in + k)
		for t, f := range frames[:k] {
			copy(x.row(p.in+t), f)
		}
		p.in += k
		x.made, frames = p.in, frames[k:]
		made += p.advance(y[made*p.m.cfg.outputSize():])
	}
	return made
}

func (p *transformerPass) close(y []float32) int {
	made := 0
	for p.T = p.in; p.done < p.T; {
		p.in += passChunk
		made += p.advance(y[made*p.m.cfg.outputSize():])
	}
	return made
}

// upto is how many frames a layer lag behind the input has the frames to make
func (p *transformerPass) upto(lag int) int { return max(min(p.in-lag, p.T), 0) }

// advance runs each layer over the frames its inputs now reach and writes the
// outputs they complete into y; it returns how many
func (p *transformerPass) advance(y []float32) int {
	m, cfg := p.m, p.m.cfg
	x := p.x[0]
	for i, f := 0, cfg.Inputs; i < len(m.convs); i, f = i+1, f/m.convs[i].Pool {
		out := p.x[i+1]
		if a, to := out.made, p.upto(out.lag); to > a {
			out.room(to)
			m.convs[i].apply(x.buf, x.base, a, to, p.T, f, out.span(a, to))
			out.made = to
		}
		x = out
	}
	if a, to := p.h.made, x.made; to > a {
		p.h.room(to)
		m.in.apply(x.span(a, to), to-a, p.h.span(a, to))
		p.h.made = to
	}
	for ti := range m.towers {
		tw, tp, in := &m.towers[ti], &p.towers[ti], p.h
		for i := range tw.layers {
			p.block(&tw.layers[i], &tp.blocks[i], in)
			in = tp.blocks[i].h
		}
		if a, to := tp.y.made, in.made; to > a {
			tp.y.room(to)
			tw.ln.apply(in.span(a, to), to-a, p.n)
			tw.out.apply(p.n, to-a, tp.y.span(a, to))
			tp.y.made = to
		}
	}
	a, to := p.done, p.upto(p.lag)
	for t := a; t < to; t++ {
		yt := y[(t-a)*cfg.outputSize():]
		for ti := range p.towers {
			yt = yt[copy(yt, p.towers[ti].y.row(t)):]
		}
	}
	p.done = max(to, a)
	return p.done - a
}

// block runs a block over the frames its input x now reaches: the queries, keys
// and values of the new ones, then the frames whose frames ahead are in
func (p *transformerPass) block(l *layer, b *blockPass, x *rows) {
	if a, to := b.k.made, x.made; to > a {
		b.q.room(to)
		b.k.room(to)
		b.v.room(to)
		l.project(x.span(a, to), to-a, p.n, b.q.span(a, to), b.k.span(a, to), b.v.span(a, to))
		b.q.made, b.k.made, b.v.made = to, to, to
	}
	a, to := b.h.made, p.upto(b.h.lag)
	if to <= a {
		return
	}
	p.m.attend(b.q.span(a, to), b.k.buf, b.v.buf, a, to, b.k.base, b.ahead, p.T, p.att)
	l.o.apply(p.att, to-a, p.o)
	b.h.room(to)
	h := b.h.span(a, to)
	for j, v := range x.span(a, to) {
		h[j] = v + p.o[j]
	}
	l.feed(h, to-a, p.n, p.ff, p.o)
	b.h.made = to
}
