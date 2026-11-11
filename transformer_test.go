package transformer

import (
	"bytes"
	"encoding/binary"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func tiny() Config {
	return Config{Inputs: 5, Dim: 8, Heads: 2, Layers: 2, FFN: 16, Window: 3,
		Outputs: []Head{{"a", 3}, {"b", 2}}}
}

// tinyCells is a cell net of every part, three channels of 21 inputs, seven
// cells, two convolutions over them, two long ones and a cell head of two channels
func tinyCells() Config {
	return Config{Inputs: 21, Channels: 3, Outputs: []Head{{"c1", 5}, {"c2", 10}, {"f", 3}}, Cells: &Cells{
		Bins: 20, Reduce: 2, Wide: 3, WideTime: 3, WideFreq: 5, Cell: 4, CellTime: 3, CellFreq: 3, Stride: 3, Depth: 2, NearTime: 3, NearFreq: 3, Long: []int{2, 5},
		First: 1, Read: 5, Heads: 2, Regions: 4, Scalars: []int{20, 62}, Frame: 5, FrameTime: 3, AgainTime: 3, Dilation: 2}}
}

func frames(T, n int, seed uint64) [][]float32 {
	m, _ := NewRandom(Config{Inputs: 1, Dim: n, Heads: 1, Layers: 0, FFN: 1, Window: 0, Outputs: []Head{{"x", 1}}}, seed)
	w := m.net.(*transformerNet).in.W
	out := make([][]float32, T)
	for t := range out {
		out[t] = make([]float32, n)
		for i := range out[t] {
			out[t][i] = w[(i+t*n)%len(w)]
		}
	}
	return out
}

// TestAttentionMatchesTheDenseReference checks the windowed heads against a
// plain float64 attention over the same window, slopes included.
func TestAttentionMatchesTheDenseReference(t *testing.T) {
	cfg := tiny()
	mm, err := NewRandom(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	m := mm.net.(*transformerNet)
	const T = 9
	q, k, v := frames(T, cfg.Dim, 1), frames(T, cfg.Dim, 2), frames(T, cfg.Dim, 3)
	flat := func(f [][]float32) []float32 {
		var out []float32
		for _, r := range f {
			out = append(out, r...)
		}
		return out
	}
	got := make([]float32, T*cfg.Dim)
	m.attend(flat(q), flat(k), flat(v), 0, T, 0, cfg.Window, T, got)
	dh := cfg.Dim / cfg.Heads
	for h := range cfg.Heads {
		for i := range T {
			var weights []float64
			var js []int
			for j := range T {
				if d := i - j; d > cfg.Window || d < -cfg.Window {
					continue
				}
				var s float64
				for d := range dh {
					s += float64(q[i][h*dh+d]) * float64(k[j][h*dh+d])
				}
				s = s/math.Sqrt(float64(dh)) - float64(m.slopes[h])*math.Abs(float64(i-j))
				weights, js = append(weights, s), append(js, j)
			}
			peak := math.Inf(-1)
			for _, w := range weights {
				peak = max(peak, w)
			}
			var sum float64
			for n, w := range weights {
				weights[n] = math.Exp(w - peak)
				sum += weights[n]
			}
			for d := range dh {
				var want float64
				for n, j := range js {
					want += weights[n] / sum * float64(v[j][h*dh+d])
				}
				if diff := math.Abs(float64(got[i*cfg.Dim+h*dh+d]) - want); diff > 1e-5 {
					t.Fatalf("head %d frame %d dim %d: %g, want %g", h, i, d, got[i*cfg.Dim+h*dh+d], want)
				}
			}
		}
	}
}

// TestLayerNormAndGELU pins the two nonlinearities to their definitions, the
// GELU to x·Φ(x) within the worst its polynomials leave over every float32:
// 4.2e-8 below zero and 2.6e-7 above, where a float32 is up to 2.4e-7 apart.
func TestLayerNormAndGELU(t *testing.T) {
	ln := newLayerNorm(4)
	out := make([]float32, 4)
	ln.apply([]float32{1, 2, 3, 4}, 1, out)
	var mean, sq float64
	for _, v := range out {
		mean += float64(v)
		sq += float64(v) * float64(v)
	}
	if math.Abs(mean) > 1e-6 || math.Abs(sq/4-1) > 1e-3 {
		t.Errorf("layer norm gave mean %g, power %g", mean, sq/4)
	}
	for i := -9 << 10; i <= 9<<10; i++ {
		x := float32(i) / 1024
		want, tol := 0.5*float64(x)*(1+math.Erf(float64(x)/math.Sqrt2)), 2.6e-7
		if x < 0 {
			tol = 4.2e-8
		}
		if g := gelu(x); math.Abs(float64(g)-want) > tol {
			t.Errorf("gelu(%g) = %g, exactly %g", x, g, want)
		}
	}
	inf := float32(math.Inf(1))
	if g, l, n := gelu(inf), gelu(-inf), gelu(float32(math.NaN())); g != inf || l != 0 || n == n {
		t.Errorf("gelu of +Inf, -Inf and NaN is %g, %g and %g", g, l, n)
	}
}

// TestWeightFileRoundTrips is the file's contract: what Save wrote, Load reads
// into the same shape with the same outputs, two channels, convolutions,
// towers and a cell net too, as it reads versions 2 and 3, and a damaged file
// is refused.
func TestWeightFileRoundTrips(t *testing.T) {
	cfg := tiny()
	staged := tiny()
	staged.Channels = 2
	staged.Convs = []Conv{{Out: 2, Time: 3, Freq: 3, Pool: 1}, {Out: 3, Time: 1, Freq: 3, Pool: 2}}
	staged.Towers = []Tower{{Layers: 1, Outputs: 1}, {Layers: 2, Outputs: 1}}
	for _, c := range []Config{staged, tinyCells(), cfg} {
		m, _ := NewRandom(c, 11)
		var b bytes.Buffer
		if err := m.Save(&b); err != nil {
			t.Fatal(err)
		}
		back, err := Load(bytes.NewReader(b.Bytes()))
		if err != nil {
			t.Fatal(err)
		}
		if back.Parameters() != m.Parameters() || back.Config().outputSize() != c.outputSize() {
			t.Fatalf("loaded %d parameters of %d", back.Parameters(), m.Parameters())
		}
		in := frames(7, c.Inputs*c.channels(), 5)
		a, got := m.Forward(in), back.Forward(in)
		for i := range a {
			for j := range a[i] {
				if a[i][j] != got[i][j] {
					t.Fatalf("%+v: frame %d output %d differs after the round trip", c.Convs, i, j)
				}
			}
		}
	}
	m, _ := NewRandom(cfg, 11)
	var b bytes.Buffer
	if err := m.Save(&b); err != nil {
		t.Fatal(err)
	}
	back, _ := Load(bytes.NewReader(b.Bytes()))
	c := back.Forward(frames(7, cfg.Inputs, 5))
	if sl := back.Slice(c[0], "b"); len(sl) != 2 || sl[0] != c[0][3] {
		t.Errorf("head b slice %v of %v", sl, c[0])
	}
	f := b.Bytes() // as version 3 wrote it, no kind after the version, and 2, no channels after the inputs
	for v, old := range map[byte][]byte{3: slices.Concat(f[:4], []byte{3, 0, 0, 0}, f[12:len(f)-8]), 2: slices.Concat(f[:4], []byte{2, 0, 0, 0}, f[12:16], f[20:len(f)-8])} {
		h := fnv.New64a()
		h.Write(old)
		m, err := Load(bytes.NewReader(binary.LittleEndian.AppendUint64(old, h.Sum64())))
		if err != nil || m.Config().Channels != 1 {
			t.Fatalf("a version %d file: %v", v, err)
		}
		for i, o := range m.Forward(frames(7, cfg.Inputs, 5)) {
			if !slices.Equal(o, c[i]) {
				t.Fatalf("a version %d file's frame %d: %v, want %v", v, i, o, c[i])
			}
		}
	}
	bad := bytes.Clone(b.Bytes())
	bad[len(bad)/2] ^= 0x55
	if _, err := Load(bytes.NewReader(bad)); err == nil {
		t.Error("a damaged file loaded")
	}
	if _, err := Load(bytes.NewReader(b.Bytes()[:len(b.Bytes())/2])); err == nil {
		t.Error("a truncated file loaded")
	}
}

// TestConvolutionMatchesItsDefinition: each output is the GELU of the bias and
// the kernel over the frames and features around it, zero past the edges, and
// the pool keeps the larger of each pair, the odd feature at the end dropped.
func TestConvolutionMatchesItsDefinition(t *testing.T) {
	const T, F = 6, 5
	m, _ := NewRandom(Config{Inputs: F, Convs: []Conv{{Out: 2, Time: 3, Freq: 3, Pool: 2}}, Dim: 2, Heads: 1, FFN: 1, Outputs: []Head{{"x", 1}}}, 4)
	c := m.net.(*transformerNet).convs[0]
	for o := range c.B {
		c.B[o] = 0.1 * float32(o+1)
	}
	in := frames(T, F, 8)
	x := make([]float32, 0, T*F)
	for _, f := range in {
		x = append(x, f...)
	}
	got := make([]float32, T*c.Out*(F/2))
	c.apply(x, 0, 0, T, T, F, got)
	for tt := range T {
		for o := range c.Out {
			var want [F]float32
			for k := range F {
				s := c.B[o]
				for dt := -1; dt <= 1; dt++ {
					for dk := -1; dk <= 1; dk++ {
						if tt+dt >= 0 && tt+dt < T && k+dk >= 0 && k+dk < F {
							s += c.W[o*9+(dt+1)*3+dk+1] * in[tt+dt][k+dk]
						}
					}
				}
				want[k] = gelu(s)
			}
			for k := range F / 2 {
				w, g := max(want[2*k], want[2*k+1]), got[(tt*c.Out+o)*(F/2)+k]
				if math.Abs(float64(w-g)) > 1e-6 {
					t.Fatalf("frame %d channel %d feature %d: %g, want %g", tt, o, k, g, w)
				}
			}
		}
	}
}

// TestCellNetMatchesItsDefinition: each output is the cell net's layers as
// format.md states them, run in float64 over the whole sequence as PyTorch pads
// it, though Forward runs it in blocks over cores, its biases not zero, a cell
// head of two channels laid one channel's cells after the other's.
func TestCellNetMatchesItsDefinition(t *testing.T) {
	cfg := tinyCells()
	m, _ := NewRandom(cfg, 5)
	n := m.net.(*cellNet)
	for i, tn := range n.tensors() {
		if len(tn.dims) == 1 {
			copy(tn.data, frames(1, len(tn.data), uint64(i))[0])
		}
	}
	const T = 3*cellBlock + 5
	in := frames(T, cfg.Inputs*cfg.Channels, 3)
	got := m.Forward(in)
	c, ch := cfg.Cells, cfg.Channels
	// conv is a layer over x, channels by frames by features, padded with zeros
	conv := func(x [][][]float64, l *conv, padT, padF, stride, dil, width int) [][][]float64 {
		y := make([][][]float64, l.Out)
		for o := range y {
			y[o] = make([][]float64, T)
			for tt := range T {
				y[o][tt] = make([]float64, width)
				for j := range width {
					s := float64(l.B[o])
					for i := range l.in {
						for dt := range l.Time {
							for dk := range l.Freq {
								u, k := tt+dt*dil-padT, j*stride+dk-padF
								if u >= 0 && u < T && k >= 0 && k < len(x[i][u]) {
									s += float64(l.W[((o*l.in+i)*l.Time+dt)*l.Freq+dk]) * x[i][u][k]
								}
							}
						}
					}
					y[o][tt][j] = s
				}
			}
		}
		return y
	}
	g := func(x [][][]float64, add [][][]float64) [][][]float64 {
		for o := range x {
			for tt := range x[o] {
				for j := range x[o][tt] {
					if add != nil {
						x[o][tt][j] += add[o][tt][j]
					}
					x[o][tt][j] = 0.5 * x[o][tt][j] * (1 + math.Erf(x[o][tt][j]/math.Sqrt2))
				}
			}
		}
		return x
	}
	x := make([][][]float64, ch)
	for i := range x {
		x[i] = make([][]float64, T)
		for tt := range T {
			for k := range c.Bins {
				x[i][tt] = append(x[i][tt], float64(in[tt][i*cfg.Inputs+k]))
			}
		}
	}
	reduced := g(conv(x, &n.reduce, 0, 0, 1, 1, c.Bins), nil)
	wide := g(conv(reduced, &n.wide, c.WideTime/2, c.WideFreq/2, 1, 1, c.Bins), nil)
	cells := g(conv(wide, &n.cells, c.CellTime/2, c.CellFreq/2, c.Stride, 1, c.cells()), nil)
	for i := range n.near {
		cells = g(conv(cells, &n.near[i], c.NearTime/2, c.NearFreq/2, 1, 1, c.cells()), cells)
	}
	for i, d := range c.Long {
		cells = g(conv(cells, &n.long[i], d, c.NearFreq/2, 1, d, c.cells()), cells)
	}
	heads := conv(cells, &n.cellout, 0, 0, 1, 1, c.cells())
	pooled := make([][][]float64, c.pooled())
	for i := range pooled {
		pooled[i] = make([][]float64, T)
		for tt := range T {
			if i >= c.Wide*c.Regions {
				pooled[i][tt] = []float64{float64(in[tt][c.Scalars[i-c.Wide*c.Regions]])}
				continue
			}
			r := i % c.Regions
			from, to := r*c.Bins/c.Regions, int(math.Ceil(float64((r+1)*c.Bins)/float64(c.Regions)))
			var s float64
			for k := from; k < to; k++ {
				s += wide[i/c.Regions][tt][k]
			}
			pooled[i][tt] = []float64{s / float64(to-from)}
		}
	}
	f1 := g(conv(pooled, &n.frame1, c.FrameTime/2, 0, 1, 1, 1), nil)
	f2 := g(conv(f1, &n.frame2, c.Dilation*(c.AgainTime/2), 0, 1, c.Dilation, 1), f1)
	for tt := range T {
		want := make([]float64, 0, cfg.outputSize())
		for ch := range n.cellout.Out {
			want = append(want, heads[ch][tt][c.First:c.First+c.Read]...)
		}
		for o := range n.frameout.Out {
			s := float64(n.frameout.B[o])
			for i := range c.Frame {
				s += float64(n.frameout.W[o*c.Frame+i]) * f2[i][tt][0]
			}
			want = append(want, s)
		}
		for i, w := range want {
			if d := math.Abs(float64(got[tt][i]) - w); d > 1e-5*(1+math.Abs(w)) {
				t.Fatalf("frame %d output %d: %g, want %g", tt, i, got[tt][i], w)
			}
		}
	}
}

// TestKernelsMatchThePlainLoops: at every length the vector kernels split
// differently, and from unaligned starts, dot and axpy agree with the plain
// loops to float32 rounding and a row's GELU with gelu bit for bit, a NaN a
// NaN, and axpy and the row write nothing outside their slices.
func TestKernelsMatchThePlainLoops(t *testing.T) {
	const u = 0x1p-24
	a, b := frames(1, 1100, 6)[0], frames(1, 1100, 7)[0]
	s, w := a[1099], a[1098]
	wide := slices.Clone(a) // past the GELU's split and its end either side of zero
	for i := range wide {
		wide[i] *= 4
	}
	wide[7], wide[8], wide[40] = float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))
	for n := range 1030 {
		if n > 100 && n != 255 && n != 256 && n != 1023 {
			continue
		}
		for off := range 4 {
			x, y := a[off:off+n], b[off:off+n]
			got, want, mag := dot(s, x, y), dotGeneric(s, x, y), math.Abs(float64(s))
			for i := range x {
				mag += math.Abs(float64(x[i]) * float64(y[i]))
			}
			if d := math.Abs(float64(got - want)); d > 2*float64(n+1)*u*mag {
				t.Fatalf("dot of %d from %d: %g, the loop %g", n, off, got, want)
			}
			ys, yp := slices.Clone(b[:off+n+1]), slices.Clone(b[:off+n+1])
			axpy(w, x, ys[off:off+n])
			axpyGeneric(w, x, yp[off:off+n])
			for i := range ys {
				tol := 0.0
				if i >= off && i < off+n {
					tol = 2 * u * (math.Abs(float64(b[i])) + math.Abs(float64(w)*float64(a[i])))
				}
				if math.Abs(float64(ys[i]-yp[i])) > tol {
					t.Fatalf("axpy of %d from %d, element %d: %g, the loop %g", n, off, i, ys[i], yp[i])
				}
			}
			g := slices.Clone(wide[:off+n+1])
			geluRow(g[off : off+n])
			for i, v := range wide[:off+n+1] {
				if i >= off && i < off+n {
					v = gelu(v)
				}
				if math.Float32bits(g[i]) != math.Float32bits(v) && (g[i] == g[i] || v == v) {
					t.Fatalf("the GELU of %d from %d, element %d: %g, gelu's %g", n, off, i, g[i], v)
				}
			}
		}
	}
}

// TestForwardShapesAndDeterminism: a frame per input, a head's worth per frame,
// the same twice.
func TestForwardShapesAndDeterminism(t *testing.T) {
	cfg := tiny()
	m, _ := NewRandom(cfg, 3)
	in := frames(20, cfg.Inputs, 9)
	a, b := m.Forward(in), m.Forward(in)
	if len(a) != 20 || len(a[0]) != 5 {
		t.Fatalf("shape %d × %d", len(a), len(a[0]))
	}
	for i := range a {
		for j := range a[i] {
			if a[i][j] != b[i][j] || math.IsNaN(float64(a[i][j])) {
				t.Fatalf("frame %d output %d: %g then %g", i, j, a[i][j], b[i][j])
			}
		}
	}
	if m.Forward(nil) != nil {
		t.Error("no frames gave frames")
	}
	if _, err := newModel(Config{Inputs: 1, Dim: 6, Heads: 4, FFN: 1, Outputs: []Head{{"x", 1}}}); err == nil {
		t.Error("a width no multiple of its heads passed")
	}
}

// streamed is in through a stream of m waiting ahead frames, cut into blocks of
// block's sizes; each output must come once wait frames follow its own
func streamed(t *testing.T, m *Model, in [][]float32, ahead, wait int, block func() int) [][]float32 {
	s := m.Stream(ahead)
	var out [][]float32
	for at := 0; at < len(in); {
		k := min(block(), len(in)-at)
		out = append(out, s.Write(in[at:at+k])...)
		if at += k; len(out) != max(at-wait, 0) {
			t.Fatalf("%d outputs after %d frames, waiting %d", len(out), at, wait)
		}
	}
	return append(out, s.Close()...)
}

// TestAStreamGivesForwardsOutputs: a stream that waits a model's reach gives
// Forward's outputs bit for bit however its frames are cut, each once the frames
// it reads are in: a transformer's with convolutions and towers, and a cell
// net's past Forward's own blocks, their biases not zero.
func TestAStreamGivesForwardsOutputs(t *testing.T) {
	staged := tiny()
	staged.Channels = 2
	staged.Convs = []Conv{{Out: 2, Time: 3, Freq: 3, Pool: 1}, {Out: 3, Time: 1, Freq: 3, Pool: 2}}
	staged.Towers = []Tower{{Layers: 1, Outputs: 1}, {Layers: 2, Outputs: 1}}
	rng := rand.New(rand.NewPCG(3, 4))
	for ci, c := range []Config{tiny(), staged, tinyCells()} {
		m, _ := NewRandom(c, 11)
		for i, tn := range m.net.tensors() {
			if len(tn.dims) == 1 {
				copy(tn.data, frames(1, len(tn.data), uint64(i))[0])
			}
		}
		r := m.Reach()
		if want := []int{6, 7, 11}[ci]; r != want {
			t.Fatalf("model %d reaches %d frames, want %d", ci, r, want)
		}
		for _, T := range []int{1, r, 3*cellBlock + 5} {
			in := frames(T, c.Inputs*c.channels(), 5)
			want := m.Forward(in)
			for cut, block := range []func() int{func() int { return 1 }, func() int { return T }, func() int { return 1 + rng.IntN(3*r) }} {
				got := streamed(t, m, in, r, r, block)
				if len(got) != T {
					t.Fatalf("model %d: %d outputs of %d frames", ci, len(got), T)
				}
				for i := range got {
					if !slices.Equal(got[i], want[i]) {
						t.Fatalf("model %d, %d frames, cut %d: frame %d is %v, Forward's %v", ci, T, cut, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestAStreamCutShortAttendsOnlyTheFramesItWaits: under its reach a stream gives
// each output once ahead frames follow its own, the same however the frames are
// cut, and a single block's is Forward's over the frames up to the last waited.
func TestAStreamCutShortAttendsOnlyTheFramesItWaits(t *testing.T) {
	one := tiny()
	one.Layers = 1
	m, _ := NewRandom(one, 3)
	in := frames(20, one.Inputs, 9)
	for i, o := range streamed(t, m, in, 1, 1, func() int { return 3 }) {
		if want := m.Forward(in[:min(i+2, len(in))])[i]; !slices.Equal(o, want) {
			t.Fatalf("frame %d is %v, Forward's over the frames up to the next %v", i, o, want)
		}
	}
	staged := tiny()
	staged.Convs = []Conv{{Out: 2, Time: 3, Freq: 3, Pool: 1}}
	staged.Towers = []Tower{{Layers: 1, Outputs: 1}, {Layers: 2, Outputs: 1}}
	m, _ = NewRandom(staged, 3)
	in = frames(40, staged.Inputs, 9)
	a, b := streamed(t, m, in, 4, 4, func() int { return 1 }), streamed(t, m, in, 4, 4, func() int { return 7 })
	for i := range a {
		if !slices.Equal(a[i], b[i]) {
			t.Fatalf("frame %d differs with the cut: %v and %v", i, a[i], b[i])
		}
	}
}

// BenchmarkForward is the baseline for performance work: 3.3 million weights,
// width 256 in eight heads, four layers, a feed-forward of 1024 and a window of
// 64 frames either side, run over 512 frames of 230 inputs into 318 outputs, on
// GOMAXPROCS cores (-cpu sets them).
func BenchmarkForward(b *testing.B) {
	m, in := benchTransformer(b)
	bench(b, len(in), func() { m.Forward(in) })
}

func benchTransformer(b *testing.B) (*Model, [][]float32) {
	m, err := NewRandom(Config{Inputs: 230, Dim: 256, Heads: 8, Layers: 4, FFN: 1024, Window: 64, Outputs: []Head{{"y", 318}}}, 1)
	if err != nil {
		b.Fatal(err)
	}
	return m, frames(512, m.cfg.Inputs, 2)
}

// bench times f, which runs T frames, and reports them a second
func bench(b *testing.B, T int, f func()) {
	b.ReportAllocs()
	for b.Loop() {
		f()
	}
	b.ReportMetric(float64(T*b.N)/b.Elapsed().Seconds(), "frames/s")
}

// BenchmarkKernels times dot and axpy at the width, each against its plain loop.
func BenchmarkKernels(b *testing.B) {
	x, y := frames(1, 256, 1)[0], frames(1, 256, 2)[0]
	var s float32
	for _, k := range []struct {
		name string
		f    func()
	}{
		{"dot", func() { s = dot(s, x, y) }},
		{"dot-loop", func() { s = dotGeneric(s, x, y) }},
		{"axpy", func() { axpy(1e-9, x, y) }},
		{"axpy-loop", func() { axpyGeneric(1e-9, x, y) }},
	} {
		b.Run(k.name, func(b *testing.B) {
			for b.Loop() {
				k.f()
			}
		})
	}
}

// BenchmarkCellNet is the cell net's: the rip's notes model of 16 channels of 214
// inputs, 71 cells and 318 outputs, over 2584 frames, 30 s, without and with the
// long convolutions that take it from 169,862 weights to 197,606.
func BenchmarkCellNet(b *testing.B) {
	for _, c := range []struct {
		name string
		long []int
	}{{"near", nil}, {"long", []int{3, 9, 27}}} {
		b.Run(c.name, func(b *testing.B) {
			m, in := benchCellNet(b, c.long)
			bench(b, len(in), func() { m.Forward(in) })
		})
	}
}

func benchCellNet(b *testing.B, long []int) (*Model, [][]float32) {
	cfg := Config{Inputs: 214, Channels: 16, Outputs: []Head{{"pitch", 69}, {"strike", 69}, {"release", 69}, {"velocity", 69}, {"frame", 42}}, Cells: &Cells{
		Bins: 213, Reduce: 8, Wide: 16, WideTime: 3, WideFreq: 39, Cell: 32, CellTime: 7, CellFreq: 3, Stride: 3, Depth: 1, NearTime: 7, NearFreq: 3, Long: long,
		First: 1, Read: 69, Heads: 4, Regions: 16, Scalars: []int{213, 1925}, Frame: 64, FrameTime: 5, AgainTime: 9, Dilation: 2}}
	m, err := NewRandom(cfg, 1)
	if err != nil {
		b.Fatal(err)
	}
	return m, frames(2584, cfg.Inputs*cfg.Channels, 2)
}

// BenchmarkStream is the two benchmarks' frames through a stream, 43 a write
// as the rip's half second: the transformer waiting its reach and waiting 46
// frames, and the cell net with its long convolutions.
func BenchmarkStream(b *testing.B) {
	t, tin := benchTransformer(b)
	c, cin := benchCellNet(b, []int{3, 9, 27})
	for _, s := range []struct {
		name  string
		m     *Model
		in    [][]float32
		ahead int
	}{{"transformer", t, tin, t.Reach()}, {"transformer-46", t, tin, 46}, {"cells", c, cin, c.Reach()}} {
		b.Run(s.name, func(b *testing.B) {
			bench(b, len(s.in), func() {
				st := s.m.Stream(s.ahead)
				for at := 0; at < len(s.in); at += 43 {
					st.Write(s.in[at:min(at+43, len(s.in))])
				}
				st.Close()
			})
		})
	}
}
