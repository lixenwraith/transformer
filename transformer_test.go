package transformer

import (
	"bytes"
	"encoding/binary"
	"hash/fnv"
	"math"
	"slices"
	"testing"
)

func tiny() Config {
	return Config{Inputs: 5, Dim: 8, Heads: 2, Layers: 2, FFN: 16, Window: 3,
		Outputs: []Head{{"a", 3}, {"b", 2}}}
}

func frames(T, n int, seed uint64) [][]float32 {
	m, _ := NewRandom(Config{Inputs: 1, Dim: n, Heads: 1, Layers: 0, FFN: 1, Window: 0, Outputs: []Head{{"x", 1}}}, seed)
	out := make([][]float32, T)
	for t := range out {
		out[t] = make([]float32, n)
		for i := range out[t] {
			out[t][i] = m.in.W[(i+t*n)%len(m.in.W)]
		}
	}
	return out
}

// TestAttentionMatchesTheDenseReference checks the windowed heads against a
// plain float64 attention over the same window, slopes included.
func TestAttentionMatchesTheDenseReference(t *testing.T) {
	cfg := tiny()
	m, err := NewRandom(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
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
	m.attend(flat(q), flat(k), flat(v), T, got)
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

// TestLayerNormAndGELU pins the two nonlinearities to their definitions.
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
	if g := gelu(1); math.Abs(float64(g)-0.8413447) > 1e-6 {
		t.Errorf("gelu(1) = %g", g)
	}
	if g := gelu(-10); g != 0 && math.Abs(float64(g)) > 1e-6 {
		t.Errorf("gelu(-10) = %g", g)
	}
}

// TestWeightFileRoundTrips is the file's contract: what Save wrote, Load reads
// into the same shape with the same outputs, two channels, convolutions and
// towers too, as it reads a version 2 file, and a damaged file is refused.
func TestWeightFileRoundTrips(t *testing.T) {
	cfg := tiny()
	staged := tiny()
	staged.Channels = 2
	staged.Convs = []Conv{{Out: 2, Time: 3, Freq: 3, Pool: 1}, {Out: 3, Time: 1, Freq: 3, Pool: 2}}
	staged.Towers = []Tower{{Layers: 1, Outputs: 1}, {Layers: 2, Outputs: 1}}
	for _, c := range []Config{staged, cfg} {
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
	f := b.Bytes() // as version 2 wrote it: no channels after the inputs
	v2 := slices.Concat(f[:4], []byte{2, 0, 0, 0}, f[8:12], f[16:len(f)-8])
	h := fnv.New64a()
	h.Write(v2)
	old, err := Load(bytes.NewReader(binary.LittleEndian.AppendUint64(v2, h.Sum64())))
	if err != nil || old.Config().Channels != 1 {
		t.Fatalf("a version 2 file: %v", err)
	}
	for i, o := range old.Forward(frames(7, cfg.Inputs, 5)) {
		if !slices.Equal(o, c[i]) {
			t.Fatalf("a version 2 file's frame %d: %v, want %v", i, o, c[i])
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
	c := m.convs[0]
	for o := range c.B {
		c.B[o] = 0.1 * float32(o+1)
	}
	in := frames(T, F, 8)
	x := make([]float32, 0, T*F)
	for _, f := range in {
		x = append(x, f...)
	}
	got := c.apply(x, T, F)
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

// TestKernelsMatchThePlainLoops: at every length the vector kernels split
// differently, and from unaligned starts, dot and axpy agree with the plain
// loops to float32 rounding, and axpy writes nothing outside its slice.
func TestKernelsMatchThePlainLoops(t *testing.T) {
	const u = 0x1p-24
	a, b := frames(1, 1100, 6)[0], frames(1, 1100, 7)[0]
	s, w := a[1099], a[1098]
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

// BenchmarkForward is the baseline for performance work: 3.3 million weights,
// width 256 in eight heads, four layers, a feed-forward of 1024 and a window of
// 64 frames either side, run over 512 frames of 230 inputs into 318 outputs, on
// GOMAXPROCS cores (-cpu sets them).
func BenchmarkForward(b *testing.B) {
	const T = 512
	m, err := NewRandom(Config{Inputs: 230, Dim: 256, Heads: 8, Layers: 4, FFN: 1024, Window: 64, Outputs: []Head{{"y", 318}}}, 1)
	if err != nil {
		b.Fatal(err)
	}
	in := frames(T, m.cfg.Inputs, 2)
	b.ReportAllocs()
	for b.Loop() {
		m.Forward(in)
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
