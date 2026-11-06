package transformer

// avx2FMA is whether the CPU has AVX2 and FMA and the OS saves the YMM registers;
// without them the plain loops run
var avx2FMA = hasAVX2FMA()

// dot is s plus a·b, a and b of one length
func dot(s float32, a, b []float32) float32 {
	if !avx2FMA || len(a) == 0 {
		return dotGeneric(s, a, b)
	}
	_ = b[len(a)-1]
	return dotAVX2(s, &a[0], &b[0], len(a))
}

// axpy adds w·x into y, x and y of one length
func axpy(w float32, x, y []float32) {
	if !avx2FMA || len(x) == 0 {
		axpyGeneric(w, x, y)
		return
	}
	_ = y[len(x)-1]
	axpyAVX2(w, &x[0], &y[0], len(x))
}

func hasAVX2FMA() bool {
	const fma, osxsave, avx, avx2, xmmYmm = 1 << 12, 1 << 27, 1 << 28, 1 << 5, 6
	if top, _, _, _ := cpuid(0, 0); top < 7 {
		return false
	}
	_, _, c, _ := cpuid(1, 0)
	if c&(fma|osxsave|avx) != fma|osxsave|avx || xgetbv()&xmmYmm != xmmYmm {
		return false
	}
	_, b, _, _ := cpuid(7, 0)
	return b&avx2 != 0
}

func cpuid(leaf, sub uint32) (a, b, c, d uint32)

func xgetbv() uint32

//go:noescape
func dotAVX2(s float32, a, b *float32, n int) float32

//go:noescape
func axpyAVX2(w float32, x, y *float32, n int)
