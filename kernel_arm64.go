package transformer

// dot is s plus a·b, a and b of one length
func dot(s float32, a, b []float32) float32 {
	if len(a) == 0 {
		return s
	}
	_ = b[len(a)-1]
	return dotNEON(s, &a[0], &b[0], len(a))
}

// axpy adds w·x into y, x and y of one length
func axpy(w float32, x, y []float32) {
	if len(x) == 0 {
		return
	}
	_ = y[len(x)-1]
	axpyNEON(w, &x[0], &y[0], len(x))
}

// Go's arm64 targets ARMv8-A, where NEON is mandatory: no detection needed

//go:noescape
func dotNEON(s float32, a, b *float32, n int) float32

//go:noescape
func axpyNEON(w float32, x, y *float32, n int)
