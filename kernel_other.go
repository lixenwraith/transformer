//go:build !amd64 && !arm64

package transformer

// dot is s plus a·b, a and b of one length
func dot(s float32, a, b []float32) float32 { return dotGeneric(s, a, b) }

// axpy adds w·x into y, x and y of one length
func axpy(w float32, x, y []float32) { axpyGeneric(w, x, y) }
