package transformer

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"math"
)

// The weight file: a magic and version, the Config, then every tensor by name
// with its shape and float32 data, little-endian, and the FNV-1a 64 of all of it.
// doc/format.md is its contract; a change to the bytes bumps fileVersion there.
// Load reads oldestVersion on: a trained file keeps the version it was written in.
const (
	fileMagic     = "VIFT"
	fileVersion   = 5
	oldestVersion = 2
	maxName       = 64
	maxTensor     = 1 << 28 // floats in one tensor
)

type tensor struct {
	name string
	dims []int
	data []float32
}

func (m *transformerNet) tensors() []tensor {
	lin := func(name string, l *linear) []tensor {
		return []tensor{{name + ".weight", []int{l.Out, l.In}, l.W}, {name + ".bias", []int{l.Out}, l.B}}
	}
	norm := func(name string, ln *layerNorm) []tensor {
		return []tensor{{name + ".weight", []int{len(ln.G)}, ln.G}, {name + ".bias", []int{len(ln.B)}, ln.B}}
	}
	var ts []tensor
	for i := range m.convs {
		c := &m.convs[i]
		p := fmt.Sprintf("convs.%d.", i)
		ts = append(ts, tensor{p + "weight", []int{c.Out, c.in, c.Time, c.Freq}, c.W}, tensor{p + "bias", []int{c.Out}, c.B})
	}
	ts = append(ts, lin("in", &m.in)...)
	for ti := range m.towers {
		tw := &m.towers[ti]
		pt := "" // a model without Towers names its one stack unprefixed
		if len(m.cfg.Towers) > 0 {
			pt = fmt.Sprintf("towers.%d.", ti)
		}
		for i := range tw.layers {
			l := &tw.layers[i]
			p := pt + fmt.Sprintf("layers.%d.", i)
			ts = append(ts, norm(p+"ln1", &l.ln1)...)
			ts = append(ts, lin(p+"q", &l.q)...)
			ts = append(ts, lin(p+"k", &l.k)...)
			ts = append(ts, lin(p+"v", &l.v)...)
			ts = append(ts, lin(p+"o", &l.o)...)
			ts = append(ts, norm(p+"ln2", &l.ln2)...)
			ts = append(ts, lin(p+"ff1", &l.ff1)...)
			ts = append(ts, lin(p+"ff2", &l.ff2)...)
		}
		ts = append(ts, norm(pt+"ln", &tw.ln)...)
		ts = append(ts, lin(pt+"out", &tw.out)...)
	}
	return ts
}

// Save writes the model as a weight file.
func (m *Model) Save(w io.Writer) error {
	h := fnv.New64a()
	bw := bufio.NewWriter(io.MultiWriter(w, h))
	le := binary.LittleEndian
	put := func(v any) {
		binary.Write(bw, le, v)
	}
	ints := func(vs ...int) {
		for _, v := range vs {
			put(uint32(v))
		}
	}
	bw.WriteString(fileMagic)
	cfg := m.cfg
	ints(fileVersion, cfg.kind(), cfg.Inputs, cfg.channels())
	kinds[cfg.kind()].save(ints, cfg)
	put(uint32(len(cfg.Outputs)))
	for _, o := range cfg.Outputs {
		put(uint16(len(o.Name)))
		bw.WriteString(o.Name)
		put(uint32(o.Size))
	}
	ts := m.net.tensors()
	put(uint32(len(ts)))
	for _, t := range ts {
		put(uint16(len(t.name)))
		bw.WriteString(t.name)
		put(uint8(len(t.dims)))
		for _, d := range t.dims {
			put(uint32(d))
		}
		for _, f := range t.data {
			put(math.Float32bits(f))
		}
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	return binary.Write(w, le, h.Sum64())
}

// Load reads a weight file into a model of the Config it names.
func Load(r io.Reader) (*Model, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	le := binary.LittleEndian
	if len(data) < 8 {
		return nil, errors.New("transformer: weight file cut short")
	}
	h := fnv.New64a()
	h.Write(data[:len(data)-8])
	if le.Uint64(data[len(data)-8:]) != h.Sum64() {
		return nil, errors.New("transformer: weight file checksum")
	}
	br := bytes.NewReader(data[:len(data)-8])
	var u32 uint32
	get := func(v any) error { return binary.Read(br, le, v) }
	magic := make([]byte, 4)
	if _, err := io.ReadFull(br, magic); err != nil || string(magic) != fileMagic {
		return nil, errors.New("transformer: not a weight file")
	}
	if err := get(&u32); err != nil || u32 < oldestVersion || u32 > fileVersion {
		return nil, fmt.Errorf("transformer: weight file version %d, want %d to %d", u32, oldestVersion, fileVersion)
	}
	version := u32
	geti := func() (int, error) {
		err := get(&u32)
		return int(u32), err
	}
	cfg := Config{Channels: 1} // a version 2 file has one and does not say so
	kind := kindTransformer
	sizes := []*int{&kind, &cfg.Inputs, &cfg.Channels}
	if version < 4 {
		sizes = sizes[1:]
	}
	if version == 2 {
		sizes = sizes[:1]
	}
	if err := fill(geti, sizes...); err != nil {
		return nil, err
	}
	if kind >= len(kinds) || kind < 0 {
		return nil, fmt.Errorf("transformer: model kind %d", kind)
	}
	if cfg.Channels < 1 {
		return nil, errors.New("transformer: no input channel")
	}
	if err := kinds[kind].load(geti, int(version), &cfg); err != nil {
		return nil, err
	}
	if err := get(&u32); err != nil || u32 > 64 {
		return nil, errors.New("transformer: output heads")
	}
	readName := func() (string, error) {
		var n uint16
		if err := get(&n); err != nil || n == 0 || n > maxName {
			return "", errors.New("transformer: a name's length")
		}
		b := make([]byte, n)
		_, err := io.ReadFull(br, b)
		return string(b), err
	}
	for range u32 {
		name, err := readName()
		if err != nil {
			return nil, err
		}
		var size uint32
		if err := get(&size); err != nil {
			return nil, err
		}
		cfg.Outputs = append(cfg.Outputs, Head{Name: name, Size: int(size)})
	}
	m, err := newModel(cfg)
	if err != nil {
		return nil, err
	}
	want := m.net.tensors()
	if err := get(&u32); err != nil || int(u32) != len(want) {
		return nil, fmt.Errorf("transformer: %d tensors, the shape wants %d", u32, len(want))
	}
	for _, t := range want {
		name, err := readName()
		if err != nil {
			return nil, err
		}
		var nd uint8
		if err := get(&nd); err != nil || int(nd) != len(t.dims) || name != t.name {
			return nil, fmt.Errorf("transformer: tensor %q of %d dims, want %q of %d", name, nd, t.name, len(t.dims))
		}
		n := 1
		for i := range t.dims {
			if err := get(&u32); err != nil || int(u32) != t.dims[i] {
				return nil, fmt.Errorf("transformer: tensor %q dim %d is %d, want %d", name, i, u32, t.dims[i])
			}
			n *= int(u32)
		}
		if n != len(t.data) || n > maxTensor {
			return nil, fmt.Errorf("transformer: tensor %q holds %d floats", name, n)
		}
		buf := make([]byte, 4*n)
		if _, err := io.ReadFull(br, buf); err != nil {
			return nil, fmt.Errorf("transformer: tensor %q: %w", name, err)
		}
		for i := range t.data {
			t.data[i] = math.Float32frombits(le.Uint32(buf[4*i:]))
		}
	}
	if br.Len() != 0 {
		return nil, errors.New("transformer: weight file has bytes past its tensors")
	}
	return m, nil
}

// fill reads the next sizes of the file into ps in turn
func fill(get func() (int, error), ps ...*int) error {
	for _, p := range ps {
		v, err := get()
		if err != nil {
			return err
		}
		*p = v
	}
	return nil
}

func saveTransformer(put func(...int), c Config) {
	put(c.Dim, c.Heads, c.Layers, c.FFN, c.Window, len(c.Convs))
	for _, cv := range c.Convs {
		put(cv.Out, cv.Time, cv.Freq, cv.Pool)
	}
	put(len(c.Towers))
	for _, t := range c.Towers {
		put(t.Layers, t.Outputs)
	}
}

func loadTransformer(get func() (int, error), _ int, c *Config) error {
	var n int
	if err := fill(get, &c.Dim, &c.Heads, &c.Layers, &c.FFN, &c.Window, &n); err != nil || n < 0 || n > 16 {
		return errors.New("transformer: convolutions")
	}
	for range n {
		var cv Conv
		if err := fill(get, &cv.Out, &cv.Time, &cv.Freq, &cv.Pool); err != nil {
			return err
		}
		c.Convs = append(c.Convs, cv)
	}
	if err := fill(get, &n); err != nil || n < 0 || n > 16 {
		return errors.New("transformer: towers")
	}
	for range n {
		var t Tower
		if err := fill(get, &t.Layers, &t.Outputs); err != nil {
			return err
		}
		c.Towers = append(c.Towers, t)
	}
	return nil
}

// sizes are a cell net's sizes in the order the file holds them, Scalars after
func (c *Cells) sizes() []*int {
	return []*int{&c.Bins, &c.Reduce, &c.Wide, &c.WideTime, &c.WideFreq, &c.Cell, &c.CellTime, &c.CellFreq, &c.Stride,
		&c.Depth, &c.NearTime, &c.NearFreq, &c.First, &c.Read, &c.Heads, &c.Regions, &c.Frame, &c.FrameTime, &c.AgainTime, &c.Dilation}
}

// these bound a cell net's lists before its Config is checked
const (
	maxScalars = 64
	maxLong    = 16
)

func saveCells(put func(...int), c Config) {
	for _, p := range c.Cells.sizes() {
		put(*p)
	}
	put(len(c.Cells.Scalars))
	put(c.Cells.Scalars...)
	put(len(c.Cells.Long))
	put(c.Cells.Long...)
}

func loadCells(get func() (int, error), version int, c *Config) error {
	c.Cells = &Cells{}
	var n int
	if err := fill(get, append(c.Cells.sizes(), &n)...); err != nil || n < 0 || n > maxScalars {
		return errors.New("transformer: a cell net's sizes")
	}
	c.Cells.Scalars = make([]int, n)
	for i := range c.Cells.Scalars {
		if err := fill(get, &c.Cells.Scalars[i]); err != nil {
			return err
		}
	}
	if version < 5 {
		return nil
	}
	if err := fill(get, &n); err != nil || n < 0 || n > maxLong {
		return errors.New("transformer: a cell net's long convolutions")
	}
	c.Cells.Long = make([]int, n)
	for i := range c.Cells.Long {
		if err := fill(get, &c.Cells.Long[i]); err != nil {
			return err
		}
	}
	return nil
}
