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
const (
	fileMagic   = "VIFT"
	fileVersion = 2
	maxName     = 64
	maxTensor   = 1 << 28 // floats in one tensor
)

type tensor struct {
	name string
	dims []int
	data []float32
}

// tensors names every weight in the order the file holds them
func (m *Model) tensors() []tensor {
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
	bw.WriteString(fileMagic)
	put(uint32(fileVersion))
	cfg := m.cfg
	put(uint32(cfg.Inputs))
	put(uint32(cfg.Dim))
	put(uint32(cfg.Heads))
	put(uint32(cfg.Layers))
	put(uint32(cfg.FFN))
	put(uint32(cfg.Window))
	put(uint32(len(cfg.Convs)))
	for _, c := range cfg.Convs {
		put([4]uint32{uint32(c.Out), uint32(c.Time), uint32(c.Freq), uint32(c.Pool)})
	}
	put(uint32(len(cfg.Towers)))
	for _, t := range cfg.Towers {
		put([2]uint32{uint32(t.Layers), uint32(t.Outputs)})
	}
	put(uint32(len(cfg.Outputs)))
	for _, o := range cfg.Outputs {
		put(uint16(len(o.Name)))
		bw.WriteString(o.Name)
		put(uint32(o.Size))
	}
	ts := m.tensors()
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
	if err := get(&u32); err != nil || u32 != fileVersion {
		return nil, fmt.Errorf("transformer: weight file version %d, want %d", u32, fileVersion)
	}
	var cfg Config
	for _, p := range []*int{&cfg.Inputs, &cfg.Dim, &cfg.Heads, &cfg.Layers, &cfg.FFN, &cfg.Window} {
		if err := get(&u32); err != nil {
			return nil, err
		}
		*p = int(u32)
	}
	if err := get(&u32); err != nil || u32 > 16 {
		return nil, errors.New("transformer: convolutions")
	}
	for range u32 {
		var c [4]uint32
		if err := get(&c); err != nil {
			return nil, err
		}
		cfg.Convs = append(cfg.Convs, Conv{Out: int(c[0]), Time: int(c[1]), Freq: int(c[2]), Pool: int(c[3])})
	}
	if err := get(&u32); err != nil || u32 > 16 {
		return nil, errors.New("transformer: towers")
	}
	for range u32 {
		var t [2]uint32
		if err := get(&t); err != nil {
			return nil, err
		}
		cfg.Towers = append(cfg.Towers, Tower{Layers: int(t[0]), Outputs: int(t[1])})
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
	want := m.tensors()
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
