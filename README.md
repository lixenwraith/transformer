# transformer

Inference for small transformer models over sequences of frames, in Go: float32,
the standard library alone, no cgo, on every platform Go builds for. A model
reads T frames of features and writes T output vectors, each its named output
heads laid end to end. A cell net, convolutions alone with the same weights at
every cell along the features, is the other kind of model it runs. The module
runs models and does not train them: a model comes from any trainer that writes
its weight file ([doc/format.md](doc/format.md)).

---

## Package layout

- `transformer`, standard library only, is the whole module:
  - `config.go`: the shape (`Config`, `Head`, `Conv`, `Tower`) and its checks;
  - `transformer.go`: the layers, `Model`, the kinds of model, `NewRandom` and
    `Forward`;
  - `cells.go`: the cell net's layers, its pass over a sequence in blocks, and
    the convolution over a block's rows it runs each layer with;
  - `kernel_*.go`, `kernel_*.s`: the inner products, on AVX2 and FMA on amd64
    when the CPU has them, NEON on arm64, and the plain loops elsewhere;
  - `weights.go`: the weight file, `Save` and `Load`.
- [doc/format.md](doc/format.md): the weight file's contract, enough for a
  trainer in any language to write one.
- [doc/todo.md](doc/todo.md): where the module may grow, and the seam each change
  would use.
- [doc/agent-notes.md](doc/agent-notes.md): the float32 check against a trainer,
  and what earlier work learned.

---

## Core types

```go
type Config struct {
    Inputs   int     // features in a channel of a frame
    Channels int     // planes of Inputs a frame lays end to end, none for one
    Convs    []Conv  // convolutions before the input layer, none for a frame as it comes
    Dim      int     // the width
    Heads    int     // attention heads; Dim is a multiple of them
    Layers   int     // blocks of the one stack, when there are no Towers
    FFN      int     // the feed-forward's width
    Window   int     // frames attended on either side
    Towers   []Tower // stacks over the shared input layer, none for one stack of Layers
    Outputs  []Head  // the output vector's named slices, in order
    Cells    *Cells  // a cell net in place of the transformer, nil for none
}

type Conv struct{ Out, Time, Freq, Pool int } // channels out, an odd kernel of frames by features, a max-pool over features
type Tower struct{ Layers, Outputs int }      // its blocks, and how many of the next heads it computes
type Head struct {
    Name string
    Size int
}

type Cells struct { // a cell net's sizes, doc/format.md's header in its order
    Bins, Reduce                    int
    Wide, WideTime, WideFreq        int
    Cell, CellTime, CellFreq        int
    Stride                          int
    Depth, NearTime, NearFreq       int
    First, Read, Heads, Regions     int
    Scalars                         []int
    Frame, FrameTime                int
    AgainTime, Dilation             int
}
```

```go
func Load(r io.Reader) (*Model, error)                  // a weight file, checked whole
func NewRandom(cfg Config, seed uint64) (*Model, error) // small random weights, for tests and a trainer's shape check

func (m *Model) Forward(frames [][]float32) [][]float32 // one output vector per frame
func (m *Model) Slice(out []float32, head string) []float32
func (m *Model) Save(w io.Writer) error
func (m *Model) Config() Config
func (m *Model) Parameters() int
```

---

## The model

- **Channels**: a frame may lay several planes of `Inputs` features end to end,
  such as two views of one spectrum; the first convolution reads them as its
  channels, or the input layer takes the frame whole.
- **Convolutions**, optional, over the frames as an image, frames by features:
  each a kernel centred on a frame and a feature, the edges zero, a GELU, then a
  max-pool over the features.
- **An input layer**, linear, from a frame's features to the width.
- **Blocks**, pre-norm: multi-head attention over a window of frames on either
  side, then an exact-GELU feed-forward, each added back to its input.
  Attention carries ALiBi biases, a slope per head on the frame distance, in
  place of position embeddings, so a model runs over a sequence of any length.
- **Stacks**: one of `Layers` blocks computing every head, or towers, each a
  stack of its own over the shared input layer computing the next run of heads.
  Each stack ends in a norm and an output layer.
- **Heads**: the scores are raw; the caller applies its own activation.

A **cell net** (`Config.Cells`) reads the same frames as an image and has no
attention: a 1×1 convolution and a wide one over the features, one to cells a
`Stride` of features apart, and `Depth` over the cells each added back, all GELU,
give each cell its heads through a 1×1 convolution, the same weights at every
cell, a head of several channels laying one channel's cells after another's;
the frame's heads come from the wide channels averaged to `Regions`, the
`Scalars` beside them, through two convolutions over the frames and a linear
layer. Its output is the cell heads, each its cells in order, then the frame
heads. `Forward` runs it in blocks of frames across the cores, each block
computing again the frames around it that its outputs reach.

[doc/format.md](doc/format.md) states the computation exactly, for a trainer to
match.

---

## The weight file

One file is one model: the magic `VIFT`, the format's version, the model's kind,
the `Config`, every tensor by name with its dims and float32 data, little-endian, and an
FNV-1a 64 checksum over all of it. `Load` checks the checksum, the version, the
shape and each tensor's name and dims, and refuses a file that disagrees in any
of them. A change to the bytes is a new version, and `Load` keeps reading the
versions trained files carry.

---

## Worked example

A random model saved and loaded again, run over a sequence, with one head read
per frame. A trained model is the same from `Load` on.

```go
package main

import (
    "bytes"
    "fmt"
    "math"

    "github.com/lixenwraith/transformer"
)

func main() {
    cfg := transformer.Config{
        Inputs: 16, Dim: 32, Heads: 4, Layers: 2, FFN: 64, Window: 8,
        Outputs: []transformer.Head{{Name: "events", Size: 3}, {Name: "level", Size: 1}},
    }
    m, err := transformer.NewRandom(cfg, 1)
    if err != nil {
        panic(err)
    }
    var file bytes.Buffer
    if err := m.Save(&file); err != nil {
        panic(err)
    }
    m, err = transformer.Load(&file)
    if err != nil {
        panic(err)
    }

    frames := make([][]float32, 100)
    for t := range frames {
        frames[t] = make([]float32, cfg.Inputs)
        for i := range frames[t] {
            frames[t][i] = float32(math.Sin(float64(t + i)))
        }
    }
    out := m.Forward(frames)
    for t := 0; t < len(out); t += 25 {
        fmt.Printf("frame %d: level %.3f, events %.3f\n", t, m.Slice(out[t], "level")[0], m.Slice(out[t], "events"))
    }
    fmt.Println(m.Parameters(), "weights")
}
```

---

## Invariants

- `Save` then `Load` gives a model of the same `Config` whose outputs are
  bit-identical to the saved one's (`TestWeightFileRoundTrips`).
- `Forward` is deterministic: one model and one sequence give bit-identical
  outputs on one platform and instruction set, on any number of cores.
- A model never changes after `Load` or `NewRandom`. `Forward` writes no model
  state and allocates its buffers per call, so any number of goroutines may run
  one model at once. A call spreads its frames over `GOMAXPROCS` goroutines,
  each output written by one of them in one order.
- `Forward` returns one vector per frame, the heads end to end as `Outputs`
  lists them, and nil for no frames. `Slice` returns a head's part of a vector,
  sharing its memory, and nil for a head the model has not.

## Gotchas

- The file decides the shape. A caller that needs particular inputs or heads
  checks `m.Config()` after `Load`.
- Every frame must hold exactly `Inputs` × `Channels` features. `Forward` does
  not check: a longer frame spills into the next one's place ([todo](doc/todo.md)).
- Outputs agree across platforms only to float32 rounding: the vector kernels
  (amd64 with AVX2 and FMA, arm64) sum in lanes, and arm64 and amd64 at
  `GOAMD64=v3` fuse the other multiplies and adds. amd64 without AVX2 at v1, 386
  and wasm run the plain loops and agree bit for bit.
- `Forward` takes every core `GOMAXPROCS` allows; callers running several at
  once share them.
- `Load` panics, rather than returning an error, on a header whose shape is too
  large to allocate ([todo](doc/todo.md)). Load files from trainers you trust.
- `Layers` is ignored when `Towers` are given, and `Heads` counts attention
  heads, not the output heads `Outputs` names. A cell net takes none of the
  transformer's sizes, and its `Cells.Heads` counts the output heads per cell.
- `Forward` allocates about 8 MB a call over 512 frames at width 256; a hot loop
  over short sequences pays for it each time. A cell net allocates its
  rearranged weights and a block's layers for each core, 5 MB a core at the
  rip's notes model's shape, and holds no whole-sequence layer.
