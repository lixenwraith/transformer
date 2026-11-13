# The weight file

A weight file holds one model: its kind, its shape, the `Config`, what its
trainer states of it, and every weight in float32. `Save` writes it and `Load` reads it, and a trainer in any language can
write it from this document alone. Every number is little-endian, and nothing
is padded or aligned.

## Layout

```
magic      4 bytes     "VIFT" (56 49 46 54)
version    u32         6
kind       u32         0 a transformer, 1 a cell net
inputs     u32         features in a channel of a frame
channels   u32         1 or more, the planes of inputs a frame lays end to end
then a transformer's sizes, or a cell net's (below), then:
nheads     u32         1 to 64 output heads, then for each:
  namelen  u16           1 to 64
  name     namelen bytes, UTF-8 by convention
  size     u32           its scores per frame, 1 or more
nconsts    u32         0 to 64 constants (version 6 on; none before), then for each:
  namelen  u16           1 to 64
  name     namelen bytes
  value    f32           a number the trainer states of the model; no pass reads it
ntensors   u32         exactly the count the shape names (below), then for each:
  namelen  u16           1 to 64
  name     namelen bytes
  ndims    u8            its rank
  dims     u32 × ndims
  data     float32 × the product of dims: IEEE 754 binary32, row-major
checksum   u64         FNV-1a 64 of every byte before it
```

A transformer's sizes:

```
dim        u32         the width
heads      u32         attention heads; dim is a multiple of them
layers     u32         the one stack's blocks; read but unused when towers are given
ffn        u32         the feed-forward's width
window     u32         frames attended on either side
nconvs     u32         0 to 16, then for each convolution:
  out      u32           channels out
  time     u32           kernel frames, odd
  freq     u32           kernel features, odd
  pool     u32           features max-pooled into one, 1 for none
ntowers    u32         0 to 16, then for each tower:
  layers   u32           its blocks
  outputs  u32           how many output heads it computes, the next ones in order
```

A cell net's, `Config.Cells` in this order:

```
bins       u32         features of each channel the convolutions read, from its first
reduce     u32         channels of the 1×1 convolution
wide       u32 × 3     channels, frames, features of the wide convolution
cell       u32 × 3     channels, frames, features of the convolution to cells
stride     u32         bins from one cell to the next
depth      u32         convolutions over the cells in turn, 0 or more
near       u32 × 2     frames, features of each of them
first      u32         the first cell the cell heads read
read       u32         how many cells they read
heads      u32         the output heads, from the first, that are per cell
regions    u32         runs of bins each wide channel averages to
frame      u32 × 2     channels, frames of the frame path's first convolution
again      u32 × 2     frames, and their spacing, of its second
nscalars   u32         0 to 64, then for each:
  scalar   u32           an input of a frame, its channels end to end, the frame path reads
nlong      u32         0 to 16, then for each (version 5 on; none before):
  long     u32           the frames apart of a convolution over the cells after the near ones
pool       u32 × 2     the first bin the regions average and how many, 0 and 0 for
                       every bin (version 6 on; every bin before)
```

The header carries the `Config`: the sizes, the lists, its `Outputs` and its
`Constants`, which say what only the trainer knows of the model, such as what
a head's first score stands for.
Version 3 has no `kind` word and reads as a transformer, and version 2 no
`channels` word either, and reads as one channel.

## The tensors

A transformer's tensors come in this order, with these names and dims, and no
others:

```
convs.{i}.weight            out_i × in_i × time_i × freq_i    in_0 = channels, in_i = out_(i-1)
convs.{i}.bias              out_i
in.weight                   dim × features
in.bias                     dim
then each stack, P being "towers.{t}." for tower t, or "" for the one stack
of `layers` blocks when ntowers is 0:
  P layers.{j}.ln1.weight   dim
  P layers.{j}.ln1.bias     dim
  P layers.{j}.q.weight     dim × dim
  P layers.{j}.q.bias       dim
  P layers.{j}.k.weight     dim × dim
  P layers.{j}.k.bias       dim
  P layers.{j}.v.weight     dim × dim
  P layers.{j}.v.bias       dim
  P layers.{j}.o.weight     dim × dim
  P layers.{j}.o.bias       dim
  P layers.{j}.ln2.weight   dim
  P layers.{j}.ln2.bias     dim
  P layers.{j}.ff1.weight   ffn × dim
  P layers.{j}.ff1.bias     ffn
  P layers.{j}.ff2.weight   dim × ffn
  P layers.{j}.ff2.bias     dim
  P ln.weight               dim
  P ln.bias                 dim
  P out.weight              size × dim                        size: the stack's output heads summed
  P out.bias                size
```

`i` counts convolutions, `t` towers and `j` a stack's blocks, each from 0.
`features` is n × c, n starting at `inputs` and c at `channels`, each
convolution in turn dividing n by its pool, rounding down, and making c its
`out`. A file holds 2·nconvs + 2 tensors before the stacks and 16·L + 4 for a
stack of L blocks.

A weight of out × in holds row o's in values from o·in on, which is PyTorch's
`nn.Linear.weight` as it stands. `convs.{i}.weight` is `nn.Conv2d.weight` over a
frame image of channels × frames × features, and a norm's `weight` and `bias`
are `nn.LayerNorm`'s gain and shift.

A cell net's come in this order, a convolution's weight being `nn.Conv2d`'s
or `nn.Conv1d`'s as it stands:

```
reduce.weight      reduce × channels × 1 × 1
reduce.bias        reduce
wide.weight        wide × reduce × wide frames × wide features
wide.bias          wide
cells.weight       cell × wide × cell frames × cell features
cells.bias         cell
near.{d}.weight    cell × cell × near frames × near features      d from 0 to depth − 1
near.{d}.bias      cell
long.{l}.weight    cell × cell × 3 × near features               l from 0 to nlong − 1
long.{l}.bias      cell
cellout.weight     out × cell × 1 × 1                            out: the cell heads' sizes over read, summed
cellout.bias       out
frame1.weight      frame × (wide·regions + nscalars) × frame frames
frame1.bias        frame
frame2.weight      frame × frame × again frames
frame2.bias        frame
frameout.weight    size × frame                                  size: the frame heads summed
frameout.bias      size
```

## What the weights compute

The `Config` fixes the computation, and a trainer's model computes this for its
weights to mean the same here. A transformer:

1. A sequence is T frames of `channels` × `inputs` features, channel c's from
   c·inputs on. With convolutions the frames are an image of `channels`
   channels, frames by features; without, a frame goes to the input layer as it
   comes. Convolution i writes, for
   each channel o, frame t and feature k, its bias plus the sum over channels c
   in and the kernel of weight[o][c][dt][df] × x[c][t + dt − time/2][k + df −
   freq/2], the halves rounded down and zero past the image's edges
   (PyTorch's `padding=(time//2, freq//2)`), then the GELU, then the max of each run of `pool` features, a
   remainder past the last whole run dropped. Frames are never pooled. After the
   last convolution a frame is its channels laid end to end, channel c's n
   features from c·n on.
2. The input layer maps each frame to `dim`.
3. Each stack starts from the input layer's output h and runs its blocks, each
   pre-norm: h += o(attend(ln1(h))), then h += ff2(gelu(ff1(ln2(h)))). The stack
   ends with out(ln(h)).
4. `attend` splits q(x), k(x) and v(x) into `heads` runs of d = dim/heads
   features, attention head a taking the a-th run. Frame i weighs each frame j
   within `window` of it, on both sides, by the softmax over j of
   q_i·k_j / √d − s_a·|i − j|, with slope s_a = 2^(−8(a+1)/heads) for a from 0,
   and sums their values; the heads' results lie back in their runs. Frames
   past the window weigh nothing, and no position embedding is added.
5. A norm takes each frame over its `dim` features to (x − mean) /
   √(variance + 1e−5) × weight + bias, the variance the biased one. The GELU is
   the exact one, 0.5·x·(1 + erf(x/√2)).
6. A frame's output is its stacks' outputs end to end in order, so its scores
   lie as `Outputs` lists the heads, tower t computing the next `outputs` of
   them. The scores are raw: no activation follows the output layer.

A cell net computes, every convolution over the frames as an image with the
kernel centred on each frame, zero past the sequence's ends:

1. Each frame's first `bins` features of each channel go through the 1×1
   `reduce` and the GELU, then `wide`, centred on each bin with zeros past the
   bins, and the GELU.
2. `cells` makes cell j from the bins `cell features` wide centred on bin
   stride·j, those before bin 0 zero, for each j whose last bin is a bin:
   (bins + ⌊cell features/2⌋ − cell features) / stride + 1 cells, rounded down,
   then the GELU. Each `near` in turn runs over the cells, zero past them, and
   adds its input back before its GELU, and each `long` after them the same way
   over three frames `long` apart, those past the sequence zero. The 1×1 `cellout` makes the cell heads'
   channels at cells `first` to `first + read`: a cell head of size k·read takes
   the next k channels, each channel's `read` cells after the one before.
3. The frame path averages each `wide` channel over `regions` runs of the n
   bins `pool` names from its first, f, or of every bin: run g from bin
   f + ⌊g·n/regions⌋ to before f + ⌈(g+1)·n/regions⌉ (PyTorch's
   `adaptive_avg_pool2d` over those bins), channel by channel, then the
   scalars as they are.
   `frame1` over those, the GELU; `frame2` over frames `again spacing` apart,
   adding back its input before the GELU; then `frameout`, the frame heads end
   to end.
4. A frame's output is the cell heads, each its cells in order, then the frame
   heads, as `Outputs` lists them.

`Forward` keeps float32 throughout but for a norm's mean and variance and the
softmax's exponential, which it takes in float64 and rounds. Its GELU is
max(x, 0) − a·(Φ(−a) − Φ(−6)) over a = |x| up to 6, the difference two
polynomials meeting at 2.5: within 4.2e-8 of the exact one below zero and
2.6e-7 above, where a float32 itself lies up to 2.4e-7 from it, and the same
bits on every platform. It agrees with a PyTorch model in float32 to rounding
([agent notes](agent-notes.md)).

## What Load refuses

`Load` returns an error for a file unless:

- it is at least 8 bytes, and its last 8 are the FNV-1a 64 (offset basis
  0xcbf29ce484222325, prime 0x100000001b3) of every byte before them;
- the magic is `VIFT`, the version 2 to 6 and the kind 0 or 1;
- it has at most 16 convolutions, at most 16 towers, 1 to 64 output heads and
  at most 64 constants;
- each convolution has `out` and `pool` of 1 or more and odd `time` and `freq`;
- each tower computes 1 or more heads and, with towers, theirs sum to `nheads`;
- a cell net has every count 1 or more but `depth`, odd kernels, at most
  `inputs` bins and a cell of them, a frame head besides its cell heads, `read`
  cells from `first` within the cells, each cell head a multiple of `read`,
  the regions' bins within the bins, and each scalar within a frame;
- `inputs`, `channels`, `features`, `dim`, `heads` and `ffn` are 1 or more, and
  `dim` a multiple of `heads`;
- the output heads' names are 1 to 64 bytes and distinct, and each size is 1 or
  more; the constants' names are 1 to 64 bytes and distinct;
- the tensors are the list above in count, names, ranks and dims, each of at
  most 2^28 floats;
- no byte lies between the last tensor and the checksum.

`Load` reads the whole file before checking it. Each u32 becomes a Go `int`, so
on a 32-bit platform a size of 2^31 or more reads negative and is refused. A
header naming a shape too large to allocate panics today instead of returning
an error ([todo](todo.md)).

## Versions

The version is the layout's: a change to these bytes bumps it, in
`fileVersion` and here in one commit, and `Load` reads the versions it knows.
`Save` writes 6, which added the constants and the bins a cell net's regions
average; `Load` reads 5, which added a cell net's `long`, 4, which added
`kind`, 3, which added `channels`, and 2, since trained files carry them. Every file trained so far carries the magic `VIFT`, so
a new magic is a versioned change of its own ([todo](todo.md)).
