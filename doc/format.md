# The weight file

A weight file holds one model: its shape, the `Config`, and every weight in
float32. `Save` writes it and `Load` reads it, and a trainer in any language can
write it from this document alone. Every number is little-endian, and nothing
is padded or aligned.

## Layout

```
magic      4 bytes     "VIFT" (56 49 46 54)
version    u32         2
inputs     u32         features in a frame
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
nheads     u32         1 to 64 output heads, then for each:
  namelen  u16           1 to 64
  name     namelen bytes, UTF-8 by convention
  size     u32           its scores per frame, 1 or more
ntensors   u32         exactly the count the shape names (below), then for each:
  namelen  u16           1 to 64
  name     namelen bytes
  ndims    u8            its rank
  dims     u32 × ndims
  data     float32 × the product of dims: IEEE 754 binary32, row-major
checksum   u64         FNV-1a 64 of every byte before it
```

The header carries the `Config`: `inputs` to `window` are its sizes, and the
three lists its `Convs`, `Towers` and `Outputs`.

## The tensors

The tensors come in this order, with these names and dims, and no others:

```
convs.{i}.weight            out_i × in_i × time_i × freq_i    in_0 = 1, in_i = out_(i-1)
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
`features` is `inputs` without convolutions; with them it is n × out of the last,
n starting at `inputs` and divided by each convolution's pool in turn, rounding
down. A file holds 2·nconvs + 2 tensors before the stacks and 16·L + 4 for a
stack of L blocks.

A weight of out × in holds row o's in values from o·in on, which is PyTorch's
`nn.Linear.weight` as it stands. `convs.{i}.weight` is `nn.Conv2d.weight` over a
frame image of channels × frames × features, and a norm's `weight` and `bias`
are `nn.LayerNorm`'s gain and shift.

## What the weights compute

The `Config` fixes the computation, and a trainer's model computes this for its
weights to mean the same here:

1. A sequence is T frames of `inputs` features. With convolutions the frames
   are an image of one channel, frames by features. Convolution i writes, for
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

`Forward` keeps float32 throughout but for a norm's mean and variance, the GELU
and the softmax's exponential, which it takes in float64 and rounds; it agrees
with a PyTorch model in float32 to rounding ([agent notes](agent-notes.md)).

## What Load refuses

`Load` returns an error for a file unless:

- it is at least 8 bytes, and its last 8 are the FNV-1a 64 (offset basis
  0xcbf29ce484222325, prime 0x100000001b3) of every byte before them;
- the magic is `VIFT` and the version 2;
- it has at most 16 convolutions, at most 16 towers and 1 to 64 output heads;
- each convolution has `out` and `pool` of 1 or more and odd `time` and `freq`;
- each tower computes 1 or more heads and, with towers, theirs sum to `nheads`;
- `inputs`, `features`, `dim`, `heads` and `ffn` are 1 or more, and `dim` a
  multiple of `heads`;
- the output heads' names are 1 to 64 bytes and distinct, and each size is 1 or
  more;
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
Version 2 is the only one today. Every file trained so far carries the magic
`VIFT`, so a new magic is a versioned change of its own ([todo](todo.md)).
