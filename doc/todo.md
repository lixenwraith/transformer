# TODO

Where the module may grow, and the seam each change would use. Each item states
what remains and the smallest thing that would close it. Delete an item when it
lands; this file is not a changelog. Source files do not carry TODO comments.

Priorities: P1 is wanted next, P2 is convenient follow-up, and P3 is an idea.

A change to the file's bytes takes the next version: `fileVersion` in
`weights.go`, the layout and the tensor list in [format.md](format.md), and a
`Load` that still reads versions 2 to 4, which trained files carry. A field a
`Config` gains reads zero for today's behaviour, so every `Config` literal and
every file keeps its meaning.

## Hardening

### Bound the shape before Load allocates it

- Priority: P2
- Affected files: `weights.go` (`Load`), `config.go`, `transformer.go`
  (`newModel`, `Forward`), `cells.go`

`Load` hands the header's `Config` to `newModel`, which allocates every tensor
before the tensors' own checks run: a 59-byte file naming a width of 2^24 panics
in `make` ("len out of range"), and one of 2^20 asks for 4 TiB. The fix sums the
floats the shape needs from the `Config`, in overflow-checked products, and
refuses past a cap, such as `maxTensor`'s 2^28 floats over the whole model,
before `newModel` runs. `Forward` copies frames without checking their length: one
longer than `Inputs` × `Channels` spills into the next frame's place, which a shorter next
frame does not wholly overwrite. It should refuse both.

## Position biases

### Offer other position biases than ALiBi

- Priority: P3
- Affected files: `config.go` (`Config`), `transformer.go` (`newModel`,
  `attend`), `weights.go`, [format.md](format.md)

A `Position` field on `Config`, its zero ALiBi, names the bias; the header writes
it after the towers. Rotary embeddings need no tensor, and rotate q and k in
`attend`; a learned relative bias adds one tensor a block, `layers.{j}.pos` of
heads × (2·window + 1), after `o` in the list; a one-sided window, for a
sequence read as it arrives, is a kind of the same field. The seam is
`Model.slopes`, built in `newModel`, and the `slopes[h]·d` term `attend`
subtracts.

## Speed

The benchmarks run the rip's shapes. A frame is 256 samples at 22050 Hz to the
rip, 86 a second of audio.

- `BenchmarkForward`: a width-256 model of eight heads, four layers, a
  feed-forward of 1024 and a window of 64, over 512 frames of 230 inputs into
  318 outputs, 8.2 MB a call.
- `BenchmarkCellNet`: the rip's notes model, 16 channels of 214 inputs, 71
  cells and 318 outputs, over 2584 frames, 30 s; `near` without long
  convolutions, 169,862 weights, and `long` with `Long` 3, 9 and 27, 197,606.
- `BenchmarkStream`: their frames through a stream, 43 a write as the rip's
  half second, the cell net's with `Long`.

Frames a second on a 4-vCPU Xeon with AVX2, medians of five, one `-cpu` a
process, with the GELU a `math.Erf` a value and with it over a row in the
kernels:

```
                                    math.Erf a value     a row in the kernels
                                    1 core    4 cores    1 core    4 cores
Forward, transformer                  2900       8910      3880      12490
Forward, cell net near                1950       7280      2810       9800
Forward, cell net long                1230       4530      1910       7050
stream, transformer at its reach      2730       5960      3780       7760
stream, transformer waiting 46        2980       6570      4420       8420
stream, cell net long                 1450       1460      2300       2230
```

A row's GELU takes 1 ns a value in the AVX2 kernel and 15 through the plain
function, where `math.Erf` took 23. A caller that ran `Forward` over a window
around each write's frames, as the rip's ear did over 150 for 43, ran each
frame three and a half times. The cell net's `Forward` runs blocks of the
stream's pass, and allocates 6.5 MB a call on one core where it took 27.5,
13.4 MB on four where it took 97.

The plain loops ran `BenchmarkForward` a fifth as fast as the kernels, both
with `math.Erf`: 390 frames a second against 1977 on an earlier Xeon, 421
against 2071 on one core of a Pixel 8 Pro (Tensor G3), whose nine cores ran
1376. A 30 s rip with 12a on the Pixel, 2584 frames, took 2.56 s on one core
and 1.07 s on nine. The NEON GELU has run only under qemu: a phone's figures
with it are unmeasured. `GOEXPERIMENT=simd`'s package would make every
importer set the experiment, so the kernels are assembly.

### Bring a short stream nearer Forward

- Priority: P2
- Affected files: `stream.go` (`transformerNet.pass`)

A transformer's stream under its reach shares its look-ahead evenly over a
stack's blocks. On random weights, whose attention is near flat
(`BenchmarkForward`'s model, three seeds, 1024 frames, outputs of rms 1),
its outputs lie from `Forward`'s by, rms and worst:

```
waiting 46 frames    15%   1.2 to 1.5
waiting 86           11%   0.9 to 1.0
waiting 128           8%   0.6 to 0.7
waiting 192         4.5%   0.4
`Forward` over 64 before and 46 after each 43, the rip's ear before streams
                      3%   0.2 to 0.3
```

Other shares of 46 did no better: all to the first block 26%, all to the last
32%, 20, 13, 8 and 5 from the first 15.1%. What is left: the measure on trained
weights, which the rip takes on its recordings; a pass that runs the frames
within the look-ahead twice, every block attending as far as the frames in and
again once the next write brings more, about the window's outputs at twice a
frame's cost; and a trainer that masks each block's attention as the short
stream does, so the weights expect it.

### Spread a stream's write over the cores

- Priority: P3
- Affected files: `cells.go` (`cellPass`), `stream.go`, `transformer.go`
  (`across`)

A cell net's stream runs on its caller's goroutine, 2300 frames a second on
any number of cores, and a transformer's hands each of four cores eleven
frames of a 43-frame write, 7760 frames a second against `Forward`'s 12490
over 512 (above). A cell net's convolution could run its cells across the
cores; where the transformer's write loses its third is unmeasured.

### Make the cell net's pass cheaper

- Priority: P2
- Affected files: `cells.go` (`conv.run`), `kernel_*`

On one core the profile gives `dot` 79%, a row's GELU 4% and the loops and
copies the rest. `dot` reads two floats a multiply, so a kernel of four weight
rows against one or two patches, which shares its loads, would near double it:
`conv.run` dots every output channel against one patch.

### Make a phone's cores beat one at 512 frames

- Priority: P3
- Affected files: `transformer.go` (`across`, `linear.apply`), `kernel_arm64.s`

On the Pixel the nine cores trail one over `BenchmarkForward`'s 512 frames, and
four only draw level, though nine win over a clip's 2584. Unmeasured: the
A510s' share of a layer's blocks, the shared caches the cores' weight reads
contend for, and the clock a phone holds with every core busy. A kernel keeping
a weight row in registers across a block's frames cuts the reads again.

### Run a batch of sequences in one call

- Priority: P3
- Affected files: `transformer.go` (`Forward`, `linear.apply`)

A `Model` holds no scratch and `Forward` allocates its buffers each call, so
goroutines over sequences work today, each reading every weight on its own. A
batch call over several sequences reads each weight row once for all of them:
`linear.apply`'s loop over frames becomes one over the batch's frames, and
`attend` keeps each sequence to itself. A workspace kept across calls would cut
the allocations. Neither touches the `Config` or the file.

## Weights

### Store weights in reduced precision

- Priority: P3
- Affected files: `weights.go` (`tensor`, `Load`, `Save`), `transformer.go`
  (`linear`), [format.md](format.md)

float16 or bfloat16 weights halve the file and the memory a frame reads, int8
with a scale per row quarters them. The file's next version writes a type byte
after each tensor's dims, float32 the one every tensor has today, and an int8
weight's scales as its own tensor, `<name>.scale` of out floats, after it in the
list. Norms and biases stay float32. `linear` gains the narrower weights and an
`apply` over them; the float32 check gains a tolerance per type.

### Write the format from Python without a trainer

- Priority: P3
- Affected files: a new `python/` folder, `transformer_test.go`

A trainer copies its exporter today. A numpy-only module that writes a file from
a `Config` and a dictionary of named arrays, in the order of the tensor list
([format.md](format.md), `tensors` in `weights.go`), lets any trainer write it,
and a small file it wrote, kept in `testdata/` with its outputs, is a test Go
runs without Python.

### Give the file a magic of its own

- Priority: P3
- Affected files: `weights.go`, [format.md](format.md)

The magic `VIFT` is in every file trained so far. A new magic starts a version,
and `Load` keeps reading `VIFT` with the versions before it.
