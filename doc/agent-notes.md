# Agent Notes

What agents working on this repository learned that no rule in
[AGENTS.md](../AGENTS.md) and nothing in the [README](../README.md) or
[format.md](format.md) states: facts that cost a session an hour to rediscover,
and how to check them. Read it before starting. When a task teaches you one, add
it in the same PR, one bullet under its topic; delete a note that proves wrong or
that a document now states, and link the document instead.

## Toolchain

- The `gofmt` on a container's `PATH` can be older than the module's Go; run
  `"$(go env GOROOT)/bin/gofmt" -l`.
- `qemu-user-static` from apt runs a test binary built for another architecture:
  `GOARCH=arm64 go test -c`, then `qemu-aarch64-static ./transformer.test`.
- The compiler fuses `s += w * v` into one multiply-add on arm64 and on amd64 at
  `GOAMD64=v3`, so `Forward`'s bits differ there from amd64 v1's and 386's (which
  agree), by float32 rounding. Compare bits within one platform, and across
  platforms with a tolerance.
- Under `-cpu 1,4` the first row a process prints for a `b.Loop` benchmark ran
  at the list's last value, which the tests' loop over `-cpu` left set: run one
  `-cpu` value a process.

## Kernels

- `dotGeneric` and `axpyGeneric` sum in `Forward`'s old order, a dot from its
  bias, so 386, wasm and amd64 without AVX2 keep their bits. To check the plain
  path on an AVX2 machine, set `avx2FMA` false in a scratch edit and compare a
  saved file's outputs with main's.
- `TestKernelsMatchThePlainLoops` runs the NEON kernels under qemu (above); a
  kernel writing past its slice crashes that run in the runtime rather than
  failing the test. Speed needs a real arm64 device: a `GOOS=linux GOARCH=arm64
  CGO_ENABLED=0 go test -c` binary runs on Android from `/data/local/tmp` under
  `adb shell`. A phone heats up over a run; give each `-test.cpu` a fresh start.

- The GELU gives one value's bits on every kernel because no multiply is fused
  with its add: the plain `gelu` rounds each product through `float32(...)`,
  which the Go spec makes a barrier to fusion on arm64 and at `GOAMD64=v3`,
  and the kernels multiply and add apart. Its minima and maxima keep a NaN:
  x86's return their second source when either is one, which Go's assembler
  writes first.
- To show it after a change to `gelu`, a kernel or `geluK`, a scratch test
  runs all 2^32 float32 through `gelu` and `geluRow` in rows of 2^16 across the
  cores, compares their bits and sums value and bits into a checksum to carry
  between platforms: 33 s on amd64, 11 minutes for arm64 under qemu, one sum
  on amd64 with AVX2, at `GOAMD64=v3`, on 386 and on arm64. The same scan
  against `math.Erf` in float64 gives the worst error the README and
  [format.md](format.md) state.
- `geluK`'s polynomials are minimax fits (Lawson's reweighting, 2000 Chebyshev
  nodes) of (Φ(−u) − Φ(−s))/(s − u) over a piece ending at s, weighted by
  u·(s − u), what an error there leaves in the GELU; each coefficient is
  rounded to float32 from the highest down, the lower ones fitted again after
  each. Degree 10 either side of 2.5 is where float32's own rounding, 3.7e-8
  below zero, takes over from the fit's 2e-8; degree 9 reads 6e-8.

## The pass

- `Forward` and a `Stream` share each kind's arithmetic (`layer.project`,
  `attend`, `layer.feed`, `conv.apply`; the cell net's `Forward` is blocks of
  its stream's pass), so a change to one moves both, and
  `TestAStreamGivesForwardsOutputs` holds them equal. Its models' biases are
  set: `NewRandom`'s are zero, and a frame read past the sequence's end as a
  layer's output over zeros, where the layer's own zero belongs, shows only
  under a bias.
- To hold `Forward`'s bits through a refactor, a scratch module with `replace`
  to the checkout prints a hash of the outputs of several shapes and lengths
  (both kinds, convolutions and towers, lengths under and over a block); build
  it on main and on the branch, at `GOMAXPROCS` 1 and 4.
- A cell net's layer starts a block only as far before it as the layers after
  it read (`rows.reach`); starting each a block's whole reach before it cost
  `BenchmarkCellNet` 2 to 5% on one core. Chunks of 8 frames (`cellChunk`) ran
  slower than 16; 16, 32 and 64 ran alike.
- A planted fault shows a test can fail. Copy the file aside first: `git
  checkout` on it throws away the uncommitted work beside the fault.

## The float32 check

- `Forward` and a trainer's model agree to float32 rounding, and only a check
  shows it: redo it after any change to the forward pass or the file, and before
  trusting a file a changed trainer wrote. The reference trainer is rip-train,
  `tool/rip-train` in `github.com/lixenwraith/stapes`: `ripmodels.py` builds the
  same model in PyTorch (`Config`, `FrameModel`) and its `export` writes the
  file.
- A CPU PyTorch installs in a container in about three minutes (`pip install
  torch --index-url https://download.pytorch.org/whl/cpu`, then `numpy`).
- From the trainer's folder, in Python: seed torch, build a model, add
  `0.1 * torch.randn_like(p)` to every parameter so the norms' gains and shifts
  are not 1 and 0, run frames through it in `eval()` under `torch.no_grad()`,
  `export` it, and write the frames and the outputs as raw little-endian
  float32. The trainer's own frames are the realistic input
  (`ripmodels.features(train.MelFile(path)[:])` on a corpus loop); any frames
  exercise the path.
- In Go, a scratch module with `replace github.com/lixenwraith/transformer =>`
  the checkout: `Load` the file, `Forward` the frames, and take the largest
  difference from PyTorch's outputs.
- Check two shapes: the trainer's default and one with convolutions and towers
  (`convs=[(4, 3, 3, 1), (6, 1, 5, 2)], towers=[(2, 2), (3, 7)]` at width 32),
  which together cover every tensor and both namings. Rounding leaves the
  largest difference near a millionth of the largest output; one orders of
  magnitude past that is a mismatch.
- A change to the input checks both shapes again at `channels=2` too, on
  `ripmodels.features(frames, 2)`, the band frames' two planes in dB, over
  three seeds.

- The cell net's check is the rip trainer's `PitchNet` (`ripmodels.py` in
  stapes), whose `export` writes version 6: build it at the trainer's shape,
  perturb every parameter as above, run the output of `features(frames, 4)`,
  the four planes a notes model hears, through it, and dump `model.stack(x)`,
  the 32 channels of 271 Go reads, with the outputs. Its heads lie in the
  trainer's order, so compare them by name (`Slice`); the bits are the same
  on one to four cores.
  `TestCellNetMatchesItsDefinition` covers what PitchNet has none of yet: two
  `near` convolutions, a cell head of two channels. At version 6 the check runs
  PitchNet over the piano's 88 pitches too (270 bins, its regions over bins 21
  to 233) with a file's constant read back, at 213 bins with every bin pooled,
  and the frame model.

## Model

- `Conv` and `Tower` are in the format, and `Load` reads them, for a trainer to
  try: they are not dead code. Keep them, their loader and their round-trip
  test.
