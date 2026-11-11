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
  in `github.com/lixenwraith/stapes` (until it moves, `tool/rip-train` in
  `github.com/lixenwraith/vif`): `ripmodel.py` builds the same model in PyTorch
  (`Config`, `RipModel`) and its `export` writes the file.
- A CPU PyTorch installs in a container in about three minutes (`pip install
  torch --index-url https://download.pytorch.org/whl/cpu`, then `numpy`).
- From the trainer's folder, in Python: seed torch, build a model, add
  `0.1 * torch.randn_like(p)` to every parameter so the norms' gains and shifts
  are not 1 and 0, run frames through it in `eval()` under `torch.no_grad()`,
  `export` it, and write the frames and the outputs as raw little-endian
  float32. The trainer's own frames are the realistic input
  (`ripmodel.features(train.MelFile(path)[:])` on a corpus loop); any frames
  exercise the path.
- In Go, a scratch module with `replace github.com/lixenwraith/transformer =>`
  the checkout: `Load` the file, `Forward` the frames, and take the largest
  difference from PyTorch's outputs.
- Check two shapes: the trainer's default and one with convolutions and towers
  (`convs=[(4, 3, 3, 1), (6, 1, 5, 2)], towers=[(2, 2), (3, 7)]` at width 32),
  which together cover every tensor and both namings. On 862 frames of a corpus
  loop the largest differences were 8.6e-6 at width 256 with four layers, on
  outputs up to 6.4, and 1.8e-6 with the convolutions and towers, on outputs up
  to 2.7. A difference orders of magnitude past those is a mismatch, not
  rounding.
- A change to the input checks both shapes again at `channels=2` too, on
  `ripmodel.features(mel, 2)`. On 862 frames of a piano recording the largest
  differences were 9.2e-6 at the default shape and 4.2e-6 to 5.4e-6 with the
  convolutions and towers over three seeds, as one channel read on those frames
  (3.1e-6 to 4.7e-6).

- The cell net's check is the rip trainer's `PitchNet` (`ripmodel.py` in
  stapes), whose `export` writes version 4: build it at the trainer's shape,
  perturb every parameter as above, run the output of `features(mel, 2)`
  through it, and dump `model.stack(x)`, the 16 channels of 214 Go reads, with
  the outputs. Its heads lie in the trainer's order, so compare them by name
  (`Slice`). On 600 and 2000 frames over three seeds the largest differences
  were 2.4e-6 to 5.1e-6 on outputs up to 4.3 to 11.1, and the bits the same on
  one to four cores. `TestCellNetMatchesItsDefinition` covers what PitchNet
  has none of yet: two `near` convolutions, a cell head of two channels.

## Model

- `Conv` and `Tower` are in no trained file yet, but trainers try them in their
  rounds: they are not dead code. Keep them, their loader and their round-trip
  test.
