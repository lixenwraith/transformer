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

## Model

- `Conv` and `Tower` are in no trained file yet, but trainers try them in their
  rounds: they are not dead code. Keep them, their loader and their round-trip
  test.
