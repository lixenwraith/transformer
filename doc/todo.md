# TODO

Where the module may grow, and the seam each change would use. Each item states
what remains and the smallest thing that would close it. Delete an item when it
lands; this file is not a changelog. Source files do not carry TODO comments.

Priorities: P1 is wanted next, P2 is convenient follow-up, and P3 is an idea.

A change to the file's bytes takes the next version: `fileVersion` in
`weights.go`, the layout and the tensor list in [format.md](format.md), and a
`Load` that still reads versions 2 and 3, which trained files carry. A field a
`Config` gains reads zero for today's behaviour, so every `Config` literal and
every file keeps its meaning.

## Hardening

### Bound the shape before Load allocates it

- Priority: P2
- Affected files: `weights.go` (`Load`), `config.go`, `transformer.go`
  (`newModel`, `Forward`)

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

`BenchmarkForward` is the before: a width-256 model of eight heads, four layers,
a feed-forward of 1024 and a window of 64, over 512 frames of 230 inputs into
318 outputs, 8.1 MB a call. On a 4-vCPU Intel Xeon container (2.1 GHz, AVX2),
medians of six:

```
                                  one core              four cores
plain loops, one goroutine        1.31 s    390 frames/s   1.31 s    390 frames/s
AVX2 kernels, frames over cores   0.258 s  1981 frames/s   0.078 s  6600 frames/s
```

A frame is 256 samples at 22050 Hz to the rip, 86 a second of audio: 4.5 times
real time before, 23 and 77 after. `GOEXPERIMENT=simd`'s package would make
every importer set the experiment, so the kernels are assembly.

### Run a batch of sequences in one call

- Priority: P3
- Affected files: `transformer.go` (`Forward`, `linear.apply`)

A `Model` holds no scratch and `Forward` allocates its buffers each call, so
goroutines over sequences work today, each reading every weight on its own. A
batch call over several sequences reads each weight row once for all of them:
`linear.apply`'s loop over frames becomes one over the batch's frames, and
`attend` keeps each sequence to itself. A workspace kept across calls would cut
the allocations. Neither touches the `Config` or the file. Within one sequence
the same holds: `linear.apply` streams every weight row once a frame, and a
kernel over four frames a row would read it once for them.

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
