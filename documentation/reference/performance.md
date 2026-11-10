# Performance

Inference times of the four MLPerf Tiny int8 models, measured by `mise
run bench` (host) and `mise run bench-wasm` (WebAssembly under
wasmtime), and a comparison with the other Rust inference engines
(`mise run bench-compare`, [below](#compared-with-other-rust-engines)):
outputs, host time, and size and RAM of a keyword-spotting firmware.

> These are **host and wasmtime numbers, not microcontroller numbers.**
> On the host and in WebAssembly, CMSIS-NN runs its portable C code; its
> Arm DSP and Helium paths have not been measured on hardware yet
> (issue #2). Use them to compare builds and targets on the same machine,
> not to predict the time on a chip.

## How it is measured

`crates/nanoglia/examples/bench.rs` loads each model from a 16-byte
aligned copy with the default features, fills its input with
pseudo-random bytes, then repeats `set_input` and `compute` for 2
seconds (at least 5 runs) and prints the median time of `compute`.
`NANOGLIA_BENCH_MS` changes the time budget per model.

`mise run bench-wasm` builds the same program for `wasm32-wasip1` twice,
in two target directories: scalar, and with SIMD128 for both the Rust
code (`RUSTFLAGS='-C target-feature=+simd128'`) and the CMSIS-NN C code
(the build scripts add `-msimd128` to the C code when the `simd128` target feature is on). It runs both under wasmtime.

## Machine

| | |
| --- | --- |
| Host | Linux 6.12 aarch64 virtual machine, 8 vCPUs, on an Apple silicon laptop |
| Rust | 1.99.1, release profile |
| C compiler (wasm32) | clang of wasi-sdk-33 |
| WebAssembly runtime | wasmtime 50.0.0 |
| Date | 2026-11-04 |
| Commit | `26f1b25` (runs 2 and 3); `de9118d` for run 1, whose inference code is the same |

## Results

Median time of one inference, from the third run; the range shows the
lowest and highest median of the runs on this machine. The virtual
machine shares its CPU with the laptop, so the spread is wide.

| Model | Operators | Host | Host, range | wasm32 | wasm32, range | wasm32 + SIMD128 | wasm32 + SIMD128, range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kws_ref_model` | 13 | 269.5 µs | 260–270 µs | 1059.8 µs | 1060–1384 µs | 508.7 µs | 509–537 µs |
| `vww_96_int8` | 31 | 1431.3 µs | 938–1431 µs | 3327.8 µs | 3244–4366 µs | 1602.7 µs | 1603–1612 µs |
| `pretrainedResnet_quant` | 16 | 946.1 µs | 859–1033 µs | 5126.5 µs | 5127–6127 µs | 1569.5 µs | 1541–1570 µs |
| `ad01_int8` | 10 | 19.4 µs | 19–21 µs | 90.1 µs | 90–99 µs | 35.4 µs | 32–35 µs |

Host and scalar wasm32 ranges cover three runs; the SIMD128 ranges cover
runs 1 and 3 (run 2 was disturbed by other work on the machine and is
left out).

Under wasmtime, SIMD128 makes these models 2 to 3.3 times faster than
the scalar build. See [Run in the browser](../how-to-guides/run-in-the-browser.md)
to enable it in your build.

## Run it yourself

```sh
mise run bench
mise run bench-wasm
NANOGLIA_BENCH_MS=5000 mise run bench
```

When you publish numbers, record the machine, the toolchain, the date
and the commit, as above.

## Compared with other Rust engines

`mise run bench-compare` runs the same models, with the same inputs, on
every Rust engine that can load a TensorFlow Lite model, and builds a
keyword-spotting firmware with each one. Its code is in
`benchmarks/compare/`, a Cargo workspace of its own with its own
`Cargo.lock` (the versions below), and a uv project (`tools/`) for the
driver and the emulator. `ROUNDS` and `BENCH_MS` change the number of
runs per engine and the time budget per model.

> The host times are **host numbers, not microcontroller numbers**, and
> the firmware run in an emulator, which gives no time at all. No engine
> has been timed on a chip here (issue #2).

### Engines

| Engine | Version | How the model gets in | Kernels | Built as |
| --- | --- | --- | --- | --- |
| nanoglia | commit `c94f1eb` | Loaded at run time | CMSIS-NN (portable C on the host, DSP on the firmware), Rust | Host: default features. Firmware: int8 only (`cmsis-nn`, `kernels`, as `examples/nano33ble`), and default features |
| [tract](https://crates.io/crates/tract-tflite) | `tract-tflite` 0.23.9 | Loaded at run time (`model_for_path`, `into_optimized`, `into_runnable`) | tract-linalg | Host only: needs `std` |
| [MicroFlow](https://crates.io/crates/microflow) | `microflow` 0.1.3, `microflow-macros` 0.1.1 | Compiled into Rust code by its `model` macro | Rust (nalgebra) | Host and firmware |
| [embedded-nn](https://github.com/leftger/embedded-nn) | 0.2.3, repository commit `69ea418` | Imported and compiled by its `embedded_nn_model` macro | Rust | Host and firmware. Its TFLite importer and macro are not published on crates.io: the runtime and the macro come from the same Git commit |
| [ember-rs](https://crates.io/crates/ember-infer-core) | `ember-infer-*` 0.1.2 | Compiled into Rust code by its `model` macro | Rust reference backend (`ember-infer-ref`) | Host and firmware. The reference backend needs `alloc`: the firmware has a 256-byte heap (`embedded-alloc` 0.7.0) |

[tfmicro](https://crates.io/crates/tfmicro) 0.1.0 (bindings to a 2020
TensorFlow Lite Micro) is left out: it does not build with this
toolchain. Its build script does not compile against `fs_extra` 1.2 or
later (missing field `content_only`; pinning `fs_extra` 1.1.0 fixes
that), then bindgen 0.53.3 fails with libclang 22 on the bundled
`flatbuffers.h` ("overload resolution selected deleted operator '='"),
and with `BINDGEN_EXTRA_CLANG_ARGS=-fdelayed-template-parsing` the
generated bindings lack `TfLiteStatus`, `FrontendState` and others.
Going further means patching the crate or installing an old libclang.

No engine was patched. The only addition one needed is the heap of
ember-rs's firmware.

### Method

- **Inputs and expected outputs.** The three golden samples of each
  model (`testdata/mlperf-tiny/golden/<model>/sample_*`), whose outputs
  come from LiteRT's reference kernels. Every engine reads the int8
  input and writes the int8 output; MicroFlow returns dequantized `f32`
  scores, quantized back with the output scale and zero point (exact,
  since its dequantization is `scale * (q - zero_point)`).
- **Correctness first.** Each host benchmark runs every sample once and
  reports the largest absolute difference from the expected output in
  int8 units, how many output elements differ, and on how many samples
  the top-1 class matches. On `ad01_int8`, the expected output itself
  is not TensorFlow Lite Micro's: LiteRT's reference FULLY_CONNECTED
  requantizes in floating point, TensorFlow Lite Micro and CMSIS-NN in
  fixed point (issue #1, the 3 pending golden cases of nanoglia), so an
  engine that follows TensorFlow Lite Micro differs there by up to 2.
- **Host time.** One binary per engine (`benchmarks/compare/engines/`),
  all built with the same release profile (`opt-level = 3`,
  `lto = "fat"`, `codegen-units = 1`), all running on a thread with a
  256 MiB stack (MicroFlow and ember-rs keep their activations on the
  stack). The timed call is the whole inference, from the `&[i8]` input
  to the `&mut [i8]` output, input and output copies included (for
  tract, building its input tensor). After a warm-up (at least 10 runs
  and 200 ms), the call is repeated for 2 seconds, cycling through the
  three samples, and the median is kept. Each engine runs 3 times, in a
  different order each round; the tables give the median of the three
  medians and their range.
- **Firmware.** One crate per engine (`benchmarks/compare/firmware/`)
  for `thumbv7em-none-eabihf` with `-C target-cpu=cortex-m4` (the
  nRF52840 of the Arduino Nano 33 BLE), all with the same profile
  (`opt-level = "s"`, `lto = "fat"`, `codegen-units = 1`,
  `panic = "abort"`), cortex-m-rt 0.7.7 and panic-halt. Each runs the
  keyword-spotting model once on a static input (golden sample 0),
  writes the scores to a static and stops; nothing is printed. A
  firmware without any engine gives the size of that common part.
  Sizes come from `arm-none-eabi-size`: `text` is code and read-only
  data, the model included (`kws_ref_model.tflite` is 53936 bytes;
  nanoglia embeds that file, the ahead-of-time engines their own
  encoding of the weights).
- **RAM.** Static RAM (`data` + `bss`) plus the peak stack. nanoglia's
  graph and context live in a static `Arena` of 23872 bytes, the
  `Arena::used()` the firmware records on this target, 96 bytes more
  than on 32-bit WebAssembly, where CMSIS-NN runs its portable C code
  (its DSP kernels may ask for other scratch sizes, see
  [memory](memory.md#mlperf-tiny-footprints));
  `MemoryRequirements::arena_bytes(1)` is 23887 bytes, 15 more for a
  buffer of unknown alignment. embedded-nn's static arena is the
  16492 bytes its planner computes (`ARENA_SIZE`).
- **Peak stack, and the firmware's output.** `tools/emulate.py` runs each
  firmware once under Unicorn 2.1.4 emulating a Cortex-M4 with the
  nRF52840's memory map (1 MiB of flash, 256 KiB of RAM). It fills the
  RAM with a pattern before the reset handler; after the run, the peak
  stack is the distance from the top of RAM to the lowest word that no
  longer holds it (stack painting). It is exact for this run (this
  input, this code path) but leaves out interrupt handlers, and does not
  count stack reserved and never written. The largest single frames of
  the disassembly agree: 80896 + 200 bytes in MicroFlow's `main`, 79872
  in ember-rs's, 4160 in nanoglia. The emulator also reads the scores
  the firmware computed and compares them with the golden output of
  sample 0. It gives no timing.

### Machine and versions

| | |
| --- | --- |
| Host | Linux 6.12 aarch64 virtual machine, 8 vCPUs, on an Apple silicon laptop |
| Rust | 1.99.1, profiles above |
| C compiler (firmware) | arm-none-eabi-gcc 14.2.1 (Debian `gcc-arm-none-eabi` 14.2.rel1-1) |
| Emulator | Unicorn 2.1.4 |
| Date | 2026-11-04 |
| Commit | `c94f1eb` |
| Crates | `tract-tflite` and `tract-core` 0.23.9, `microflow` 0.1.3, `microflow-macros` 0.1.1, `nalgebra` 0.32.6, `embedded-nn` and `embedded-nn-macros` 0.2.3 (Git `69ea4187a298431fcaa313c3ab2a2ec1c473c4c7`), `ember-infer-core`, `-macros` and `-ref` 0.1.2, `embedded-alloc` 0.7.0, `cortex-m-rt` 0.7.7 |

The whole run was also made at commit `13dff8d`, which differs only in
the text of one skip reason: every correctness and firmware number was
the same, and every host time within 3%.

### Correctness on the golden samples

Largest absolute difference from the expected output, in int8 units,
then (output elements that differ / all output elements of the three
samples, samples whose top-1 class matches; not given for the
autoencoder).

| Engine | `kws_ref_model` | `vww_96_int8` | `pretrainedResnet_quant` | `ad01_int8` |
| --- | ---: | ---: | ---: | ---: |
| nanoglia | **0** (0/36, 3/3) | **0** (0/6, 3/3) | **0** (0/30, 3/3) | 2 (601/1920) |
| tract | 129 (36/36, 0/3) | 156 (6/6, 3/3) | does not load: "No super type" on ADD | 170 (1920/1920) |
| MicroFlow | 6 (2/36, 3/3) | 5 (6/6, 3/3) | does not build: "unsupported operator: ADD" | **0** (0/1920) |
| embedded-nn | 204 (36/36, 3/3) | 26 (6/6, 3/3) | 202 (29/30, 3/3) | 2 (601/1920) |
| ember-rs | 6 (2/36, 3/3) | **0** (0/6, 3/3) | 25 (5/30, 3/3) | **0** (0/1920) |

- nanoglia gives the reference output byte for byte on the three
  classifiers. On the autoencoder it differs by up to 2 on 601 values,
  the TensorFlow Lite Micro requantization explained above; embedded-nn
  differs on as many values, by as much.
- MicroFlow and ember-rs compute parts of these models in floating point
  (MicroFlow's requantization, ember-rs's softmax and ADD): differences
  of 5 to 25, the top-1 class unchanged on these samples. Both match
  LiteRT exactly on the autoencoder.
- **embedded-nn's outputs differ** on the three classifiers: its code
  generator emits SOFTMAX with fixed parameters (multiplier 1073741824,
  shift 20, `diff_min` -256, `crates/embedded-nn-codegen/src/emit_rust.rs`
  at `69ea418`) instead of deriving them from the input scale, so the
  scores are far too flat (sample 0 of keyword spotting: -74 for the
  top class where the reference gives 126). The top class still
  matched on these samples.
- **tract's outputs differ** on all three models it loads (another
  class on every keyword-spotting sample, every value different on the
  autoencoder); the quantization it reports for the keyword-spotting output
  (zero point 14, scale 0.1447) is not the model's (-128, 1/256). The cause
  is not established: it may lie in how our harness drives tract rather than
  in tract itself. `BENCH_DEBUG=1` prints the outputs, and tract's input and
  output facts; corrections are welcome in an issue.

### Host time per inference

Median of the three run medians, and the range of the run medians.
Times of an engine whose outputs differ from the reference are given
with the largest difference (Δ); read the section above first.

| Engine | `kws_ref_model` | `vww_96_int8` | `pretrainedResnet_quant` | `ad01_int8` |
| --- | ---: | ---: | ---: | ---: |
| nanoglia | **249.2 µs** (249–250) | **869.9 µs** (869–870) | **779.9 µs** (778–781) | **18.2 µs** (18–18), Δ2 |
| tract | 1391.5 µs (1389–1392), outputs differ | 4367.1 µs (4367–4380), outputs differ | — | 40.2 µs (40–40), outputs differ |
| MicroFlow | 2355.0 µs (2352–2361), Δ6 | 7057.4 µs (7054–7058), Δ5 | — | 357.1 µs (357–357) |
| embedded-nn | 1962.2 µs (1960–1965), outputs differ | 3953.1 µs (3952–3955), outputs differ | 8223.1 µs (8212–8231), outputs differ | 69.3 µs (69–70), Δ2 |
| ember-rs | 4748.5 µs (4744–4752), Δ6 | 14473.8 µs (14459–14494) | 16021.6 µs (15993–16095), Δ25 | 320.5 µs (320–321) |

On this host, nanoglia is the fastest on all four models: 9.5 and 19
times faster than MicroFlow and ember-rs on keyword spotting, 4.5 to
20.5 times faster than every other engine on the two image models, 2.2
times faster than tract on the autoencoder. The host runs CMSIS-NN's
portable C code; on a Cortex-M the gap depends on the chip and has not
been measured.

### Keyword-spotting firmware

`thumbv7em-none-eabihf`, sizes in bytes. RAM total = static RAM + peak
stack. Output: largest difference from the golden output of sample 0,
computed by the firmware under the emulator.

| Firmware | text | data | bss | Static RAM | Peak stack | RAM total | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No engine | 1724 | 0 | 16 | 16 | 40 | 56 | — |
| nanoglia, int8 only | 164644 | 0 | 23892 | 23892 | 5136 | 29028 | **0** |
| nanoglia, default features | 191096 | 0 | 23892 | 23892 | 5136 | 29028 | 0 |
| MicroFlow | 45296 | 0 | 16 | 16 | 90676 | 90692 | 0 |
| embedded-nn | **36532** | 0 | 16512 | 16512 | 960 | **17472** | 200 (differs) |
| ember-rs | 42592 | 0 | 304 | 304 | 80752 | 81056 | 0 |

- **Flash: nanoglia is 3.6 to 4.5 times larger.** Without the 53936
  bytes of the model and the 1724 bytes common to every firmware, the
  int8-only nanoglia is about 109 KB of code: the model verifier, the
  memory planner, and every int8 kernel of CMSIS-NN and of the Rust
  backend, since the model is only known at run time. The
  ahead-of-time engines compile only the operators of this model, with
  its weights as plain arrays. The float32 and int16 kernels add 26452
  bytes. A firmware that holds several models, or receives them later,
  pays for the engine once. This firmware is built with every operator
  (`all-ops`); a firmware that knows the operators of its models can
  link only their kernels ([Memory](memory.md#code-size)).
- **RAM: nanoglia needs 28.3 KiB, known before running**: the arena
  (23.3 KiB, from `MemoryRequirements`) and 5 KiB of stack. MicroFlow
  and ember-rs keep every activation in the stack frame of the
  generated code: 88.6 and 79.2 KiB, known only by measuring.
  embedded-nn needs the least, 17.1 KiB (its 16.1 KiB arena holds the
  activations only, where nanoglia's also holds the graph and its
  prepared parameters), but its scores differ from the reference.
- Under the emulator, the nanoglia firmware, with CMSIS-NN's DSP
  kernels, gives the reference output of sample 0 byte for byte. This
  is an instruction-set emulator, not the chip: issue #2 stays open.

### Not measured

- **Time on a microcontroller**, for every engine: no hardware was used,
  and the emulator does not model time.
- **WebAssembly**: the comparison has not been built for `wasm32-wasip1`.
- **tract firmware**: tract needs `std`. **tfmicro**: does not build
  (above).
- **The other three models on the firmware**: only keyword spotting
  was built for the Cortex-M4.
