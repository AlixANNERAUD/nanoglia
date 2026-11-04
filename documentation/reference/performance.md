# Performance

Inference times of the four MLPerf Tiny int8 models, measured by `mise
run bench` (host) and `mise run bench-wasm` (WebAssembly under
wasmtime).

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
(`CFLAGS_wasm32_wasip1=-msimd128`). It runs both under wasmtime.

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
