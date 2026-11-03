# AGENTS.md

Guidance for coding agents working on this repository. `CLAUDE.md` and `GEMINI.md` are
symlinks to this file: edit `AGENTS.md` only.

## What this repository is

nanoglia runs TensorFlow Lite (`.tflite`) models **loaded at runtime** on microcontrollers,
through one `#![no_std]` API modelled on wasi-nn, with the best kernels of each chip. A Cargo
workspace:

| Crate | Role |
| --- | --- |
| `crates/nanoglia` | The library: `Graph` (load, select kernels, plan memory), `ExecutionContext` (arena, `compute`), `Registry`. |
| `crates/nanoglia-core` | Shared types (`TensorDesc`, `OpDesc`, `Error`/`ErrorCode`/`Detail`), the `Backend`/`Kernel` interface, TFLite-exact quantization math (`quant`), `Arena`, `Block`. |
| `crates/nanoglia-tflite` | Zero-copy, allocation-free, panic-free `.tflite` verifier and reader (own flatbuffer code, `#![forbid(unsafe_code)]`). |
| `crates/nanoglia-cmsis-nn-sys` | Vendored CMSIS-NN v8.0.0 integer kernels (`cmsis-nn/`, do not edit; see `VENDORED.md`), built with `cc`, hand-written FFI. |
| `crates/nanoglia-cmsis-nn` | CMSIS-NN backend: int8/int16 conv, depthwise, FC, pooling, softmax, add, mul, pad, min/max. |
| `crates/nanoglia-kernels` | Pure-Rust backend: what CMSIS-NN lacks (reshape, quantize, mean, LUT activations, broadcasting binary ops, int16 softmax/FC...) and float32. |

Other directories: `fuzz/` (cargo-fuzz target `parse`, own workspace), `examples/nano33ble/`
(firmware for the Arduino Nano 33 BLE, own workspace), `tools/oracle/` (uv project generating
`testdata/` with LiteRT's reference kernels), `tools/check-no-malloc.sh`, `testdata/` (golden
cases, see `testdata/README.md`).

## Invariants

- `#![no_std]` everywhere; no HAL, RTOS or SDK dependency. The only C code is CMSIS-NN (and
  ESP-NN later), pure C compiled with `cc`.
- **Every byte of dynamic memory comes from the allocator the caller passes**
  (`allocator_api2::alloc::Allocator`: `Global`, a custom one, or `Arena`). A graph makes one
  allocation, an execution context one, `compute` none. No C `malloc` (`mise run no-malloc`).
  `crates/nanoglia/tests/memory.rs` checks this; keep it passing.
- Model bytes are borrowed and never copied; constant buffers must be aligned for their type.
- Models are untrusted input: no panic on any model (`unwrap`, indexing, overflow); return
  `Error`. Fuzz with `mise run fuzz`.
- Kernels are **bit-exact with TFLite**: port the reference / TFLite Micro code (prepare and
  eval) faithfully, including whether scales are multiplied in `f32` or `f64`. Requantization
  uses double rounding by default (TFLite Micro, CMSIS-NN); the `single-rounding` feature
  switches every crate to `TFLITE_SINGLE_ROUNDING`.
- A backend's `select` returns a kernel only if it fully supports the operator; `prepare`
  computes everything from the model, `eval` never reads the model and never allocates.

## Testing

With [mise](https://mise.jdx.dev): `mise install`, then `mise run check` (fmt, clippy, tests,
Arm cross builds) before pushing. Cargo goes through mbx (mr-boxington, a shared build cache);
do not run several builds in the same target directory at once.

- `crates/nanoglia/tests/golden.rs` runs every case of `testdata/` end to end. `PENDING` lists
  the cases expected to fail, with the reason; a listed case that passes fails the test.
- `mise run oracle` regenerates `testdata/` (deterministic; LiteRT `BUILTIN_REF`). LiteRT's
  int8 and 16x8 FULLY_CONNECTED requantize in floating point, unlike TFLite Micro (issue #1).
- `mise run test-wasip1` runs the whole suite compiled to wasm32-wasip1 under wasmtime;
  `cross-arm`, `cross-riscv32`, `cross-wasm32` build the libraries without a global allocator.
- Host tests run CMSIS-NN's portable C code only; the Arm DSP/Helium paths have not run on
  hardware yet (issue #2). Say what you actually checked.

## CI and releases

`.github/workflows/ci.yml` mirrors the mise tasks (fmt, clippy, tests, cross builds, wasm,
no-malloc, firmware, cargo-deny, MSRV, docs, fuzz). `coverage.yml` uploads `mise run coverage`
to Codecov. `release.yml` runs release-plz (`release-plz.toml`): a release PR with versions and
changelogs from the commit messages, then publication. `pr-title.yml` checks PR titles.

## Conventions

- Rust 2024, MSRV 1.89, `cargo fmt`; English code and documentation; document public items.
- Commit messages and PR titles follow Conventional Commits (`feat:`, `fix:`, `perf:`,
  `refactor:`, `docs:`, `test:`, `build:`, `ci:`, `chore:`), with a body explaining the why.
- No AI attribution: no `Co-Authored-By` trailers for assistants, no "generated with" lines.
- Design notes and plans stay out of the repository.
