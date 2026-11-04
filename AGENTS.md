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
| `crates/nanoglia-cmsis-nn-sys` | CMSIS-NN v8.0.0 as a git submodule (`cmsis-nn/`, do not edit; pin and bump procedure in the crate's README): integer kernels only, built with `cc`, hand-written FFI. |
| `crates/nanoglia-cmsis-nn` | CMSIS-NN backend: int8/int16 conv, depthwise, FC, pooling, softmax, add, mul, pad, min/max. |
| `crates/nanoglia-esp-nn-sys` | Vendored ESP-NN v1.4.2 (`esp-nn/`, do not edit; see `VENDORED.md`): ANSI C kernels everywhere, ESP32-S3 optimized ones on `xtensa-esp32s3`; hand-written FFI and the ESP-NN scratch try-lock. |
| `crates/nanoglia-esp-nn` | ESP-NN backend (facade feature `esp-nn`, tried first): int8 conv, depthwise, FC, pooling, softmax, add, mul; runs the CMSIS-NN kernel when another context holds the ESP-NN lock. |
| `crates/nanoglia-kernels` | Pure-Rust backend: what CMSIS-NN lacks (reshape, quantize, mean, LUT activations, broadcasting binary ops, int16 softmax/FC...) and float32. |

Other directories: `fuzz/` (cargo-fuzz target `parse`, own workspace), `examples/nano33ble/`
(firmware for the Arduino Nano 33 BLE, own workspace), `tools/oracle/` (uv project generating
`testdata/` with LiteRT's reference kernels), `tools/check-no-malloc.sh`, `testdata/` (golden
cases, see `testdata/README.md`).

## Invariants

- `#![no_std]` everywhere; no HAL, RTOS or SDK dependency. The only C code is CMSIS-NN and
  ESP-NN (built without `ESP_PLATFORM`), pure C/assembly compiled with `cc`.
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
- Host tests run CMSIS-NN's portable C code and ESP-NN's ANSI C code only; the Arm DSP/Helium
  paths and ESP-NN's ESP32-S3 kernels have not run on hardware yet (issue #2). `mise run
  cross-esp32s3` builds them (toolchain: `mise run install-esp`). Say what you actually checked.

## CI and releases

`.github/workflows/ci.yml` mirrors the mise tasks (fmt, clippy, tests, cross builds, wasm,
no-malloc, firmware, cargo-deny, MSRV, docs, fuzz). `coverage.yml` uploads `mise run coverage`
to Codecov. `release.yml` runs release-plz (`release-plz.toml`): a release PR with versions and
changelogs from the commit messages, then publication. `pr-title.yml` checks PR titles.

## Documentation

User documentation lives in `documentation/` and follows [Diátaxis](https://diataxis.fr);
`documentation/README.md` is the index. `tutorials/` are lessons followed from start to finish,
`how-to-guides/` reach one goal each, `explanation/` says why, `reference/` lists every item and
number.

- Full words in file and directory names (`how-to-guides`, `development-tasks`), relative links
  between pages. Add new pages to the index and to the Documentation table of the root
  `README.md`.
- Check every API name, signature, feature, size and default against the source, and every
  number against a command you ran (say which, with the machine, date and commit). Quote time
  and memory figures only from `reference/performance.md` and `reference/memory.md`, with a link.
- Describe users generically ("your application", "an embedded OS"). Do not present host or
  wasmtime timings as microcontroller numbers, and keep the hardware-validation status (issue #2)
  visible where it matters.
- When a change lands, update: `reference/supported-operators.md` (a `select`/`plan` condition,
  a kernel or an operator), `reference/features.md` (Cargo features), `reference/errors.md` (an
  `Error`, `ErrorCode` or `Detail` returned by a public function), `reference/memory.md` (the
  allocation contract or the block layout; re-measure the MLPerf Tiny footprints with
  `cargo test -p nanoglia --test memory -- --nocapture`), `reference/performance.md` (after
  `mise run bench` / `bench-wasm` on a performance change), `reference/development-tasks.md`
  (`mise.toml` or `.github/workflows/`), and the tutorial or guides whose code or output changes
  (run their code again).

## Conventions

- Rust 2024, MSRV 1.89, `cargo fmt`; English code and documentation; document public items.
- Commit messages and PR titles follow Conventional Commits (`feat:`, `fix:`, `perf:`,
  `refactor:`, `docs:`, `test:`, `build:`, `ci:`, `chore:`), with a body explaining the why.
- No AI attribution: no `Co-Authored-By` trailers for assistants, no "generated with" lines.
- Design notes and plans stay out of the repository.
- A bug or limitation found in a dependency (CMSIS-NN, ESP-NN, LiteRT, ...) gets a GitHub issue
  with the `upstream` label: upstream version and link to the exact lines, the problem, the
  impact on nanoglia and the workaround (with its file), and the steps to report or drop it.
