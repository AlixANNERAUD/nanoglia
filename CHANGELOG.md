# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0](https://github.com/AlixANNERAUD/nanoglia/compare/v0.1.1...v0.2.0) - 2026-11-10

### Added

- int16 DEPTHWISE_CONV_2D, pooling, LOGISTIC and TANH ([#33](https://github.com/AlixANNERAUD/nanoglia/pull/33))
- reductions, layout ops, PRELU, unary and binary math, comparisons, SELECT_V2 and CAST ([#35](https://github.com/AlixANNERAUD/nanoglia/pull/35))
- [**breaking**] support owned models and graph-owning execution contexts ([#37](https://github.com/AlixANNERAUD/nanoglia/pull/37))
- check whether a model runs, with every kernel and the memory needed ([#42](https://github.com/AlixANNERAUD/nanoglia/pull/42))

### Fixed

- *(cmsis-nn)* bound the 16x8 CONV_2D and FULLY_CONNECTED accumulators ([#39](https://github.com/AlixANNERAUD/nanoglia/pull/39))

### Miscellaneous

- run the golden cases under emulation on Cortex-M4F, M33, M0+ and riscv32imc ([#41](https://github.com/AlixANNERAUD/nanoglia/pull/41))

## [0.1.1](https://github.com/AlixANNERAUD/nanoglia/compare/v0.1.0...v0.1.1) - 2026-11-04

### Added

- add RESIZE_NEAREST_NEIGHBOR and RESIZE_BILINEAR to the Rust backend
- add TRANSPOSE_CONV to the Rust backend
- run int8 TRANSPOSE_CONV on CMSIS-NN
- decode the options of ARG_MAX, ARG_MIN, RESIZE_* and TRANSPOSE_CONV
- add ARG_MAX and ARG_MIN to the Rust backend

### Documentation

- document ARG_MAX, ARG_MIN, RESIZE_* and TRANSPOSE_CONV

## [0.1.0](https://github.com/AlixANNERAUD/nanoglia/releases/tag/v0.1.0) - 2026-11-04

### Added

- compute the memory a model needs without loading it
- add the esp-nn feature
- [**breaking**] make kernels traits borrowed from their backend
- observe each operator with ExecutionContext::compute_with
- implement standard traits on tensors, graphs and plain values
- [**breaking**] run inferences step by step, asynchronously, or until a deadline

### Changed

- [**breaking**] rename the project to nanoglia and license it under Apache-2.0

### Documentation

- add badges and docs.rs settings
- add runnable examples to the main API
- rewrite the crate READMEs for crates.io and docs.rs
- document stateful backends, observers and the new graph sizes
- explain how to run long inferences without blocking

### Fixed

- check constant alignment when computing memory requirements
- saturate MemoryRequirements::arena_bytes instead of overflowing

### Miscellaneous

- export a PyTorch model with litert-torch and run it with nanoglia

### Performance

- add an MLPerf Tiny benchmark for the host and wasm32
