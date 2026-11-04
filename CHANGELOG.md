# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
