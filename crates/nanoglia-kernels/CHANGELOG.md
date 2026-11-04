# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-kernels-v0.1.0...nanoglia-kernels-v0.1.1) - 2026-11-04

### Added

- add ARG_MAX and ARG_MIN to the Rust backend
- add RESIZE_NEAREST_NEIGHBOR and RESIZE_BILINEAR to the Rust backend
- add TRANSPOSE_CONV to the Rust backend

### Documentation

- document ARG_MAX, ARG_MIN, RESIZE_* and TRANSPOSE_CONV

## [0.1.0](https://github.com/AlixANNERAUD/nanoglia/releases/tag/nanoglia-kernels-v0.1.0) - 2026-11-04

### Added

- [**breaking**] make kernels traits borrowed from their backend
- add TRANSPOSE, SLICE and STRIDED_SLICE to the Rust backend
- add GATHER to the Rust backend
- add SPLIT and SPLIT_V to the Rust backend

### Changed

- [**breaking**] rename the project to nanoglia and license it under Apache-2.0

### Documentation

- add badges and docs.rs settings
- rewrite the crate READMEs for crates.io and docs.rs
- document TRANSPOSE, SLICE, STRIDED_SLICE, GATHER, SPLIT and SPLIT_V

### Fixed

- *(kernels)* never panic on int32 overflow when requantizing
- *(kernels)* require TFLite's output zero point for quantized LOGISTIC

### Miscellaneous

- build for browsers and run the tests under WASI
