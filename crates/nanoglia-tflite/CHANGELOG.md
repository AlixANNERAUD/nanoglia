# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-tflite-v0.1.3...nanoglia-tflite-v0.1.4) - 2026-11-14

### Added

- add float BATCH_MATMUL, GELU and L2_NORMALIZATION ([#57](https://github.com/AlixANNERAUD/nanoglia/pull/57))

## [0.1.3](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-tflite-v0.1.2...nanoglia-tflite-v0.1.3) - 2026-11-11

### Miscellaneous

- updated the following local packages: nanoglia-core

## [0.1.2](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-tflite-v0.1.1...nanoglia-tflite-v0.1.2) - 2026-11-10

### Added

- reductions, layout ops, PRELU, unary and binary math, comparisons, SELECT_V2 and CAST ([#35](https://github.com/AlixANNERAUD/nanoglia/pull/35))
- check whether a model runs, with every kernel and the memory needed ([#42](https://github.com/AlixANNERAUD/nanoglia/pull/42))

## [0.1.1](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-tflite-v0.1.0...nanoglia-tflite-v0.1.1) - 2026-11-04

### Added

- decode the options of ARG_MAX, ARG_MIN, RESIZE_* and TRANSPOSE_CONV

## [0.1.0](https://github.com/AlixANNERAUD/nanoglia/releases/tag/nanoglia-tflite-v0.1.0) - 2026-11-04

### Added

- decode the options of STRIDED_SLICE, GATHER, SPLIT and SPLIT_V

### Changed

- [**breaking**] rename the project to nanoglia and license it under Apache-2.0

### Documentation

- add badges and docs.rs settings
- rewrite the crate READMEs for crates.io and docs.rs
