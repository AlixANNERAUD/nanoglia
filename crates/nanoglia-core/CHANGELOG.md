# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-core-v0.1.1...nanoglia-core-v0.1.2) - 2026-11-07

### Added

- reductions, layout ops, PRELU, unary and binary math, comparisons, SELECT_V2 and CAST ([#35](https://github.com/AlixANNERAUD/nanoglia/pull/35))

## [0.1.1](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-core-v0.1.0...nanoglia-core-v0.1.1) - 2026-11-04

### Added

- decode the options of ARG_MAX, ARG_MIN, RESIZE_* and TRANSPOSE_CONV

## [0.1.0](https://github.com/AlixANNERAUD/nanoglia/releases/tag/nanoglia-core-v0.1.0) - 2026-11-04

### Added

- [**breaking**] make kernels traits borrowed from their backend
- implement standard traits on tensors, graphs and plain values
- [**breaking**] run inferences step by step, asynchronously, or until a deadline
- decode the options of STRIDED_SLICE, GATHER, SPLIT and SPLIT_V

### Changed

- [**breaking**] rename the project to nanoglia and license it under Apache-2.0

### Documentation

- add badges and docs.rs settings
- rewrite the crate READMEs for crates.io and docs.rs
- document stateful backends, observers and the new graph sizes

### Miscellaneous

- add the CI, coverage and PR title workflows
