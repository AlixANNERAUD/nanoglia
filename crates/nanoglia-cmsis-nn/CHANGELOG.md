# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-cmsis-nn-v0.1.1...nanoglia-cmsis-nn-v0.1.2) - 2026-11-07

### Added

- int16 DEPTHWISE_CONV_2D, pooling, LOGISTIC and TANH ([#33](https://github.com/AlixANNERAUD/nanoglia/pull/33))

## [0.1.1](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-cmsis-nn-v0.1.0...nanoglia-cmsis-nn-v0.1.1) - 2026-11-04

### Added

- run int8 TRANSPOSE_CONV on CMSIS-NN

### Documentation

- document ARG_MAX, ARG_MIN, RESIZE_* and TRANSPOSE_CONV

## [0.1.0](https://github.com/AlixANNERAUD/nanoglia/releases/tag/nanoglia-cmsis-nn-v0.1.0) - 2026-11-04

### Added

- *(cmsis-nn)* expose the int8 kernels
- [**breaking**] make kernels traits borrowed from their backend

### Changed

- [**breaking**] rename the project to nanoglia and license it under Apache-2.0

### Documentation

- add badges and docs.rs settings
- *(cmsis-nn)* correct the supported-operator table and two comments
- rewrite the crate READMEs for crates.io and docs.rs
