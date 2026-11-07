# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-cmsis-nn-sys-v0.1.0...nanoglia-cmsis-nn-sys-v0.1.1) - 2026-11-04

### Added

- run int8 TRANSPOSE_CONV on CMSIS-NN

## [0.1.0](https://github.com/AlixANNERAUD/nanoglia/releases/tag/nanoglia-cmsis-nn-sys-v0.1.0) - 2026-11-04

### Changed

- [**breaking**] rename the project to nanoglia and license it under Apache-2.0

### Documentation

- add badges and docs.rs settings

### Fixed

- *(cmsis-nn-sys)* use DSP on Armv8-M only when the target enables it

### Miscellaneous

- *(cmsis-nn-sys)* track CMSIS-NN as a git submodule

### Performance

- vectorize the C kernels with SIMD128 whenever Rust does
