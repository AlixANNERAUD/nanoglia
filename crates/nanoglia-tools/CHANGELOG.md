# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-tools-v0.2.0...nanoglia-tools-v0.2.1) - 2026-11-14

### Added

- add float BATCH_MATMUL, GELU and L2_NORMALIZATION ([#57](https://github.com/AlixANNERAUD/nanoglia/pull/57))

## [0.2.0](https://github.com/AlixANNERAUD/nanoglia/compare/nanoglia-tools-v0.1.0...nanoglia-tools-v0.2.0) - 2026-11-11

### Added

- [**breaking**] add one Cargo feature per operator, to link only the kernels a firmware needs ([#48](https://github.com/AlixANNERAUD/nanoglia/pull/48))

### Documentation

- depend on the published crates ([#50](https://github.com/AlixANNERAUD/nanoglia/pull/50))

## [0.1.0](https://github.com/AlixANNERAUD/nanoglia/releases/tag/nanoglia-tools-v0.1.0) - 2026-11-10

### Added

- check whether a model runs, with every kernel and the memory needed ([#42](https://github.com/AlixANNERAUD/nanoglia/pull/42))
