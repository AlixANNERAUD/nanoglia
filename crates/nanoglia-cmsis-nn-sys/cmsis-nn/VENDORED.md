# Vendored CMSIS-NN

- Upstream: <https://github.com/ARM-software/CMSIS-NN>
- Version: tag `v8.0.0`, commit `13c97dbb6f781d4aab38ed34e6e441f42b79aff4`
- License: Apache-2.0 (`LICENSE`)

The files are copied unmodified. Only the integer subset is kept:

- `Include/`: `arm_nnfunctions.h`, `arm_nnsupportfunctions.h`, `arm_nn_math_types.h`, `arm_nn_tables.h`, `arm_nn_types.h`, `Internal/arm_nn_compiler.h`, `Internal/arm_nn_config.h`.
- `Source/`: every `.c` file except the float ones.

Removed:

- `Source/Bindings/` (C++/pybind11 buffer-size bindings for Python);
- every float file: `*_f16.c`, `*_f32.c`, `*_flt.c` (among them `arm_softmax_f32.c`, `arm_softmax_f16.c` and `arm_convolve_f16.c`), `Source/NNSupportFunctions/arm_get_buffer_size_common.h` (only used by them), the `*_flt.h` headers and the float-only `Include/Internal/` headers. The float API is disabled by default upstream (`ARM_NN_ENABLE_F32`/`ARM_NN_ENABLE_F16` are `0`), so the integer headers never include them;
- the CMake files, documentation, examples, tests, Python tooling and packaging files.

To update: check out the new tag, copy the same subset, update this file, and compare the structs and function signatures used by `src/lib.rs` with the new headers.
