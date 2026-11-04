# Vendored ESP-NN

- Upstream: <https://github.com/espressif/esp-nn>
- Version: tag `v1.4.2`, commit `744d48deae6aa4a4d7b93b6a80f548d5298813a4`
- License: Apache-2.0 (`LICENSE`)

The files are copied unmodified: `include/`, `src/` and `LICENSE`. The CMake and Kconfig files,
tests, test application, tools and documentation are left out.

`build.rs` compiles:

- every `*_ansi.c` file except `src/logistic/esp_nn_logistic_ansi.c` (a float LUT built with
  libm's `expf`);
- on ESP32-S3, in addition, every `*_esp32s3.c`/`*_esp32s3.S` file, the generic optimizations
  (`*_opt.c`, some ESP32-S3 kernels fall back to them) and `src/common/esp_nn_multicore.c`,
  `src/convolution/esp_nn_conv_mt_split.c`, `src/basic_math/esp_nn_mul_mt.c` (the list of
  ESP-NN's `CMakeLists.txt` for this target).

The ESP32-P4 kernels (`*_riscv_pie.*`) are vendored but not built: they need Espressif's RISC-V
GCC (PIE extension).

To update: check out the new tag, copy the same directories, update this file, and compare the
structs and function signatures used by `src/lib.rs` and `src/dispatch.c` with the new headers.
Check that `src/common/common_functions.h` still requantizes like TFLite (double rounding) when
`CONFIG_NN_SKIP_NUDGE` is undefined, and that no new file needs ESP-IDF, the heap or libm.
