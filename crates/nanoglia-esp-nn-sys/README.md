# nanoglia-esp-nn-sys

Raw, `no_std` bindings to [ESP-NN](https://github.com/espressif/esp-nn) v1.4.2, Espressif's neural-network kernel library, for the [nanoglia](https://github.com/AlixANNERAUD/nanoglia) crates.

- ESP-NN is vendored in `esp-nn/` (see `esp-nn/VENDORED.md`) and compiled with the [`cc`](https://crates.io/crates/cc) crate, always optimized (`-O2`, or the `s`/`z`/`3` level of the profile), even in debug builds.
- The ANSI C reference kernels are built on every target, so the crate builds and its users can be tested anywhere (host, Arm, RISC-V, wasm32). On ESP32-S3 (`xtensa-esp32s3-*` targets) the build adds ESP-NN's optimized C and assembly kernels (PIE SIMD instructions); the `nanoglia_esp_nn_*` functions call the implementation the build selected, [`OPTIMIZED_ESP32S3`] tells which. The ESP32-P4 (RISC-V PIE) kernels need Espressif's RISC-V GCC and are not built yet.
- ESP-NN is compiled without `ESP_PLATFORM`: single core, no ESP-IDF, FreeRTOS or heap dependency. The `printf` of its error paths is compiled out and the float LUT logistic (libm) is left out: nothing links the C library except `memcpy`/`memset`, which `compiler_builtins` provides. `CONFIG_NN_SKIP_NUDGE` stays undefined, so the requantization is the double rounding of TensorFlow Lite.
- ESP-NN keeps the scratch buffers of its kernels in global variables: it is not reentrant. [`try_lock`] gives exclusive access to them without blocking; hold the [`ScratchLock`] from the `set_*_scratch_buf` call to the end of the kernel call.
- The bindings are written by hand (no bindgen at build time); a test checks the size, alignment and field offsets of every struct against the C compiler.

Cross-compiling needs a C compiler for the target and its C headers (`stdint.h`, `string.h`...), passed through the usual `CC_<target>`/`CFLAGS_<target>` variables. For ESP32-S3 that is Espressif's `xtensa-esp32s3-elf-gcc` (installed by [`espup`](https://github.com/esp-rs/espup)).

The build script exports the include directory as `DEP_ESP_NN_INCLUDE`.

## License

The Rust code of this crate and the vendored ESP-NN sources (`esp-nn/LICENSE`) are both Apache-2.0.
