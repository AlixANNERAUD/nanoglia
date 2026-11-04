# nanoglia-esp-nn-sys

Raw, `no_std` bindings to [ESP-NN](https://github.com/espressif/esp-nn) v1.4.2, Espressif's neural-network kernel library, for the [nanoglia](https://github.com/AlixANNERAUD/nanoglia) crates.

- ESP-NN is compiled with the [`cc`](https://crates.io/crates/cc) crate, always optimized (`-O2`, or the `s`/`z`/`3` level of the profile), even in debug builds.
- The ANSI C reference kernels are built on every target, so the crate builds and its users can be tested anywhere (host, Arm, RISC-V, wasm32). On ESP32-S3 (`xtensa-esp32s3-*` targets) the build adds ESP-NN's optimized C and assembly kernels (PIE SIMD instructions); the `nanoglia_esp_nn_*` functions call the implementation the build selected, [`OPTIMIZED_ESP32S3`] tells which. The ESP32-P4 (RISC-V PIE) kernels need Espressif's RISC-V GCC and are not built yet.
- ESP-NN is compiled without `ESP_PLATFORM`: single core, no ESP-IDF, FreeRTOS or heap dependency. The `printf` of its error paths is compiled out and the float LUT logistic (libm) is left out: nothing links the C library except `memcpy`/`memset`, which `compiler_builtins` provides. `CONFIG_NN_SKIP_NUDGE` stays undefined, so the requantization is the double rounding of TensorFlow Lite.
- ESP-NN keeps the scratch buffers of its kernels in global variables: it is not reentrant. [`try_lock`] gives exclusive access to them without blocking; hold the [`ScratchLock`] from the `set_*_scratch_buf` call to the end of the kernel call.
- The bindings are written by hand (no bindgen at build time); a test checks the size, alignment and field offsets of every struct against the C compiler.

Cross-compiling needs a C compiler for the target and its C headers (`stdint.h`, `string.h`...), passed through the usual `CC_<target>`/`CFLAGS_<target>` variables. For ESP32-S3 that is Espressif's `xtensa-esp32s3-elf-gcc` (installed by [`espup`](https://github.com/esp-rs/espup)).

The build script exports the include directory as `DEP_ESP_NN_INCLUDE`.

## ESP-NN sources

In the repository, `esp-nn/` is a git submodule of <https://github.com/espressif/esp-nn>, pinned at tag `v1.4.2` (commit `744d48deae6aa4a4d7b93b6a80f548d5298813a4`). Run `git submodule update --init --recursive` after cloning; without it the build script stops and says so. Do not edit its files.

`build.rs` reads only `esp-nn/include/` and `esp-nn/src/`, and compiles:

- every `*_ansi.c` file except `src/logistic/esp_nn_logistic_ansi.c` (a float LUT built with libm's `expf`);
- on ESP32-S3, in addition, every `*_esp32s3.c`/`*_esp32s3.S` file, the generic optimizations (`*_opt.c`, some ESP32-S3 kernels fall back to them) and `src/common/esp_nn_multicore.c`, `src/convolution/esp_nn_conv_mt_split.c`, `src/basic_math/esp_nn_mul_mt.c` (the list of ESP-NN's `CMakeLists.txt` for this target).

The ESP32-P4 kernels (`*_riscv_pie.*`) are not built: they need Espressif's RISC-V GCC (PIE extension). The tests, test application and tools of the repository are never built. The published package carries only the headers, the files above and `esp-nn/LICENSE` (the `include` list of `Cargo.toml`, kept in sync with `build.rs`).

To move to a new ESP-NN release:

1. Check out the new tag in `esp-nn/` (`git -C crates/nanoglia-esp-nn-sys/esp-nn checkout <tag>`) and stage the submodule.
2. Compare the list of files compiled before and after (`ar t` on `libesp_nn.a` in the build-script output directory, for the host and for `xtensa-esp32s3-none-elf`) and with ESP-NN's `CMakeLists.txt`; update `build.rs` and the `include` list of `Cargo.toml` together. No new file may need ESP-IDF, the heap or libm (`mise run no-malloc`, `mise run cross-esp32s3`).
3. Compare the structs and function signatures used by `src/lib.rs` and `src/dispatch.c` with the new headers, and check that `src/common/common_functions.h` still requantizes like TFLite (double rounding) when `CONFIG_NN_SKIP_NUDGE` is undefined.
4. Run the golden suite with the ESP-NN backend (`mise run check` runs `cargo test -p nanoglia --features esp-nn --test golden`) and `mise run package`, then update the version in this file, the crate description and `AGENTS.md`.

## License

The Rust code of this crate and the ESP-NN sources (`esp-nn/LICENSE`) are both Apache-2.0.
