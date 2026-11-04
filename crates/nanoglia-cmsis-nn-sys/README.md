# nanoglia-cmsis-nn-sys

Raw, `no_std` bindings to [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) v8.0.0, the neural-network kernel library of Arm, for the [nanoglia](https://github.com/AlixANNERAUD/nanoglia) crates.

- The integer kernels of CMSIS-NN (int8, int16 and int4) are compiled with the [`cc`](https://crates.io/crates/cc) crate for any target, always optimized (`-O2`, or the `s`/`z`/`3` level of the profile), even in debug builds.
- On Arm Cortex-M the compiler flags expose the DSP extension (Armv7E-M: Cortex-M4/M7; Armv8-M Mainline: Cortex-M33) and Helium (Armv8.1-M with the `mve` target feature), so CMSIS-NN uses its optimized code paths. Elsewhere (host, RISC-V, Xtensa...) it compiles its portable C code.
- The float kernels are left out: they would pull in libm. No function calls `malloc` or any other part of the C library except `memcpy`/`memset`/`memmove`, which `compiler_builtins` provides.
- The bindings are written by hand (no bindgen at build time); a test checks the size, alignment and field offsets of every struct against the C compiler.

Cross-compiling needs a C compiler for the target and its C headers (`stdint.h`, `string.h`...), for example `arm-none-eabi-gcc` with newlib's headers, passed through the usual `CC_<target>`/`CFLAGS_<target>` variables; nothing links the C library.

The build script exports the include directory as `DEP_CMSIS_NN_INCLUDE` to crates that compile C code against CMSIS-NN.

## CMSIS-NN sources

In the repository, `cmsis-nn/` is a git submodule of <https://github.com/ARM-software/CMSIS-NN>, pinned at tag `v8.0.0` (commit `13c97dbb6f781d4aab38ed34e6e441f42b79aff4`). Run `git submodule update --init --recursive` after cloning; without it the build script stops and says so. Do not edit its files.

`build.rs` compiles every `.c` file of `cmsis-nn/Source/` except the float ones (`*_f16.c`, `*_f32.c`, `*_flt.c`, among them `arm_softmax_f32.c`, `arm_softmax_f16.c` and `arm_convolve_f16.c`), with `ARM_NN_ENABLE_F32` and `ARM_NN_ENABLE_F16` set to `0`, so the integer headers never include the float ones. `Source/Bindings/` (C++/pybind11 buffer-size bindings for Python) is never built. The published package carries only `cmsis-nn/Include/`, those integer sources and `cmsis-nn/LICENSE` (the `include` list of `Cargo.toml`).

To move to a new CMSIS-NN release:

1. Check out the new tag in `cmsis-nn/` (`git -C crates/nanoglia-cmsis-nn-sys/cmsis-nn checkout <tag>`) and stage the submodule.
2. Compare the list of files compiled before and after (`ar t` on `libcmsis_nn.a` in the build-script output directory, for the host and an Arm target such as `thumbv7em-none-eabihf`): a new float file must stay out (update `is_integer_source` in `build.rs` and the `include` list of `Cargo.toml` together), a new integer file must be needed and free of libm and `malloc` (`mise run no-malloc`).
3. Compare the structs and function signatures used by `src/lib.rs` with the new headers (the layout test checks the structs).
4. Run the golden suite (`mise run check`) and `mise run package`, then update the version in this file, the crate description and `AGENTS.md`.

## License

The Rust code of this crate and the CMSIS-NN sources (`cmsis-nn/LICENSE`) are both Apache-2.0.
