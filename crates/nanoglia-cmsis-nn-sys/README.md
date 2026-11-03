# nanoglia-cmsis-nn-sys

Raw, `no_std` bindings to [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) v8.0.0, the neural-network kernel library of Arm, for the [nanoglia](https://github.com/AlixANNERAUD/nanoglia) crates.

- The integer kernels of CMSIS-NN (int8, int16 and int4) are vendored in `cmsis-nn/` (see `cmsis-nn/VENDORED.md`) and compiled with the [`cc`](https://crates.io/crates/cc) crate for any target, always optimized (`-O2`, or the `s`/`z`/`3` level of the profile), even in debug builds.
- On Arm Cortex-M the compiler flags expose the DSP extension (Armv7E-M: Cortex-M4/M7; Armv8-M Mainline: Cortex-M33) and Helium (Armv8.1-M with the `mve` target feature), so CMSIS-NN uses its optimized code paths. Elsewhere (host, RISC-V, Xtensa...) it compiles its portable C code.
- The float kernels are left out: they would pull in libm. No function calls `malloc` or any other part of the C library except `memcpy`/`memset`/`memmove`, which `compiler_builtins` provides.
- The bindings are written by hand (no bindgen at build time); a test checks the size, alignment and field offsets of every struct against the C compiler.

Cross-compiling needs a C compiler for the target and its C headers (`stdint.h`, `string.h`...), for example `arm-none-eabi-gcc` with newlib's headers, passed through the usual `CC_<target>`/`CFLAGS_<target>` variables; nothing links the C library.

The build script exports the include directory as `DEP_CMSIS_NN_INCLUDE` to crates that compile C code against CMSIS-NN.

## License

The Rust code of this crate and the vendored CMSIS-NN sources (`cmsis-nn/LICENSE`) are both Apache-2.0.
