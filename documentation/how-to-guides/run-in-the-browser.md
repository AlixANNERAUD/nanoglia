# Run in the browser

nanoglia builds for WebAssembly: `wasm32-unknown-unknown` for browsers
and other JavaScript hosts, `wasm32-wasip1` for WASI runtimes such as
wasmtime. This guide sets up the C compiler CMSIS-NN needs, turns on
SIMD128, and runs nanoglia under wasmtime.

## 1. Get a C compiler for wasm32

The `cmsis-nn` feature (on by default) compiles CMSIS-NN's C sources for
the target, so the build needs a `clang` that targets wasm32 and the
declarations of a few C headers (`stdint.h`, `string.h`...). CMSIS-NN
calls nothing from the C library except `memcpy`, `memset` and
`memmove`, which Rust's `compiler_builtins` provides, so no C library is
linked: the headers are enough.

[wasi-sdk](https://github.com/WebAssembly/wasi-sdk) has both: a `clang`
for wasm32 and, in its sysroot, the headers of wasi-libc. Download a
release (nanoglia's CI uses `wasi-sdk-33`) and put its `bin` directory on
the `PATH` of your wasm32 builds only: its `clang` and `llvm-ar` would
otherwise replace your system's for host builds too (nanoglia's
`mise.toml` declares it as a tool of the wasm tasks only, for this
reason). Then point `cc` at it (replace `$WASI_SDK` with the directory of
wasi-sdk):

```sh
export CC_wasm32_unknown_unknown=clang
export AR_wasm32_unknown_unknown=llvm-ar
export CFLAGS_wasm32_unknown_unknown="-isystem $WASI_SDK/share/wasi-sysroot/include/wasm32-wasip1"
```

Make sure the `clang` and `llvm-ar` found on the `PATH` are wasi-sdk's,
or give their full paths.

Without `cmsis-nn` (`default-features = false, features = ["kernels"]`),
no C compiler is needed, but the int8 convolutions, fully-connected
layers, pooling and int8 softmax are then unsupported
([Cargo features](../reference/features.md)).

## 2. Build for the browser

```sh
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
```

In nanoglia's repository, `mise run cross-wasm32` builds all the
libraries this way, with wasi-sdk-33, and CI runs it on every change.

nanoglia needs nothing from the JavaScript host. Expose your own
functions to JavaScript with the tool of your choice; inside them, load
the model with `Graph::load_in` (or `Graph::load` with feature `alloc`,
since Rust's standard library brings a global allocator to this
target), and keep the graph and its execution context between calls.
The model bytes must stay alive as long as the graph and start on a
16-byte boundary ([Memory](../reference/memory.md#model-bytes)): copy
bytes received from JavaScript into a buffer aligned that way.

## 3. Turn on SIMD128

WebAssembly's 128-bit SIMD instructions speed up both the Rust kernels
and CMSIS-NN's C code. Enable them for both compilers:

```sh
export RUSTFLAGS="-C target-feature=+simd128"
export CFLAGS_wasm32_unknown_unknown="-isystem $WASI_SDK/share/wasi-sysroot/include/wasm32-wasip1 -msimd128"
cargo build --release --target wasm32-unknown-unknown
```

The resulting module only runs on hosts that support fixed-width SIMD.
Under wasmtime, SIMD128 makes the MLPerf Tiny models 2 to 3.3 times
faster than the scalar build ([Performance](../reference/performance.md)).

## 4. Run under WASI with wasmtime

`wasm32-wasip1` gives a program files, a clock and standard output,
which makes it convenient to test and benchmark outside a browser.
wasi-sdk provides a ready-configured compiler for it:

```sh
rustup target add wasm32-wasip1
export CC_wasm32_wasip1=wasm32-wasip1-clang
export AR_wasm32_wasip1=llvm-ar
export CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --dir=."
cargo run --release --target wasm32-wasip1
```

`--dir=.` lets the program read the current directory, for example a
model file. For SIMD128, add `-C target-feature=+simd128` to `RUSTFLAGS`
and set `CFLAGS_wasm32_wasip1=-msimd128`.

In nanoglia's repository:

- `mise run test-wasip1` runs the whole test suite, golden cases
  included, compiled to `wasm32-wasip1` under wasmtime 50.0.0;
- `mise run bench-wasm` times the MLPerf Tiny models under wasmtime,
  scalar and with SIMD128.

## See also

- [Development tasks](../reference/development-tasks.md)
- [Performance](../reference/performance.md)
