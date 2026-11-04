# Use nanoglia from C

The `nanoglia-c` crate is the C API of nanoglia: a static library,
`libnanoglia_c.a`, its header
[`crates/nanoglia-c/include/nanoglia.h`](../../crates/nanoglia-c/include/nanoglia.h)
and a CMake project. Your C or C++ program, or your firmware, loads a
`.tflite` model at run time and runs it with the same kernels and the same
bit-exact results as the Rust API. The library takes every byte it uses
from **one buffer you provide**: it calls no `malloc`, and needs no C
library at all on a microcontroller.

This guide builds the library on your computer and for a firmware,
explains where the memory goes and how long each pointer must live, then
runs inferences in one call or one operator at a time and handles the
errors. The full description of each function is in the header.

## Build and link on your computer

You need Rust (the toolchain of the repository: `mise install`), CMake
3.21 or later and a C compiler. Add the directory of the crate to your
CMake project and link the `nanoglia::nanoglia` target:

```cmake
add_subdirectory(nanoglia/crates/nanoglia-c)
target_link_libraries(app PRIVATE nanoglia::nanoglia)
```

The target runs `cargo rustc` to build the archive, adds the header
directory to your include path and the system libraries the Rust standard
library needs (`pthread`, `dl` and `m` on Linux). The CMake options:

| Option | Default | Effect |
| --- | --- | --- |
| `NANOGLIA_CARGO_TARGET` | empty (the host) | Rust target triple to build for, such as `thumbv7em-none-eabihf` |
| `NANOGLIA_STD` | `ON` on the host, `OFF` with a target | Link the Rust standard library (needed on a host) |
| `NANOGLIA_PANIC_HANDLER` | `ON` | Without `NANOGLIA_STD`, include a panic handler ([Panics](#panics)) |
| `NANOGLIA_INT8_ONLY` | `OFF` | Leave out the float32 and int16 operators (features `cmsis-nn`, `kernels` only) |
| `NANOGLIA_ESP_NN` | `OFF` | Use the ESP-NN kernels first (feature `esp-nn`) |
| `NANOGLIA_CARGO_PROFILE` | `release` | Cargo profile |
| `NANOGLIA_CARGO_FLAGS` | empty | Extra Cargo arguments, such as `--features single-rounding` |
| `NANOGLIA_CARGO_TARGET_DIR` | `<build>/cargo` | Cargo target directory |
| `NANOGLIA_BUILD_TESTS`, `NANOGLIA_BUILD_EXAMPLES` | `ON` when top level | The C tests and the keyword-spotting example |

To use an installed copy instead, install it and find the package:

```sh
cmake -S crates/nanoglia-c -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /opt/nanoglia
```

```cmake
find_package(nanoglia 0.1 REQUIRED)  # with CMAKE_PREFIX_PATH=/opt/nanoglia
target_link_libraries(app PRIVATE nanoglia::nanoglia)
```

Without CMake, build the archive with Cargo and link it yourself:

```sh
cargo rustc -p nanoglia-c --release --lib --crate-type staticlib --features std
cc app.c -Icrates/nanoglia-c/include target/release/libnanoglia_c.a -lpthread -ldl -lm
```

`cargo rustc -p nanoglia-c --release --lib --crate-type staticlib
--features std -- --print native-static-libs` lists the system libraries
to link on your platform.

Run the example to check the build: it loads the MLPerf Tiny
keyword-spotting model from a 16-byte-aligned static buffer and runs one
golden input step by step.

```console
$ cmake -S crates/nanoglia-c -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
$ build/nanoglia_keyword_spotting testdata/mlperf-tiny/kws_ref_model.tflite \
      testdata/mlperf-tiny/golden/kws_ref_model/sample_0/input_0.bin
graph 7712 bytes, context 16160 bytes, buffer 24335 of 32768 bytes
13 steps: class 9, probability 0.992
```

(Linux aarch64 host, 2026-11-04, commit `2c08c67`.) `ctest --test-dir build` runs it with
the C tests.

## Link it into a firmware

Point `NANOGLIA_CARGO_TARGET` at the Rust target of your microcontroller
(install it with `rustup target add`) and give CMSIS-NN a C compiler and
the C headers of the target, through the variables of the `cc` crate in
the environment of the build, as for a Rust firmware
([Run on a microcontroller](run-on-a-microcontroller.md#2-give-cmsis-nn-a-c-compiler)):

```sh
export CC_thumbv7em_none_eabihf=arm-none-eabi-gcc
export AR_thumbv7em_none_eabihf=arm-none-eabi-ar
export CFLAGS_thumbv7em_none_eabihf="-isystem /usr/include/newlib"
cmake -S crates/nanoglia-c -B build-firmware -DNANOGLIA_CARGO_TARGET=thumbv7em-none-eabihf \
      -DNANOGLIA_INT8_ONLY=ON -DNANOGLIA_BUILD_TESTS=OFF -DNANOGLIA_BUILD_EXAMPLES=OFF
cmake --build build-firmware
```

The archive is `build-firmware/cargo/thumbv7em-none-eabihf/release/libnanoglia_c.a`.
In a CMake firmware project that already uses your cross toolchain, add
the directory with these options and link `nanoglia::nanoglia`; with
another build system, link the archive with `--gc-sections` so that the
kernels your model does not use are dropped. Without `NANOGLIA_STD` the
archive is self-contained: it brings its own `memcpy` and `memset`
(from Rust's `compiler_builtins`) and references no `malloc`, `free` or
any other C library function. `mise run cross-c` checks it by linking
it with `arm-none-eabi-gcc -nostdlib` into a program that calls every
function; the Cortex-M4F DSP code paths of CMSIS-NN have not run on
hardware yet ([issue #2](https://github.com/AlixANNERAUD/nanoglia/issues/2)).

`NANOGLIA_INT8_ONLY` is enough for fully quantized int8 models such as
the MLPerf Tiny ones, and saves flash
([Cargo features](../reference/features.md)).

### Panics

nanoglia does not panic on any model, valid or not: a malformed model
returns an error. A panic would be a bug. A Rust static library for a
target without the standard library must still contain a panic handler,
and a program can contain only one:

- by default (`NANOGLIA_PANIC_HANDLER=ON`, Cargo feature
  `panic-handler`), the archive has one that loops forever, which your
  watchdog can catch;
- if your firmware links other Rust code, it cannot link a second Rust
  static library next to this one (both would define the panic handler
  and the Rust core library). Build one static library of your own
  instead: a Rust crate of type `staticlib` that depends on `nanoglia-c`
  with `default-features = false` and the backend features you need,
  re-exports it (`pub use nanoglia_c;`), and defines the program's
  `#[panic_handler]`. The functions of the C API are exported from it.
  That is why `nanoglia-c` is an rlib for Cargo and the CMake project
  builds the archive with `cargo rustc --crate-type staticlib`.

With `NANOGLIA_STD` (on a host), the standard library's handler prints the
message, and every function catches the panic and returns
`NANOGLIA_STATUS_UNKNOWN`.

## The memory

Everything lives in two buffers your program owns:

- **the model bytes**, on a 16-byte boundary (`NANOGLIA_MODEL_ALIGNMENT`):
  borrowed and never copied, so they can stay in flash. Weights are read
  in place, which is why they must be aligned; a misaligned model returns
  `NANOGLIA_STATUS_INVALID_ARGUMENT`;
- **the memory buffer**, of any alignment, which need not be initialized.
  `nanoglia_load()` lays it out like this:

```text
memory ──► │ padding │ handle │ graph block │ padding │ context arena │ unused │
             0 to 15   fixed    graph_bytes   0 to 15   context_bytes
             bytes     size
```

The handle (`nanoglia`, opaque) holds the loaded graph and one execution
context; its address is what `nanoglia_load()` writes to `nn`, aligned on
16 bytes inside your buffer. The graph block holds the kernel table, the
prepared parameters and the memory plan; the context arena the
activations, the inputs and outputs, and the scratch memory of the
kernels ([Memory](../reference/memory.md#what-the-blocks-hold)).
Nothing else is allocated, before, during or after an inference.

`nanoglia_memory_requirements()` computes the sizes without loading:

```c
static _Alignas(NANOGLIA_MODEL_ALIGNMENT) const uint8_t model[] = { /* .tflite */ };
static unsigned char memory[32 * 1024];

nanoglia_memory needs;
nanoglia_error error;
/* `memory` serves as the planner's scratch memory meanwhile (about 20 bytes per tensor). */
if (nanoglia_memory_requirements(model, sizeof model, memory, sizeof memory, &needs, &error) ==
    NANOGLIA_STATUS_OK) {
    /* needs.memory_bytes: the size nanoglia_load() needs, at any alignment of `memory` */
}
```

`memory_bytes` is exact: the C tests load the keyword-spotting model into
a buffer of exactly `memory_bytes` bytes at all 16 alignments, and check
that one byte less fails when the buffer starts one byte past a 16-byte
boundary. It is the `arena_bytes(1)` of the Rust API plus the size of the
handle: 448 bytes on a 64-bit host (Linux aarch64), 224 bytes on
`thumbv7em-none-eabihf` (both measured on 2026-11-04 at commit `2c08c67`). For the
keyword-spotting model on that host, `memory_bytes` is 24335 bytes
(`graph_bytes` 7712, `context_bytes` 16160); the sizes of the other
models and on 32-bit targets are in [Memory](../reference/memory.md#mlperf-tiny-footprints).
As in Rust, compute them for your target and features, or print them on
the device: CMSIS-NN may ask for other scratch sizes with the DSP
extension ([Size the memory](size-memory.md)).

### Lifetimes

- The model bytes and the memory buffer must stay valid and unchanged,
  except through the API, from `nanoglia_load()` until
  `nanoglia_release()`. They must not overlap.
- `nanoglia_release()` ends the handle; then the model and the buffer are
  yours again. Releasing twice, or using a released handle, returns
  `NANOGLIA_STATUS_INVALID_ARGUMENT` as long as the buffer has not been
  reused: the handle carries a marker that release clears.
- The `data` pointer of an input (`nanoglia_input()`) stays the same until
  release: fill it in place. Write the inputs **before each inference**:
  the memory plan reuses their memory once the operators have read them.
- The contents of an output (`nanoglia_output()`) hold the result once
  `nanoglia_compute()` succeeds or `nanoglia_step()` sets `done`, until
  the next inference starts. They are read-only.
- Tensor names (`name`, `name_length`) point into the model bytes and are
  not NUL-terminated.
- A handle is used by one thread (or task) at a time. Handles in different
  buffers are independent and can run at the same time; each holds its own
  copy of the graph block.

## Run an inference

```c
nanoglia *nn;
if (nanoglia_load(model, sizeof model, memory, sizeof memory, &nn, &error) != NANOGLIA_STATUS_OK) {
    /* see Errors */
}

nanoglia_tensor input;
nanoglia_input(nn, 0, &input);           /* or nanoglia_input_index(nn, "input_1", 7, &i) */
/* input.element_type == NANOGLIA_TYPE_INT8, input.rank == 4, input.dims = {1, 49, 10, 1},
   input.scale and input.zero_point quantize the features */
memcpy(input.data, features, input.size);

if (nanoglia_compute(nn, &error) == NANOGLIA_STATUS_OK) {
    nanoglia_tensor output;
    nanoglia_output(nn, 0, &output);
    const int8_t *scores = output.data;
    /* probability of class i: output.scale * (scores[i] - output.zero_point) */
}

nanoglia_release(nn);
```

### One operator at a time

A long inference blocks for as long as all its operators take.
`nanoglia_step()` runs one operator per call, so that a super-loop or an
RTOS task does something else in between, as `ExecutionContext::step` in
Rust ([Run long inferences](run-long-inferences.md)):

```c
bool done = false;
while (!done) {
    if (nanoglia_step(nn, &done, &error) != NANOGLIA_STATUS_OK) {
        break;              /* the inference is abandoned; the next step starts a new one */
    }
    poll_radio();           /* anything the firmware must do between two operators */
}
```

- `nanoglia_operator_count()` is the number of steps of an inference.
- Stop at any time: stop calling `nanoglia_step()`, or call
  `nanoglia_cancel()`. The next step, or `nanoglia_compute()`, starts a new
  inference on the current inputs; write them again first, since the
  operators that ran may have reused their memory.
- Do not change the inputs while an inference is in progress.
- A single operator is never split: a large convolution runs to its end.

## Errors

Every function checks its pointers (null), its handle and its sizes, and
returns a `nanoglia_status`; the counts return 0 for an invalid handle.
`nanoglia_memory_requirements()`, `nanoglia_load()`,
`nanoglia_compute()` and `nanoglia_step()` also fill a `nanoglia_error`
when you pass one (`NULL` otherwise), with the precise reason of the Rust
`Error` ([Errors](../reference/errors.md)):

```c
nanoglia_error error;
if (nanoglia_load(model, size, memory, sizeof memory, &nn, &error) != NANOGLIA_STATUS_OK) {
    printf("%s: %s\n", nanoglia_status_description(error.status),
           nanoglia_detail_description(error.detail));
    if (error.detail == NANOGLIA_DETAIL_UNSUPPORTED_OPERATOR) {
        printf("operator %u (TFLite builtin code %d)\n", (unsigned)error.index, (int)error.code);
    } else if (error.detail == NANOGLIA_DETAIL_OUT_OF_MEMORY) {
        printf("%zu bytes did not fit\n", error.size);
    }
}
```

The statuses are the wasi-nn error codes, numbered from 1, after
`NANOGLIA_STATUS_OK` (0):

| Status | Returned when |
| --- | --- |
| `NANOGLIA_STATUS_INVALID_ARGUMENT` | A null pointer, a handle that is not a live one, a model off its 16-byte boundary, an input or output index out of range |
| `NANOGLIA_STATUS_INVALID_ENCODING` | The bytes are not a valid `.tflite` model (truncated, corrupted, another format) |
| `NANOGLIA_STATUS_UNSUPPORTED_OPERATION` | No enabled backend supports an operator, a type or the structure of the model |
| `NANOGLIA_STATUS_TOO_LARGE` | The memory buffer, or the scratch memory of `nanoglia_memory_requirements()`, is too small |
| `NANOGLIA_STATUS_NOT_FOUND` | No input or output has the name given to `nanoglia_input_index()` or `nanoglia_output_index()` |
| `NANOGLIA_STATUS_RUNTIME_ERROR` | A kernel library failed while computing (detail `NANOGLIA_DETAIL_BACKEND`, with its name and status) |
| `NANOGLIA_STATUS_UNKNOWN` | An internal inconsistency, or a panic caught with `NANOGLIA_STD` |
| `NANOGLIA_STATUS_TIMEOUT`, `NANOGLIA_STATUS_SECURITY` | Not returned yet |

Each `nanoglia_detail` value says which fields of `nanoglia_error` it sets
(`index`, `code`, `size`, `actual`, `backend`); the header lists them.
Models are untrusted input: the C tests load truncated models and models
with random bytes flipped, and every load returns an error or a model
that runs, never a crash.

## See also

- [`crates/nanoglia-c/examples/keyword_spotting.c`](../../crates/nanoglia-c/examples/keyword_spotting.c)
  and [`crates/nanoglia-c/tests/test.c`](../../crates/nanoglia-c/tests/test.c):
  the example and the tests, compiled with `-Wall -Wextra -Wpedantic
  -Werror`.
- [Development tasks](../reference/development-tasks.md): `test-c`,
  `cross-c`, `header`, `check-header`.
- [Cargo features](../reference/features.md#nanoglia-c).
