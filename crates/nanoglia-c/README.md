# nanoglia-c

**C API of [nanoglia](https://github.com/AlixANNERAUD/nanoglia): run TensorFlow Lite models loaded at run time from C or C++, with every byte in one buffer you provide.** A static library (`libnanoglia_c.a`), its header [`include/nanoglia.h`](include/nanoglia.h) and a CMake project; nothing calls `malloc`, and a malformed model returns an error, never crashes.

```c
#include "nanoglia.h"

static _Alignas(NANOGLIA_MODEL_ALIGNMENT) const uint8_t model[] = { /* .tflite bytes */ };
static unsigned char memory[24 * 1024]; /* nanoglia_memory_requirements() gives the size */

nanoglia *nn;
nanoglia_error error;
if (nanoglia_load(model, sizeof model, memory, sizeof memory, &nn, &error) != NANOGLIA_STATUS_OK) {
    printf("%s: %s\n", nanoglia_status_description(error.status),
           nanoglia_detail_description(error.detail));
}

nanoglia_tensor input, output;
nanoglia_input(nn, 0, &input);
memcpy(input.data, features, input.size);   /* write the input in place */
nanoglia_compute(nn, &error);               /* or nanoglia_step() until done */
nanoglia_output(nn, 0, &output);            /* int8 scores, output.scale, output.zero_point */
nanoglia_release(nn);
```

With CMake, add the directory and link the `nanoglia` target, which builds the static library with Cargo:

```cmake
add_subdirectory(nanoglia/crates/nanoglia-c)
target_link_libraries(app PRIVATE nanoglia::nanoglia)
```

For a microcontroller, set `-DNANOGLIA_CARGO_TARGET=thumbv7em-none-eabihf` (or another Rust target) and link the archive into your firmware. Without CMake:

```sh
cargo rustc -p nanoglia-c --release --lib --crate-type staticlib --features std            # host
cargo rustc -p nanoglia-c --release --lib --crate-type staticlib --features panic-handler \
    --target thumbv7em-none-eabihf                                                   # firmware
```

The crate is an rlib for Cargo: the static library is built with `cargo rustc --crate-type staticlib`, so that a Rust program, or a firmware that links several Rust libraries through one static library of its own, can depend on it and provide its own panic handler.

| Feature | Default | Enables |
| --- | :---: | --- |
| `cmsis-nn`, `kernels`, `float`, `int16`, `esp-nn`, `single-rounding`, `all-ops`, `op-*` | as in `nanoglia` | The backends and operators of [`nanoglia`](https://crates.io/crates/nanoglia). `all-ops` (default) builds the kernels of every operator; without it, the `op-*` features name the operators to build (`op-conv-2d`...), to save flash. The CMake option `NANOGLIA_OPERATORS` (`CONV_2D;SOFTMAX`) sets them. |
| `std` | no | Links the standard library, needed to build the static library for a host; a panic (a bug) returns `NANOGLIA_STATUS_UNKNOWN` instead of aborting. |
| `panic-handler` | no | A panic handler that loops forever, for a static library built for a target without `std`. |

The guide [Use nanoglia from C](https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/how-to-guides/use-from-c.md) covers the memory layout, lifetimes, the step API and errors. The header is generated from `src/lib.rs` by cbindgen (`mise run header`) and committed.
