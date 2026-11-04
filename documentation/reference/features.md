# Cargo features

Every Cargo feature of the nanoglia crates. Your application normally
depends on `nanoglia` only and sets its features; the other crates
receive theirs from it.

## nanoglia

Default features: `cmsis-nn`, `kernels`, `float`, `int16`.

| Feature | Default | Enables |
| --- | :---: | --- |
| `alloc` | no | The functions that use the Rust global allocator: `Graph::load`, `Graph::init_execution_context`, `MemoryRequirements::compute`, and the re-export `nanoglia::Global`. Forwards `nanoglia-core/alloc`. Without it, every allocation goes through an allocator you pass (`Graph::load_in`, `init_execution_context_in`, `MemoryRequirements::compute_in`), for example an `Arena`. |
| `cmsis-nn` | yes | The CMSIS-NN backend (`nanoglia-cmsis-nn`), first in `Registry::default()` (after ESP-NN with `esp-nn`). Compiles the CMSIS-NN C sources shipped with `nanoglia-cmsis-nn-sys`, so it needs a C compiler and the C headers of the target ([Run on a microcontroller](../how-to-guides/run-on-a-microcontroller.md), [Run in the browser](../how-to-guides/run-in-the-browser.md)). |
| `esp-nn` | no | The ESP-NN backend (`nanoglia-esp-nn`), tried before CMSIS-NN in `Registry::default()`: ESP-NN's optimized kernels on `xtensa-esp32s3-*` targets, its ANSI C kernels elsewhere (slower than CMSIS-NN's portable C: enable it there for testing only). Compiles the ESP-NN C sources shipped with `nanoglia-esp-nn-sys` and CMSIS-NN's, whose kernels it runs while another context uses ESP-NN ([Supported operators](supported-operators.md#esp-nn)). Claims nothing with `single-rounding`. |
| `kernels` | yes | The pure Rust backend (`nanoglia-kernels`), after CMSIS-NN in `Registry::default()`. |
| `float` | yes | The f32 kernels of the Rust backend: `CONV_2D`, `DEPTHWISE_CONV_2D`, `TRANSPOSE_CONV`, `FULLY_CONNECTED`, pooling, `SOFTMAX`, `RESIZE_BILINEAR` and the f32 activations (`RELU`, `RELU6`, `RELU_N1_TO_1`, `LEAKY_RELU`, `LOGISTIC`, `TANH`, `HARD_SWISH`). Has an effect only with `kernels`. |
| `int16` | yes | The 16x8 kernels: CMSIS-NN `CONV_2D`, `FULLY_CONNECTED`, `ADD` and `MUL` on int16, and the Rust `FULLY_CONNECTED` 16x8 and int16 `SOFTMAX`. |
| `single-rounding` | no | Requantizes with a single rounding (`TFLITE_SINGLE_ROUNDING`) in every crate, CMSIS-NN included (`CMSIS_NN_USE_SINGLE_ROUNDING`), instead of the double rounding of TensorFlow Lite Micro, CMSIS-NN and the TFLite reference kernels. Results then differ from the golden data by design ([Bit-exactness](../explanation/bit-exactness.md#single-and-double-rounding)). |
| `defmt` | no | `defmt::Format` for `Error`, `ErrorCode`, `Detail`, `DataType`, `Dims`, `BuiltinOp`, `Padding`, `Activation` and `BackendKind`. Forwards `nanoglia-core/defmt`. |

`float` and `int16` only stop the backends from offering kernels: the
code that is no longer reachable is dropped by the linker. Some f32 and
int16 operators stay available without them, because they share a
kernel with other types:

- without `float`: f32 `ADD`, `SUB`, `MUL`, `MAXIMUM`, `MINIMUM`, `MEAN`,
  `PAD`, `CONCATENATION`, `RESHAPE`, `ARG_MAX`, `ARG_MIN`,
  `RESIZE_NEAREST_NEIGHBOR` and the `QUANTIZE` / `DEQUANTIZE` conversions;
- without `int16`: int16 `ADD`, `SUB` and `MUL` (Rust kernel, with
  broadcasting), `MEAN`, `RELU`, `RELU6`, `LEAKY_RELU`, `MAXIMUM`,
  `MINIMUM`, `PAD`, `ARG_MAX`, `ARG_MIN`, `RESIZE_NEAREST_NEIGHBOR`,
  `RESIZE_BILINEAR` and `QUANTIZE` / `DEQUANTIZE`.

The full matrix is in [Supported operators](supported-operators.md).

For an int8-only model, such as the MLPerf Tiny models, use:

```toml
nanoglia = { git = "https://github.com/AlixANNERAUD/nanoglia", default-features = false, features = ["cmsis-nn", "kernels"] }
```

This is what the Nano 33 BLE example does (plus `defmt`). CI runs the
library tests and the memory tests in this configuration (job
"Test (int8 only)" of `ci.yml`). Keep `kernels`: even int8 models need the
Rust backend for `RESHAPE` and the other operators CMSIS-NN does not
have.

Without `cmsis-nn`, no C compiler is needed, and every int8 convolution,
fully-connected layer, pooling and int8 softmax becomes unsupported: the
Rust backend has no int8 version of them.

## nanoglia-core

| Feature | Default | Enables |
| --- | :---: | --- |
| `alloc` | no | `allocator_api2`'s `alloc` feature, so `Global` (the Rust global allocator) implements `Allocator`. |
| `single-rounding` | no | `quant::SINGLE_ROUNDING = true`: `quant::multiply_by_quantized_multiplier` rounds once. |
| `defmt` | no | `defmt::Format` for the error, tensor and operator types. Adds the `defmt` 1 dependency. |

## nanoglia-cmsis-nn

| Feature | Default | Enables |
| --- | :---: | --- |
| `int16` | yes (off through `nanoglia` unless its `int16` is on) | The 16x8 kernels: `CONV_2D`, `FULLY_CONNECTED`, `ADD`, `MUL`. |
| `single-rounding` | no | `nanoglia-cmsis-nn-sys/single-rounding` and `nanoglia-core/single-rounding`. |

## nanoglia-cmsis-nn-sys

| Feature | Default | Enables |
| --- | :---: | --- |
| `single-rounding` | no | Compiles CMSIS-NN with `CMSIS_NN_USE_SINGLE_ROUNDING`. |

The build script always compiles the integer kernels only
(`ARM_NN_ENABLE_F32=0`, `ARM_NN_ENABLE_F16=0`), optimized (`-O2`, or the
`s`, `z` or `3` level of the profile) even in debug builds, with the DSP
or Helium flags of the Arm target.

## nanoglia-kernels

| Feature | Default | Enables |
| --- | :---: | --- |
| `float` | yes (off through `nanoglia` unless its `float` is on) | The f32 kernels listed for `nanoglia/float`. |
| `int16` | yes (off through `nanoglia` unless its `int16` is on) | `rust/fully_connected_16x8` and `rust/softmax_i16`. |
| `single-rounding` | no | `nanoglia-core/single-rounding`. |

## nanoglia-esp-nn, nanoglia-esp-nn-sys

No features. `nanoglia-esp-nn` depends on `nanoglia-cmsis-nn` for its
fallback kernels. `nanoglia-esp-nn-sys` compiles ESP-NN's ANSI C kernels
on every target and adds its optimized C and assembly kernels on
`xtensa-esp32s3-*` targets, always optimized like CMSIS-NN, without
`ESP_PLATFORM` (no ESP-IDF, FreeRTOS or heap dependency).

## nanoglia-tflite

No features.

## nanoglia-c

The C API ([Use nanoglia from C](../how-to-guides/use-from-c.md)).
Default features: `cmsis-nn`, `kernels`, `float`, `int16`. The CMake
project of the crate sets the features from its options (`NANOGLIA_STD`,
`NANOGLIA_PANIC_HANDLER`, `NANOGLIA_INT8_ONLY`, `NANOGLIA_ESP_NN`,
`NANOGLIA_CARGO_FLAGS`).

| Feature | Default | Enables |
| --- | :---: | --- |
| `cmsis-nn`, `esp-nn`, `kernels`, `float`, `int16`, `single-rounding` | as `nanoglia` | The same feature of `nanoglia`. |
| `std` | no | Links the standard library: needed to build the static library (`cargo rustc --crate-type staticlib`) for a target with `std`, such as the host. Every function then catches a panic and returns `NANOGLIA_STATUS_UNKNOWN`. |
| `panic-handler` | no | A `#[panic_handler]` that loops forever, compiled without `std`: needed to build the static library for a target without `std`. Leave it out when another crate of the program provides the panic handler. |

Neither `std` nor `panic-handler` is on by default: the crate is also an
rlib, which a Rust program links with its own panic handler or `std`.
