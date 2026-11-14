# Cargo features

Every Cargo feature of the nanoglia crates. Your application normally
depends on `nanoglia` only and sets its features; the other crates
receive theirs from it.

## nanoglia

Default features: `cmsis-nn`, `kernels`, `float`, `int16`, `all-ops`.

| Feature | Default | Enables |
| --- | :---: | --- |
| `alloc` | no | The functions that use the Rust global allocator: `Graph::load`, `Graph::init_execution_context`, `MemoryRequirements::compute`, and the re-export `nanoglia::Global`. Forwards `nanoglia-core/alloc`. Without it, every allocation goes through an allocator you pass (`Graph::load_in`, `init_execution_context_in`, `MemoryRequirements::compute_in`), for example an `Arena`. |
| `cmsis-nn` | yes | The CMSIS-NN backend (`nanoglia-cmsis-nn`), first in `Registry::default()` (after ESP-NN with `esp-nn`). Compiles the CMSIS-NN C sources shipped with `nanoglia-cmsis-nn-sys`, so it needs a C compiler and the C headers of the target ([Run on a microcontroller](../how-to-guides/run-on-a-microcontroller.md), [Run in the browser](../how-to-guides/run-in-the-browser.md)). |
| `esp-nn` | no | The ESP-NN backend (`nanoglia-esp-nn`), tried before CMSIS-NN in `Registry::default()`: ESP-NN's optimized kernels on `xtensa-esp32s3-*` targets, its ANSI C kernels elsewhere (slower than CMSIS-NN's portable C: enable it there for testing only). Compiles the ESP-NN C sources shipped with `nanoglia-esp-nn-sys` and CMSIS-NN's, whose kernels it runs while another context uses ESP-NN ([Supported operators](supported-operators.md#esp-nn)). Claims nothing with `single-rounding`. |
| `kernels` | yes | The pure Rust backend (`nanoglia-kernels`), after CMSIS-NN in `Registry::default()`. |
| `float` | yes | The f32 kernels of the Rust backend: `CONV_2D`, `DEPTHWISE_CONV_2D`, `TRANSPOSE_CONV`, `FULLY_CONNECTED`, pooling, `SOFTMAX`, `RESIZE_BILINEAR`, the f32 unary operators (`ABS`, `NEG`, `SQRT`, `RSQRT`, `EXP`, `LOG`), the f32 activations (`RELU`, `RELU6`, `RELU_N1_TO_1`, `LEAKY_RELU`, `PRELU`, `LOGISTIC`, `TANH`, `HARD_SWISH`, `GELU`), `BATCH_MATMUL` and `L2_NORMALIZATION`. Has an effect only with `kernels`. |
| `int16` | yes | The 16x8 kernels: CMSIS-NN `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `ADD` and `MUL` on int16, and the Rust `FULLY_CONNECTED` 16x8, int16 `SOFTMAX`, `LOGISTIC` and `TANH`. |
| `single-rounding` | no | Requantizes with a single rounding (`TFLITE_SINGLE_ROUNDING`) in every crate, CMSIS-NN included (`CMSIS_NN_USE_SINGLE_ROUNDING`), instead of the double rounding of TensorFlow Lite Micro, CMSIS-NN and the TFLite reference kernels. Results then differ from the golden data by design ([Bit-exactness](../explanation/bit-exactness.md#single-and-double-rounding)). |
| `defmt` | no | `defmt::Format` for `Error`, `ErrorCode`, `Detail`, `DataType`, `Dims`, `BuiltinOp`, `Padding`, `Activation` and `BackendKind`. Forwards `nanoglia-core/defmt`. |
| `all-ops` | yes | Every operator: all the `op-*` features ([Operator features](#operator-features)). |
| `op-*` (67 features) | yes, through `all-ops` | The kernels of one operator, in every enabled backend that has some: `op-conv-2d` for `CONV_2D`. Without the feature of an operator, no backend claims it. |

`float` and `int16` only stop the backends from offering kernels: the
code that is no longer reachable is dropped by the linker. Some f32 and
int16 operators stay available without them, because they share a
kernel with other types:

- without `float`: f32 `ADD`, `SUB`, `MUL`, `DIV`, `SQUARED_DIFFERENCE`,
  `MAXIMUM`, `MINIMUM`, `MEAN`, `SUM`, `REDUCE_MAX`, `REDUCE_MIN`,
  `PAD`, the layout operators (`PACK`, `UNPACK`, `SPACE_TO_DEPTH`,
  `DEPTH_TO_SPACE`, `SPACE_TO_BATCH_ND`, `BATCH_TO_SPACE_ND`),
  `CONCATENATION`, `RESHAPE`, `ARG_MAX`, `ARG_MIN`,
  `RESIZE_NEAREST_NEIGHBOR`, the comparisons (`EQUAL`, `NOT_EQUAL`,
  `LESS`, `LESS_EQUAL`, `GREATER`, `GREATER_EQUAL`), `SELECT_V2`, `CAST`
  and the `QUANTIZE` / `DEQUANTIZE` conversions;
- without `int16`: int16 `ADD`, `SUB`, `MUL` and `SQUARED_DIFFERENCE`
  (Rust kernel, with broadcasting), `MEAN`, `SUM`, `REDUCE_MAX`,
  `REDUCE_MIN`, `PACK`, `UNPACK`, `SPACE_TO_BATCH_ND`,
  `BATCH_TO_SPACE_ND`, `RELU`, `RELU6`, `LEAKY_RELU`, `PRELU`, `ABS`,
  `MAXIMUM`, `MINIMUM`, `PAD`, `ARG_MAX`, `ARG_MIN`,
  `RESIZE_NEAREST_NEIGHBOR`, `RESIZE_BILINEAR`, `SELECT_V2`, `CAST` and
  `QUANTIZE` / `DEQUANTIZE`.

The full matrix is in [Supported operators](supported-operators.md).

For int8-only models, such as the MLPerf Tiny models, use:

```toml
nanoglia = { version = "0.3", default-features = false, features = ["cmsis-nn", "kernels", "all-ops"] }
```

CI runs the library tests and the memory tests in this configuration
(job "Test (int8 only)" of `ci.yml`). Keep `kernels`: even int8 models
need the Rust backend for `RESHAPE` and the other operators CMSIS-NN
does not have. Keep `all-ops`, or list the operators of your models
instead, as the Nano 33 BLE example does
([Operator features](#operator-features)): **without the default
features and without any `op-*` feature, the backends have no kernel and
no model loads.**

Without `cmsis-nn`, no C compiler is needed, and every int8 convolution,
fully-connected layer, pooling and int8 softmax becomes unsupported: the
Rust backend has no int8 version of them.

### Operator features

Models are loaded at run time, so the linker cannot tell which kernels a
firmware needs: it keeps every kernel a backend may select. Each
operator therefore has a Cargo feature, named after it in lower case with
dashes (`BuiltinOp::feature` returns the name: `op-conv-2d` for
`CONV_2D`, `op-relu-n1-to-1` for `RELU_N1_TO_1`). The feature puts the
operator in the dispatch table of the backends and compiles its kernels;
without it, no bundled backend claims the operator, a model that uses it
fails to load with `Detail::UnsupportedOp` ([Errors](errors.md)),
`ModelReport` reports it without a kernel, and its kernels, with the
CMSIS-NN and ESP-NN functions they call, are not linked.

`all-ops`, a default feature, turns them all on. To link only the
kernels of your models, turn the default features off and list the
backends, the types and the operators
([Reduce the code size](../how-to-guides/reduce-code-size.md), with
measured sizes in [Memory](memory.md#code-size)):

```toml
nanoglia = { version = "0.3", default-features = false, features = [
  "cmsis-nn", "kernels",
  "op-conv-2d", "op-depthwise-conv-2d", "op-average-pool-2d", "op-reshape", "op-fully-connected", "op-softmax",
] }
```

- Features only add: when two crates of a program ask for different
  operators, the program gets them all.
- An `op-*` feature covers every type of the operator, but its f32 and
  16x8 kernels still need `float` and `int16`: `op-conv-2d` alone runs
  int8 `CONV_2D` (CMSIS-NN); with `float`, f32 too (Rust kernel); with
  `int16`, 16x8 too. Without `float`, `op-conv-2d` adds nothing to the
  Rust backend.
- A feature is forwarded to every enabled backend that has kernels for
  the operator (last column). ESP-NN's kernels fall back to CMSIS-NN's,
  so with `esp-nn` the feature turns on the CMSIS-NN kernel as well.
- Operators that share a kernel have one feature each, and the shared
  code is compiled as soon as one of them is on: the unary operators
  (`ABS`, `NEG`, `SQRT`, `RSQRT`, `EXP`, `LOG`), the comparisons, the
  Rust `ADD`, `SUB`, `MUL`, `DIV`, `MAXIMUM`, `MINIMUM` and
  `SQUARED_DIFFERENCE`, `RESHAPE`, `SQUEEZE` and `EXPAND_DIMS`, and the
  other families of [Supported operators](supported-operators.md). The
  backend still only claims the operators whose own feature is on.
- A backend of your own ([Write a backend](../how-to-guides/write-a-backend.md))
  is not affected by these features.

`crates/nanoglia-tools/tests/op_features.rs` checks that every operator
with a kernel has its feature, in every crate, and `mise run
op-features` builds and tests each feature alone
([Development tasks](development-tasks.md)).

| Operator | Feature | Backends with kernels |
| --- | --- | --- |
| `ABS` | `op-abs` | `nanoglia-kernels` |
| `ADD` | `op-add` | `nanoglia-esp-nn`, `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `ARG_MAX` | `op-arg-max` | `nanoglia-kernels` |
| `ARG_MIN` | `op-arg-min` | `nanoglia-kernels` |
| `AVERAGE_POOL_2D` | `op-average-pool-2d` | `nanoglia-esp-nn`, `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `BATCH_MATMUL` | `op-batch-matmul` | `nanoglia-kernels` |
| `BATCH_TO_SPACE_ND` | `op-batch-to-space-nd` | `nanoglia-kernels` |
| `CAST` | `op-cast` | `nanoglia-kernels` |
| `CONCATENATION` | `op-concatenation` | `nanoglia-kernels` |
| `CONV_2D` | `op-conv-2d` | `nanoglia-esp-nn`, `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `DEPTHWISE_CONV_2D` | `op-depthwise-conv-2d` | `nanoglia-esp-nn`, `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `DEPTH_TO_SPACE` | `op-depth-to-space` | `nanoglia-kernels` |
| `DEQUANTIZE` | `op-dequantize` | `nanoglia-kernels` |
| `DIV` | `op-div` | `nanoglia-kernels` |
| `EQUAL` | `op-equal` | `nanoglia-kernels` |
| `EXP` | `op-exp` | `nanoglia-kernels` |
| `EXPAND_DIMS` | `op-expand-dims` | `nanoglia-kernels` |
| `FULLY_CONNECTED` | `op-fully-connected` | `nanoglia-esp-nn`, `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `GATHER` | `op-gather` | `nanoglia-kernels` |
| `GELU` | `op-gelu` | `nanoglia-kernels` |
| `GREATER` | `op-greater` | `nanoglia-kernels` |
| `GREATER_EQUAL` | `op-greater-equal` | `nanoglia-kernels` |
| `HARD_SWISH` | `op-hard-swish` | `nanoglia-kernels` |
| `L2_NORMALIZATION` | `op-l2-normalization` | `nanoglia-kernels` |
| `LEAKY_RELU` | `op-leaky-relu` | `nanoglia-kernels` |
| `LESS` | `op-less` | `nanoglia-kernels` |
| `LESS_EQUAL` | `op-less-equal` | `nanoglia-kernels` |
| `LOG` | `op-log` | `nanoglia-kernels` |
| `LOGISTIC` | `op-logistic` | `nanoglia-kernels` |
| `MAXIMUM` | `op-maximum` | `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `MAX_POOL_2D` | `op-max-pool-2d` | `nanoglia-esp-nn`, `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `MEAN` | `op-mean` | `nanoglia-kernels` |
| `MINIMUM` | `op-minimum` | `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `MUL` | `op-mul` | `nanoglia-esp-nn`, `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `NEG` | `op-neg` | `nanoglia-kernels` |
| `NOT_EQUAL` | `op-not-equal` | `nanoglia-kernels` |
| `PACK` | `op-pack` | `nanoglia-kernels` |
| `PAD` | `op-pad` | `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `PADV2` | `op-padv2` | `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `PRELU` | `op-prelu` | `nanoglia-kernels` |
| `QUANTIZE` | `op-quantize` | `nanoglia-kernels` |
| `REDUCE_MAX` | `op-reduce-max` | `nanoglia-kernels` |
| `REDUCE_MIN` | `op-reduce-min` | `nanoglia-kernels` |
| `RELU` | `op-relu` | `nanoglia-kernels` |
| `RELU6` | `op-relu6` | `nanoglia-kernels` |
| `RELU_N1_TO_1` | `op-relu-n1-to-1` | `nanoglia-kernels` |
| `RESHAPE` | `op-reshape` | `nanoglia-kernels` |
| `RESIZE_BILINEAR` | `op-resize-bilinear` | `nanoglia-kernels` |
| `RESIZE_NEAREST_NEIGHBOR` | `op-resize-nearest-neighbor` | `nanoglia-kernels` |
| `RSQRT` | `op-rsqrt` | `nanoglia-kernels` |
| `SELECT_V2` | `op-select-v2` | `nanoglia-kernels` |
| `SLICE` | `op-slice` | `nanoglia-kernels` |
| `SOFTMAX` | `op-softmax` | `nanoglia-esp-nn`, `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `SPACE_TO_BATCH_ND` | `op-space-to-batch-nd` | `nanoglia-kernels` |
| `SPACE_TO_DEPTH` | `op-space-to-depth` | `nanoglia-kernels` |
| `SPLIT` | `op-split` | `nanoglia-kernels` |
| `SPLIT_V` | `op-split-v` | `nanoglia-kernels` |
| `SQRT` | `op-sqrt` | `nanoglia-kernels` |
| `SQUARED_DIFFERENCE` | `op-squared-difference` | `nanoglia-kernels` |
| `SQUEEZE` | `op-squeeze` | `nanoglia-kernels` |
| `STRIDED_SLICE` | `op-strided-slice` | `nanoglia-kernels` |
| `SUB` | `op-sub` | `nanoglia-kernels` |
| `SUM` | `op-sum` | `nanoglia-kernels` |
| `TANH` | `op-tanh` | `nanoglia-kernels` |
| `TRANSPOSE` | `op-transpose` | `nanoglia-kernels` |
| `TRANSPOSE_CONV` | `op-transpose-conv` | `nanoglia-cmsis-nn`, `nanoglia-kernels` |
| `UNPACK` | `op-unpack` | `nanoglia-kernels` |

## nanoglia-core

| Feature | Default | Enables |
| --- | :---: | --- |
| `alloc` | no | `allocator_api2`'s `alloc` feature, so `Global` (the Rust global allocator) implements `Allocator`. |
| `single-rounding` | no | `quant::SINGLE_ROUNDING = true`: `quant::multiply_by_quantized_multiplier` rounds once. |
| `defmt` | no | `defmt::Format` for the error, tensor and operator types. Adds the `defmt` 1 dependency. |

## nanoglia-cmsis-nn

| Feature | Default | Enables |
| --- | :---: | --- |
| `int16` | yes (off through `nanoglia` unless its `int16` is on) | The 16x8 kernels: `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `ADD`, `MUL`. |
| `single-rounding` | no | `nanoglia-cmsis-nn-sys/single-rounding` and `nanoglia-core/single-rounding`. |
| `all-ops` | yes (off through `nanoglia` unless its `all-ops` is on) | The 13 `op-*` features of the crate. |
| `op-*` | yes, through `all-ops` | The kernels of one operator ([Operator features](#operator-features)): `op-conv-2d`, `op-depthwise-conv-2d`, `op-fully-connected`, `op-average-pool-2d`, `op-max-pool-2d`, `op-softmax`, `op-add`, `op-mul`, `op-maximum`, `op-minimum`, `op-pad`, `op-padv2`, `op-transpose-conv`. `CmsisNn::has_operator` tells whether one is on. |

The kernels of the public module `nanoglia_cmsis_nn::kernels` are there
with the feature of their operator; `ADD_S8` and `MUL_S8` with either of
`op-add` and `op-mul`, the two pooling kernels with either of
`op-average-pool-2d` and `op-max-pool-2d`.

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
| `int16` | yes (off through `nanoglia` unless its `int16` is on) | `rust/fully_connected_16x8`, `rust/softmax_i16`, `rust/logistic_i16` and `rust/tanh_i16`. |
| `single-rounding` | no | `nanoglia-core/single-rounding`. |
| `all-ops` | yes (off through `nanoglia` unless its `all-ops` is on) | The 67 `op-*` features of the crate. |
| `op-*` | yes, through `all-ops` | The kernels of one operator, every operator of [Operator features](#operator-features). `RustKernels::has_operator` tells whether one is on. |

## nanoglia-esp-nn

`nanoglia-esp-nn` depends on `nanoglia-cmsis-nn` for its fallback
kernels.

| Feature | Default | Enables |
| --- | :---: | --- |
| `all-ops` | yes (off through `nanoglia` unless its `all-ops` is on) | The 8 `op-*` features of the crate. |
| `op-*` | yes, through `all-ops` | The kernels of one operator ([Operator features](#operator-features)), and the same feature of `nanoglia-cmsis-nn` for the fallback: `op-conv-2d`, `op-depthwise-conv-2d`, `op-fully-connected`, `op-average-pool-2d`, `op-max-pool-2d`, `op-softmax`, `op-add`, `op-mul`. `EspNn::has_operator` tells whether one is on. |

## nanoglia-esp-nn-sys

No features. `nanoglia-esp-nn-sys` compiles ESP-NN's ANSI C kernels
on every target and adds its optimized C and assembly kernels on
`xtensa-esp32s3-*` targets, always optimized like CMSIS-NN, without
`ESP_PLATFORM` (no ESP-IDF, FreeRTOS or heap dependency).

## nanoglia-tflite

No features.

## nanoglia-c

The C API ([Use nanoglia from C](../how-to-guides/use-from-c.md)).
Default features: `cmsis-nn`, `kernels`, `float`, `int16`, `all-ops`.
The CMake project of the crate sets the features from its options
(`NANOGLIA_STD`, `NANOGLIA_PANIC_HANDLER`, `NANOGLIA_INT8_ONLY`,
`NANOGLIA_ESP_NN`, `NANOGLIA_OPERATORS`, `NANOGLIA_CARGO_FLAGS`).

| Feature | Default | Enables |
| --- | :---: | --- |
| `cmsis-nn`, `esp-nn`, `kernels`, `float`, `int16`, `single-rounding`, `all-ops`, `op-*` | as `nanoglia` | The same feature of `nanoglia`. The CMake option `NANOGLIA_OPERATORS` takes operator names (`CONV_2D;SOFTMAX`) and enables their `op-*` features; empty, it enables `all-ops`. |
| `std` | no | Links the standard library: needed to build the static library (`cargo rustc --crate-type staticlib`) for a target with `std`, such as the host. Every function then catches a panic and returns `NANOGLIA_STATUS_UNKNOWN`. |
| `panic-handler` | no | A `#[panic_handler]` that loops forever, compiled without `std`: needed to build the static library for a target without `std`. Leave it out when another crate of the program provides the panic handler. |

Neither `std` nor `panic-handler` is on by default: the crate is also an
rlib, which a Rust program links with its own panic handler or `std`.

## nanoglia-tools

The `nanoglia` command-line tool
([Check whether a model runs](../how-to-guides/check-whether-a-model-runs.md)).
It always builds the three backends (CMSIS-NN, ESP-NN, the Rust kernels)
and chooses among them at run time with `--backends`, so it has no
feature for them; the features below are fixed when the tool is built,
and its output lists those that are on. Default features: `float`,
`int16`, `all-ops`.

| Feature | Default | Enables |
| --- | :---: | --- |
| `float` | yes | Checks with the f32 kernels, as feature `float` of `nanoglia`. Forwards `nanoglia-kernels/float`. |
| `int16` | yes | Checks with the 16x8 kernels, as feature `int16` of `nanoglia`. Forwards `nanoglia-cmsis-nn/int16` and `nanoglia-kernels/int16`. |
| `single-rounding` | no | Checks with single-rounding requantization, as feature `single-rounding` of `nanoglia`: ESP-NN then claims no operator. |
| `all-ops` | yes | Checks with every operator, as feature `all-ops` of `nanoglia`. |
| `op-*` | yes, through `all-ops` | Checks with one operator, as the same feature of `nanoglia`: forwarded to the three backends. Without `all-ops`, an operator whose feature is off is reported without a kernel. |

Build it for one configuration with `-p`, which keeps the features of
the other crates of the workspace out: `cargo run -p nanoglia-tools
--no-default-features --features all-ops -- check model.tflite` checks
for an int8-only firmware; with `op-*` features in place of `all-ops`,
it checks for a firmware built with those operators.
