# nanoglia

**Run TensorFlow Lite models loaded at run time on microcontrollers, in `no_std` Rust, with the
best kernels of each chip.**

`nanoglia` is a TinyML inference engine: it reads a `.tflite` (LiteRT) model from flash, RAM, an
SD card or the network, without copying its weights, and runs it with Arm
[CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) on Cortex-M, Espressif
[ESP-NN](https://github.com/espressif/esp-nn) on the ESP32-S3 (feature `esp-nn`), and pure Rust
kernels for the rest. No RTOS, no HAL, no vendor SDK, and every byte of memory comes from the
allocator you pass, a static buffer included. Results are bit-exact with TensorFlow Lite.

```rust,ignore
use nanoglia::{Arena, Encoding, Graph, Target};

static MODEL: &[u8] = include_bytes!("model.tflite"); // on a 16-byte boundary in practice

let mut memory = [0u8; 24 * 1024]; // MemoryRequirements tells how much a model needs
let arena = Arena::new(&mut memory);

let graph = Graph::load_in(&[MODEL], Encoding::Autodetect, Target::Auto, &arena)?;
let mut ctx = graph.init_execution_context_in(&arena)?;

ctx.input_mut(0)?.as_mut_slice::<i8>()?.copy_from_slice(&features);
ctx.compute()?; // allocates nothing
let scores = ctx.output(0)?.as_slice::<i8>()?;
```

- **Models loaded at run time:** update or swap a model without reflashing the firmware.
- **Your memory only:** a [`Graph`](https://docs.rs/nanoglia/latest/nanoglia/struct.Graph.html)
  makes one allocation, each
  [`ExecutionContext`](https://docs.rs/nanoglia/latest/nanoglia/struct.ExecutionContext.html) one
  more, `compute` none. They come from the global allocator (feature `alloc`), your own, or an
  [`Arena`](https://docs.rs/nanoglia/latest/nanoglia/struct.Arena.html) over a static buffer,
  sized ahead of time with
  [`MemoryRequirements`](https://docs.rs/nanoglia/latest/nanoglia/struct.MemoryRequirements.html).
- **Accelerated:** CMSIS-NN (Cortex-M DSP and Helium), ESP-NN (ESP32-S3 SIMD), portable code
  everywhere else, WebAssembly included.
- **Bit-exact with TensorFlow Lite:** the outputs match the model run in Python, checked on every
  commit against golden outputs, including the MLPerf Tiny models.
- **Extensible:** a [`Registry`](https://docs.rs/nanoglia/latest/nanoglia/struct.Registry.html)
  of your own puts your backends first; their kernels may borrow your drivers. An
  [`Observer`](https://docs.rs/nanoglia/latest/nanoglia/trait.Observer.html) passed to
  `compute_with` sees each operator, to time it, read its output or stop at a deadline.
- **Non-blocking:** `step()` runs one operator at a time, `compute_async()` yields to any async
  executor between operators.
- **Safe on untrusted models:** the flatbuffer is fully verified before use; errors carry a
  wasi-nn [`ErrorCode`](https://docs.rs/nanoglia/latest/nanoglia/enum.ErrorCode.html) and a
  precise [`Detail`](https://docs.rs/nanoglia/latest/nanoglia/enum.Detail.html), never a panic.

| Cargo feature | Default | Effect |
| --- | --- | --- |
| `cmsis-nn` | ✅ | CMSIS-NN kernels (Arm DSP/Helium on Cortex-M, portable C elsewhere) |
| `kernels` | ✅ | Pure Rust kernels for the operators the C libraries lack |
| `float` | ✅ | float32 operators |
| `int16` | ✅ | int16 (16x8) operators |
| `esp-nn` | | ESP-NN kernels, tried first (ESP32-S3 SIMD on `xtensa-esp32s3`) |
| `alloc` | | `Graph::load` and friends, with the Rust global allocator |
| `single-rounding` | | `TFLITE_SINGLE_ROUNDING` requantization instead of TensorFlow Lite Micro's double rounding |
| `defmt` | | `defmt::Format` for the error types |

An int8-only firmware can use `default-features = false, features = ["cmsis-nn", "kernels"]`.

## Owned models and persistent contexts

`OwnedGraph::load_in` takes an aligned `nanoglia::core::Block` filled by the caller,
retaining its allocation without copying model bytes. `GraphExecutionContext::new_in`
accepts an owned graph handle (including `alloc::sync::Arc` with feature `alloc`) and
allocates its arena once. `ExecutionContext` is the context that borrows its `Graph`.
See the workspace documentation's memory reference for ownership and allocation details.

**Learn more:** the [repository](https://github.com/AlixANNERAUD/nanoglia), the
[first-inference tutorial](https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/tutorials/first-inference.md),
[supported operators](https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/reference/supported-operators.md),
[sizing the memory](https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/how-to-guides/size-memory.md),
[running on a microcontroller](https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/how-to-guides/run-on-a-microcontroller.md)
and [writing a backend](https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/how-to-guides/write-a-backend.md).

Licensed under Apache-2.0.
