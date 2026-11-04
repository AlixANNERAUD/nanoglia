# nanoglia

Run TensorFlow Lite models **loaded at runtime** on microcontrollers, with one `#![no_std]` API
modelled on [wasi-nn](https://github.com/WebAssembly/wasi-nn) and the best kernels of each chip.

```rust,ignore
use nanoglia::{Arena, Encoding, Graph, Target};

static MODEL: &[u8] = include_bytes!("model.tflite"); // or bytes read from flash, SD card, network...

let mut memory = [0u8; 64 * 1024];
let arena = Arena::new(&mut memory);

let graph = Graph::load_in(&[MODEL], Encoding::Autodetect, Target::Auto, &arena)?;
let mut ctx = graph.init_execution_context_in(&arena)?;

ctx.input_mut(0)?.as_mut_slice::<i8>()?.copy_from_slice(&features);
ctx.compute()?; // allocates nothing
let scores = ctx.output(0)?.as_slice::<i8>()?;
```

- **Zero-copy:** the graph borrows the model bytes; weights stay where they are (flash or RAM).
- **Your memory only:** a graph makes one allocation, each execution context one more, and
  `compute` none. They come from the allocator you pass: the global allocator (feature `alloc`),
  your own, or an [`Arena`] over a static buffer. [`MemoryRequirements`] tells, without loading
  the model, how large that buffer must be.
- **Accelerated:** CMSIS-NN (Cortex-M DSP/Helium, portable C elsewhere), ESP-NN (ESP32-S3, feature `esp-nn`), pure Rust kernels for the rest.
- **Safe on untrusted models:** the flatbuffer is fully verified before use; errors, never panics.

Errors carry a wasi-nn [`ErrorCode`] plus a precise [`Detail`].
