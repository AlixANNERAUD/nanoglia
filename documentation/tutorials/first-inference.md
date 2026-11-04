# Your first inference

In this tutorial you run a neural network with nanoglia: the keyword
spotting model of the [MLPerf Tiny](https://github.com/mlcommons/tiny)
benchmark, which recognizes spoken words such as "yes", "no" or "stop"
from audio features. Everything runs on your computer, so you need no
board. You first let nanoglia take its memory from the heap, then give it
a fixed static buffer instead, as firmware on a microcontroller would.

By the end you will have loaded a `.tflite` model, written an input,
run an inference, read and dequantized the output, and checked how much
memory nanoglia used.

## What you need

- Rust 1.89 or later (`rustc --version` prints it).
- A C compiler: nanoglia compiles the CMSIS-NN kernels, written in C,
  for your computer. `gcc` or `clang` both work (on Debian or Ubuntu,
  the `build-essential` package).
- Git and curl.

## 1. Create the project

```sh
cargo new first-inference
cd first-inference
```

## 2. Add nanoglia

nanoglia is not published on crates.io yet: add it from its Git
repository. The `alloc` feature lets nanoglia use the Rust global
allocator, which is the simplest start on a computer.

```sh
cargo add nanoglia --git https://github.com/AlixANNERAUD/nanoglia --features alloc
```

`Cargo.toml` now has:

```toml
[dependencies]
nanoglia = { git = "https://github.com/AlixANNERAUD/nanoglia", version = "0.1.0", features = ["alloc"] }
```

## 3. Download the model

The keyword spotting model is in nanoglia's test data:

```sh
curl -L -o kws_ref_model.tflite https://raw.githubusercontent.com/AlixANNERAUD/nanoglia/main/testdata/mlperf-tiny/kws_ref_model.tflite
```

It is a 53936-byte int8 model (a DS-CNN). Its input is 49 frames of 10
MFCC coefficients, computed from one second of audio; its output is a
score for each of 12 classes: ten words, silence and unknown (the class
order comes from the
[MLPerf Tiny training script](https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/get_dataset.py)).

## 4. Load the model

Replace the content of `src/main.rs` with the start of the program:

```rust
use nanoglia::{Encoding, Graph, Target};

/// The model bytes, placed on a 16-byte boundary so that its constant buffers are aligned.
#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);

static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!("../kws_ref_model.tflite"));

/// The 12 classes of the model, in the order of its outputs.
const LABELS: [&str; 12] = [
    "down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "silence", "unknown",
];

fn main() -> Result<(), nanoglia::Error> {
    // Load the model: verify it, choose a kernel for each operator, plan the memory.
    let graph = Graph::load(&[&MODEL.0], Encoding::Autodetect, Target::Auto)?;
    println!(
        "{} operators, graph {} B, context {} B",
        graph.operator_count(),
        graph.graph_bytes(),
        graph.context_bytes()
    );
    for i in 0..graph.operator_count() {
        println!("  operator {i}: {}", graph.kernel_name(i).unwrap_or("?"));
    }
```

`include_bytes!` puts the model in the program. nanoglia never copies
it: the graph reads the weights where they are, so the model must stay
alive as long as the graph. The `Aligned` wrapper places its first byte
on a 16-byte boundary; nanoglia reads constant buffers (such as int32
biases) as typed arrays and refuses a model whose buffers are not
aligned for their type.

`Graph::load` takes the model bytes, their encoding (here detected from
the bytes) and where to run it (`Target::Auto`: the best kernels
available). It checks the whole file, chooses a kernel for every
operator, and plans where each intermediate result will live.

## 5. Write the input

Continue `main` with:

```rust
    // What the model expects.
    let input = graph.input(0)?;
    println!(
        "input: {:?} {:?}, scale {}, zero point {}",
        input.dtype,
        input.dims,
        input.quant.scale(),
        input.quant.zero_point()
    );

    // The execution context holds the activations of one inference.
    let mut ctx = graph.init_execution_context()?;

    // Stand-in features: 49 frames of 10 MFCC coefficients, all 0.0, quantized to int8.
    let features = [0.0f32; 49 * 10];
    let (scale, zero_point) = (input.quant.scale(), input.quant.zero_point());
    let mut tensor = ctx.input_mut(0)?;
    for (q, &x) in tensor.as_mut_slice::<i8>()?.iter_mut().zip(&features) {
        *q = ((x / scale).round() as i32 + zero_point).clamp(-128, 127) as i8;
    }
```

The model is quantized: its input is int8, and a real value `x` is
stored as `round(x / scale) + zero_point`. The loop applies this rule to
the features. A real application would compute MFCC features from a
microphone; here every feature is 0.0.

An execution context holds the memory of one inference. `input_mut`
gives a view of the input directly in that memory, and `as_mut_slice`
checks that the element type you ask for (`i8`) is the tensor's.

## 6. Run the model and read the output

Finish `main` with:

```rust
    ctx.compute()?;

    // The output is quantized too: dequantize it to read probabilities.
    let output = ctx.output(0)?;
    let quantization = output.quantization();
    let (scale, zero_point) = (quantization.scale(), quantization.zero_point());
    for (label, &q) in LABELS.iter().zip(output.as_slice::<i8>()?) {
        let probability = scale * (i32::from(q) - zero_point) as f32;
        println!("{label:>8}: {q:4} -> {probability:.3}");
    }
    Ok(())
}
```

`compute` runs every operator in turn and allocates nothing. The output
is int8 as well; `scale * (q - zero_point)` turns it back into a real
value, here a probability between 0 and 1.

## 7. Run it

```sh
cargo run --release
```

The first build compiles CMSIS-NN, which takes a little while. Then the
program prints:

```text
13 operators, graph 7712 B, context 16160 B
  operator 0: cmsis-nn/conv2d_s8
  operator 1: cmsis-nn/depthwise_conv2d_s8
  operator 2: cmsis-nn/conv2d_s8
  operator 3: cmsis-nn/depthwise_conv2d_s8
  operator 4: cmsis-nn/conv2d_s8
  operator 5: cmsis-nn/depthwise_conv2d_s8
  operator 6: cmsis-nn/conv2d_s8
  operator 7: cmsis-nn/depthwise_conv2d_s8
  operator 8: cmsis-nn/conv2d_s8
  operator 9: cmsis-nn/average_pool2d_s8
  operator 10: rust/reshape
  operator 11: cmsis-nn/fully_connected_s8
  operator 12: cmsis-nn/softmax_s8
input: I8 [1, 49, 10, 1], scale 0.5847029, zero point 83
    down: -112 -> 0.062
      go: -112 -> 0.062
    left: -124 -> 0.016
      no: -121 -> 0.027
     off: -114 -> 0.055
      on: -112 -> 0.062
   right: -125 -> 0.012
    stop: -107 -> 0.082
      up: -110 -> 0.070
     yes: -124 -> 0.016
 silence: -128 -> 0.000
 unknown:   10 -> 0.539
```

The kernel list shows how nanoglia split the work: the convolutions,
pooling, fully-connected layer and softmax run on CMSIS-NN, and the
reshape, which CMSIS-NN does not have, on nanoglia's own Rust kernels.
With features that do not look like speech, the model answers "unknown".
The two sizes on the first line are all the memory nanoglia allocated:
7712 bytes for the graph and 16160 for the execution context. The
graph sizes are those of a 64-bit computer; a 32-bit microcontroller
needs slightly less.

## 8. Run without a heap

Microcontroller firmware often has no heap. nanoglia can take its memory
from any allocator you give it, including `Arena`, which hands out a
buffer you own. Change the program in four places.

Import `Arena`:

```rust
use nanoglia::{Arena, Encoding, Graph, Target};
```

Add a static buffer after `MODEL`:

```rust
/// All the memory nanoglia may use: the graph and one execution context.
static mut MEMORY: [u8; 32 * 1024] = [0; 32 * 1024];
```

At the start of `main`, create the arena and load the model into it with
`load_in`:

```rust
    // SAFETY: `main` runs once and is the only user of `MEMORY`.
    let arena = Arena::new(unsafe { &mut *core::ptr::addr_of_mut!(MEMORY) });

    // Load the model: verify it, choose a kernel for each operator, plan the memory.
    let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
```

Create the execution context in the arena too, and print how much of it
was used:

```rust
    let mut ctx = graph.init_execution_context_in(&arena)?;
    println!("arena: {} of {} bytes used", arena.used(), arena.capacity());
```

Run it again:

```sh
cargo run --release
```

The scores are the same, and a new line appears:

```text
arena: 23879 of 32768 bytes used
```

The graph and the context took 23872 bytes from the buffer, plus a few
bytes to start on a 16-byte boundary (up to 15, depending on where the
linker placed `MEMORY`). nanoglia allocated nothing else. If the buffer is too
small, `load_in` or `init_execution_context_in` returns an error whose
code is `ErrorCode::TooLarge`.

## What you did

You loaded a TFLite model at run time without copying it, ran it with
CMSIS-NN and Rust kernels, converted its input and output between real
values and int8, and ran it from a fixed buffer with no heap.

## Next steps

- [Size the memory](../how-to-guides/size-memory.md) of a static buffer
  exactly, for your own model.
- [Run on a microcontroller](../how-to-guides/run-on-a-microcontroller.md):
  the same model on an Arduino Nano 33 BLE.
- [Convert a PyTorch model](../how-to-guides/convert-a-pytorch-model.md)
  to a `.tflite` file nanoglia can run.
- [Memory](../reference/memory.md): why inputs must be written before
  every `compute`.
