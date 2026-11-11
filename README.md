# nanoglia: run TensorFlow Lite models on microcontrollers, in Rust

[![crates.io](https://img.shields.io/crates/v/nanoglia.svg)](https://crates.io/crates/nanoglia)
[![docs.rs](https://img.shields.io/docsrs/nanoglia)](https://docs.rs/nanoglia)
[![CI](https://github.com/AlixANNERAUD/nanoglia/actions/workflows/ci.yml/badge.svg)](https://github.com/AlixANNERAUD/nanoglia/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/AlixANNERAUD/nanoglia/graph/badge.svg)](https://codecov.io/gh/AlixANNERAUD/nanoglia)
[![no_std](https://img.shields.io/badge/no__std-no%20alloc-success)](documentation/reference/memory.md)
[![MSRV 1.89](https://img.shields.io/badge/MSRV-1.89-informational)](Cargo.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Load a `.tflite` neural network at run time and run it on a microcontroller, without a heap,
an RTOS or a vendor SDK.** `nanoglia` is a `no_std` TinyML inference engine for Rust: it reads
TensorFlow Lite (LiteRT) models straight from flash, RAM, an SD card or the network, without
copying their weights, and runs them with the fastest kernels of your chip: Arm
[CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) on Cortex-M (STM32, nRF52/nRF53, RP2350,
NXP, Renesas...) and Espressif [ESP-NN](https://github.com/espressif/esp-nn) on the ESP32-S3,
with pure Rust kernels for everything else. One API, close to
[wasi-nn](https://github.com/WebAssembly/wasi-nn), from a Cortex-M0 to WebAssembly.

```rust,ignore
let graph = Graph::load_in(&[MODEL], Encoding::Autodetect, Target::Auto, &arena)?;
let mut ctx = graph.init_execution_context_in(&arena)?;
ctx.input_mut(0)?.as_mut_slice::<i8>()?.copy_from_slice(&features);
ctx.compute()?; // allocates nothing
let scores = ctx.output(0)?.as_slice::<i8>()?;
```

| At a glance | |
| --- | --- |
| Models | TensorFlow Lite / LiteRT `.tflite`, **loaded at run time**: update a model over the air without reflashing the firmware |
| Accuracy | **Bit-exact with TensorFlow Lite**: 373 of 376 golden cases pass (quantized outputs identical byte for byte, float32 ones within a relative 1e-5), including the [MLPerf Tiny](https://github.com/mlcommons/tiny) keyword-spotting, streaming-wakeword, visual-wake-words and image-classification models ([why the other 3 differ](documentation/explanation/bit-exactness.md)) |
| Memory | **One allocation per model, one per inference context, none per inference**, all from the allocator you pass: the global allocator, your own, or a static buffer. Keyword spotting needs **23.3 KiB** for the model ([measured](documentation/reference/memory.md)), 28.3 KiB with the stack in a Cortex-M4 firmware ([measured](documentation/reference/performance.md#keyword-spotting-firmware)) |
| Speed | CMSIS-NN (Cortex-M DSP and Helium), ESP-NN (ESP32-S3 SIMD); in WebAssembly, SIMD128 makes the models **2 to 3.3 times faster** ([measured](documentation/reference/performance.md)). On a laptop CPU (Linux virtual machine), the fastest of the five Rust engines compared on the four MLPerf Tiny models ([compared](documentation/reference/performance.md#compared-with-other-rust-engines)) |
| Types | int8 (the TinyML standard), int16 (16x8) and float32, 38 operators ([list](documentation/reference/supported-operators.md)) |
| Safety | Untrusted models are fully verified before use: errors, never panics. The model reader is `#![forbid(unsafe_code)]` and fuzzed in CI |
| Targets | Cortex-M0+, M4F, M33, ESP32-S3 (Xtensa), RISC-V, WebAssembly and desktops, each built in CI ([see below](#supported-platforms)) |
| Languages | Rust, and C or C++ through a static library, a header and a CMake project ([see below](#c-and-c)) |

## Why nanoglia?

TinyML on microcontrollers usually means one of two compromises. TensorFlow Lite Micro is C++, and
reaching each vendor's optimized kernels means its build system and SDK (ESP-IDF for ESP-NN,
STM32Cube for ST, ...). The pure Rust engines compile the model *into* the firmware at build time,
so changing a model means rebuilding and reflashing, and they leave the hardware acceleration on
the table. `nanoglia` keeps the model a file and the firmware a plain Rust binary.

- **Models loaded at run time.** Store models in flash, receive them over BLE, Wi-Fi or a serial
  link, pick one at boot: `Graph::load_in` verifies the bytes and runs them. Weights stay where
  they are (zero copy), even in memory-mapped flash.
- **The best kernels of each chip, no SDK.** CMSIS-NN and ESP-NN ship with the crates and are
  compiled by Cargo with `cc`: no ESP-IDF, no STM32Cube, no CMake. Each operator runs on the fastest
  backend that supports it exactly, and falls back to portable code otherwise.
- **Your own backends.** A backend of your own, listed before the bundled ones, can claim any
  operator; its kernels may borrow your drivers (an accelerator, a DMA controller) with no global
  state. `compute_with` calls an observer around each operator, to time it or check its output
  ([write a backend](documentation/how-to-guides/write-a-backend.md)).
- **Your memory, and only yours.** `#![no_std]`, no global allocator needed. A graph takes one block,
  an execution context one more, and `compute` none. Use an `Arena` over a `static` buffer, sized
  ahead of time with `MemoryRequirements`: no fragmentation, no surprises at run time.
- **The same numbers as on your PC.** Every kernel reproduces TensorFlow Lite's fixed-point
  arithmetic bit for bit, so a model validated in Python gives the same outputs on the device.
  376 golden cases generated with LiteRT's reference kernels run on every commit, on the host and in
  WebAssembly.
- **Safe on untrusted input.** A corrupted or malicious `.tflite` file returns an error with a
  wasi-nn error code and a precise reason. It never panics, never reads out of bounds.

## Quick start

Add [`nanoglia`](https://crates.io/crates/nanoglia) from crates.io:

```toml
[dependencies]
nanoglia = { version = "0.3", features = ["alloc"] }
```

Run the MLPerf Tiny keyword-spotting model on your computer:

```rust,ignore
use nanoglia::{Encoding, Graph, Target};

#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);
static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!("kws_ref_model.tflite"));

fn main() -> Result<(), nanoglia::Error> {
    let graph = Graph::load(&[&MODEL.0], Encoding::Autodetect, Target::Auto)?;
    let mut ctx = graph.init_execution_context()?;

    ctx.input_mut(0)?.as_mut_slice::<i8>()?.fill(0); // 49 frames x 10 MFCC features
    ctx.compute()?;

    let scores = ctx.output(0)?;
    println!("scores: {:?}", scores.as_slice::<i8>()?);
    Ok(())
}
```

On a microcontroller, drop `alloc` and give the graph a static buffer instead:

```rust,ignore
use nanoglia::{Arena, Encoding, Graph, Target};

static mut MEMORY: [u8; 24 * 1024] = [0; 24 * 1024]; // MemoryRequirements::arena_bytes(1) = 23 791

let arena = Arena::new(unsafe { &mut *core::ptr::addr_of_mut!(MEMORY) });
let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
let mut ctx = graph.init_execution_context_in(&arena)?;
```

The [first-inference tutorial](documentation/tutorials/first-inference.md) walks through both,
and [`examples/nano33ble`](examples/nano33ble) is a complete firmware for the Arduino Nano 33 BLE.

## C and C++

The [`nanoglia-c`](crates/nanoglia-c) crate is a C API with the same memory contract: a static
library, its header [`nanoglia.h`](crates/nanoglia-c/include/nanoglia.h) and a CMake project. The
model and one buffer come from your program, and nothing calls `malloc`:

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
memcpy(input.data, features, input.size);
nanoglia_compute(nn, &error); /* or nanoglia_step() until done, between other work */
nanoglia_output(nn, 0, &output);
```

With CMake, add the directory and link the `nanoglia::nanoglia` target, which builds the library
with Cargo for your computer or, with `-DNANOGLIA_CARGO_TARGET=thumbv7em-none-eabihf`, for a
firmware:

```cmake
add_subdirectory(nanoglia/crates/nanoglia-c)
target_link_libraries(app PRIVATE nanoglia::nanoglia)
```

[Use nanoglia from C](documentation/how-to-guides/use-from-c.md) covers the build, the memory
layout, lifetimes, the step API and errors.

## Supported platforms

| Target | Example hardware | Kernels | Checked in CI |
| --- | --- | --- | --- |
| `thumbv7em-none-eabihf` | Cortex-M4F/M7: nRF52840 (Arduino Nano 33 BLE), STM32F4/H7 | CMSIS-NN with DSP | Library build, the Nano 33 BLE firmware, golden cases under an emulator |
| `thumbv8m.main-none-eabihf` | Cortex-M33: nRF5340, STM32U5, RP2350 | CMSIS-NN (DSP with `-C target-cpu=cortex-m33`) | Library build, golden cases under an emulator (with DSP) |
| `thumbv6m-none-eabi` | Cortex-M0+: RP2040, STM32G0 | CMSIS-NN, portable C | Library build, golden cases under an emulator |
| `xtensa-esp32s3-none-elf` | ESP32-S3 | ESP-NN with PIE SIMD (feature `esp-nn`), CMSIS-NN | Library build with the optimized ESP-NN kernels |
| `riscv32imc-unknown-none-elf` | ESP32-C3 and other RV32 cores | CMSIS-NN, portable C | Library build, golden cases under an emulator |
| `wasm32-unknown-unknown` | Browsers | Portable C and Rust, SIMD128 | Library build |
| `wasm32-wasip1` | WASI runtimes (wasmtime...) | Portable C and Rust | Whole test suite under wasmtime |
| Linux, macOS, Windows | Desktop, CI, embedded Linux | Portable C and Rust | Whole test suite (Linux) |

Every library builds without any global allocator, and a CI job checks that no C code references
`malloc`. The golden cases run in CI under the Unicorn emulator for Cortex-M4F, M33 (CMSIS-NN's DSP
code paths), M0+ and riscv32imc (`mise run test-emulated`); the ESP32-S3 SIMD code paths are only
built. An emulator says nothing about timing, caches, flash wait states or alignment faults:
`mise run test-hardware-nano33ble` flashes an Arduino Nano 33 BLE over USB, runs every golden case
on it and times each operator
([run the golden cases on a board](documentation/how-to-guides/run-the-golden-cases-on-a-board.md)).
It has run once, on one board (2026-11-10): 373 of the 376 golden cases matched LiteRT (bit for bit;
the float32 ones within their tolerance of 1e-5), the other 3 the known LiteRT difference of
`ad01_int8` (issue #1), with CMSIS-NN's DSP code paths on a real Cortex-M4F; the times are in [Performance](documentation/reference/performance.md#on-an-arduino-nano-33-ble).
No other board has run anything, and CMSIS-NN's Helium paths and ESP-NN's ESP32-S3 kernels have run
nowhere: [issue #2](https://github.com/AlixANNERAUD/nanoglia/issues/2) tracks that validation.

## Compared to other options

As of November 2026:

| | Runtime-loaded models | Rust API | Hardware kernels | `no_std`, no heap | Notes |
| --- | --- | --- | --- | --- | --- |
| **nanoglia** | ✅ `.tflite` | ✅ | CMSIS-NN, ESP-NN | ✅ | Bit-exact with TensorFlow Lite |
| [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro) | ✅ `.tflite` | ❌ C++ | CMSIS-NN, Ethos-U, ESP-NN (through ESP-IDF) | ✅ (C++) | The reference C++ runtime |
| [tfmicro](https://crates.io/crates/tfmicro) | ✅ `.tflite` | ✅ bindings | CMSIS-NN (Arm) | ✅ | Rust bindings to a 2020 TensorFlow Lite Micro; does not build with current toolchains (bindgen 0.53 and libclang 22) |
| [MicroFlow](https://crates.io/crates/microflow) | ❌ compiled in at build time | ✅ | ❌ | ✅ | Pure Rust, model turned into code by a macro |
| [embedded-nn](https://crates.io/crates/embedded-nn) | ❌ compiled ahead of time | ✅ | ❌ | ✅ | Pure Rust kernels and toolchain |
| [ember-rs](https://crates.io/crates/ember-infer-core) | ❌ compiled in at build time | ✅ | ESP-NN (separate `ember-esp` backend) | ❌ the reference backend allocates | Fork of MicroFlow with a backend trait |
| [tract](https://crates.io/crates/tract-tflite) | ✅ ONNX, TFLite... | ✅ | ❌ | ❌ needs `std` | Desktop and server inference |

Measured on the MLPerf Tiny keyword-spotting model, with the same inputs, profiles and machine for
every engine ([method, versions and all four models](documentation/reference/performance.md#compared-with-other-rust-engines)):

| | Outputs vs TensorFlow Lite (3 classifier models) | Host time | Cortex-M4 firmware: flash | Firmware: RAM |
| --- | --- | ---: | ---: | ---: |
| **nanoglia** | Identical on all three | 249 µs | 160.8 KiB | 28.3 KiB |
| tract (`tract-tflite` 0.23.9) | Differ (its output quantization is not the model's in our harness); ResNet does not load | 1392 µs | — (needs `std`) | — |
| MicroFlow 0.1.3 | Off by up to 6; ResNet not supported (ADD) | 2355 µs | 44.2 KiB | 88.6 KiB |
| embedded-nn 0.2.3 | Scores differ by up to 204 (its code generator fixes the SOFTMAX parameters) | 1962 µs | 35.7 KiB | 17.1 KiB |
| ember-rs 0.1.2 | Off by up to 25 | 4749 µs | 41.6 KiB | 79.2 KiB |

Host times come from a Linux aarch64 virtual machine, not from a microcontroller; the firmware
(flash includes the model, RAM is static data plus peak stack) ran under a Cortex-M4 emulator.
The other engines were run through our own harness, as documented by their authors: `mise run
bench-compare` reproduces everything, and corrections are welcome in an issue.

Pick `nanoglia` when the model must stay a file (updates, several models, user-provided models),
the firmware must stay a plain Rust binary without a vendor SDK, outputs must match TensorFlow
Lite, and memory use must be known ahead of time. Pick an ahead-of-time engine when the model
never changes and flash is tight: their keyword-spotting firmware is 3.6 to 4.5 times smaller
than nanoglia's with every int8 kernel, which runs models it only sees at run time. A firmware
that knows the operators of its models can leave the other kernels out
([reduce the code size](documentation/how-to-guides/reduce-code-size.md)).

## Documentation

The [documentation](documentation/README.md) follows the [Diátaxis](https://diataxis.fr) layout.

| | |
| --- | --- |
| **Tutorials** | [Run your first inference](documentation/tutorials/first-inference.md), [from training to the device](documentation/tutorials/train-and-deploy.md) |
| **How-to guides** | [Check whether a model runs](documentation/how-to-guides/check-whether-a-model-runs.md), [size the memory](documentation/how-to-guides/size-memory.md), [convert a PyTorch model](documentation/how-to-guides/convert-a-pytorch-model.md), [run on a microcontroller](documentation/how-to-guides/run-on-a-microcontroller.md), [run the golden cases on a board](documentation/how-to-guides/run-the-golden-cases-on-a-board.md), [reduce the code size](documentation/how-to-guides/reduce-code-size.md), [run in the browser](documentation/how-to-guides/run-in-the-browser.md), [run long inferences without blocking](documentation/how-to-guides/run-long-inferences.md), [write a backend](documentation/how-to-guides/write-a-backend.md), [use nanoglia from C](documentation/how-to-guides/use-from-c.md) |
| **Explanation** | [Architecture](documentation/explanation/architecture.md), [bit-exactness with TensorFlow Lite](documentation/explanation/bit-exactness.md) |
| **Reference** | [Supported operators](documentation/reference/supported-operators.md), [Cargo features](documentation/reference/features.md), [errors](documentation/reference/errors.md), [memory](documentation/reference/memory.md), [performance](documentation/reference/performance.md), [development tasks](documentation/reference/development-tasks.md) |

The API documentation is on [docs.rs](https://docs.rs/nanoglia).

## Crates

| Crate | Role |
| --- | --- |
| [`nanoglia`](crates/nanoglia) | The library: load a model, create execution contexts, compute |
| [`nanoglia-core`](crates/nanoglia-core) | Tensor, operator and error types, kernel-backend interface, TFLite-exact quantization math, arena allocator |
| [`nanoglia-tflite`](crates/nanoglia-tflite) | Zero-copy, allocation-free, panic-free TensorFlow Lite model verifier and reader |
| [`nanoglia-cmsis-nn`](crates/nanoglia-cmsis-nn) / [`-sys`](crates/nanoglia-cmsis-nn-sys) | CMSIS-NN backend: int8 and int16 convolutions, fully-connected layers, pooling, softmax... |
| [`nanoglia-esp-nn`](crates/nanoglia-esp-nn) / [`-sys`](crates/nanoglia-esp-nn-sys) | ESP-NN backend for the ESP32-S3 (feature `esp-nn`) |
| [`nanoglia-kernels`](crates/nanoglia-kernels) | Pure Rust kernels: the operators the C libraries lack, and float32 |
| [`nanoglia-c`](crates/nanoglia-c) | C API: a static library, its header and a CMake project |
| [`nanoglia-tools`](crates/nanoglia-tools) | The `nanoglia check` command: whether a model runs, with which kernels and how much memory |

## Frequently asked questions

**How do I run a TensorFlow Lite model on a microcontroller in Rust?** Add `nanoglia`, place the
`.tflite` bytes on a 16-byte boundary, and call `Graph::load_in` then `compute` as in the
[quick start](#quick-start). No C++ toolchain, no TensorFlow Lite Micro build, no vendor SDK: Cargo
compiles the C kernels it needs with your target's C compiler.

**Can I update the model without reflashing the firmware?** Yes: that is what nanoglia is built
for. The model is data. Write a new `.tflite` file to flash or receive it over the network, and
load it.

**Which models work?** Fully quantized int8 models (the usual TinyML export), 16x8 models and float32
models, using the [supported operators](documentation/reference/supported-operators.md): convolutions,
depthwise and transpose convolutions, fully-connected layers, pooling, softmax, element-wise
operations, activations, reshapes, transposes, slices, gathers, splits, padding, means, resizing,
arg max/min, quantization... Loading
fails with the index of the first unsupported operator otherwise.

**I train in PyTorch. Can I use nanoglia?** Yes: export to `.tflite` with Google's
[`litert-torch`](https://github.com/google-ai-edge/litert-torch), see
[convert a PyTorch model](documentation/how-to-guides/convert-a-pytorch-model.md). ONNX models can
be converted with [onnx2tf](https://github.com/PINTO0309/onnx2tf). The data-movement operators these
exports produce (`TRANSPOSE`, `SLICE`, `STRIDED_SLICE`, `GATHER`, `SPLIT`, `SPLIT_V`) are supported.

**How much RAM and flash does it need?** RAM is exactly what `MemoryRequirements` reports for your
model: 23.3 KiB for MLPerf Tiny keyword spotting, 92 KiB for visual wake words. The Nano 33 BLE
example firmware, built with the kernels of the six operators of its model only, is 138 KB of
flash, 54 KB of which is the model; with every int8 operator it is 274 KB
([code size](documentation/reference/memory.md#code-size),
[reduce the code size](documentation/how-to-guides/reduce-code-size.md)).

**Why TensorFlow Lite and not ONNX?** A `.tflite` file can be read in place, without parsing or
copying, and its int8 arithmetic is defined precisely enough to be reproduced bit for bit; the
whole microcontroller ecosystem (MLPerf Tiny, Edge Impulse, vendor tools) uses it. See
[bit-exactness](documentation/explanation/bit-exactness.md).

**Does a long inference block my firmware?** No need: `step()` runs one operator at a time
from any main loop, `compute_async()` yields to an async executor (embassy, RTIC...) between
operators, and an observer can stop it at a deadline. See
[run long inferences](documentation/how-to-guides/run-long-inferences.md).

**Does it run in a web browser?** Yes, as WebAssembly, with SIMD128 for speed: see
[run in the browser](documentation/how-to-guides/run-in-the-browser.md).

## Contributing

Issues and pull requests are welcome. The repository uses [mise](https://mise.jdx.dev) for its
toolchain and tasks: `mise install`, `mise run submodules` (CMSIS-NN and ESP-NN are git
submodules), then `mise run check` before pushing. [`AGENTS.md`](AGENTS.md)
describes the invariants (memory contract, bit-exactness, no panics) and the tests, and
[development tasks](documentation/reference/development-tasks.md) lists every task. Pull request
titles follow [Conventional Commits](https://www.conventionalcommits.org).

## License

Apache-2.0, see [`LICENSE`](LICENSE). The CMSIS-NN and ESP-NN sources (git submodules of the `-sys` crates) are Apache-2.0 as well.
