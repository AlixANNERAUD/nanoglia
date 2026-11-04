# nanoglia-esp-nn

**Espressif ESP-NN acceleration for TensorFlow Lite models on the ESP32-S3, in `no_std` Rust, without ESP-IDF:** int8 convolutions, fully-connected layers, pooling and softmax with the chip's SIMD instructions, bit-exact with TensorFlow Lite. Most applications depend on [`nanoglia`](https://crates.io/crates/nanoglia) with its `esp-nn` feature instead.

The [ESP-NN](https://github.com/espressif/esp-nn) kernel backend of [nanoglia](https://github.com/AlixANNERAUD/nanoglia).

[`BACKEND`] implements [`nanoglia_core::backend::Backend`]: it claims the int8 operators ESP-NN runs exactly like the TFLite reference kernels and leaves the others to the next backend. Its parameter computations are ported from esp-tflite-micro's ESP-NN kernels (`tensorflow/lite/micro/kernels/esp_nn/`) and TensorFlow Lite Micro; they run once, at prepare time. Nothing allocates: parameters live in the persistent buffer and ESP-NN works in the shared scratch buffer the interpreter provides.

On ESP32-S3 (`xtensa-esp32s3-*` targets) ESP-NN runs its optimized kernels (PIE SIMD instructions); on other targets its ANSI C reference kernels, which are mostly useful to test the backend: register it there only if you want to.

| Operator | Types | Conditions |
|---|---|---|
| `CONV_2D` | int8 in/out, int8 weights (per-tensor or per-channel), int32 bias | rank 4, no dilation, any stride/padding, grouped convolutions |
| `DEPTHWISE_CONV_2D` | same | rank 4, no dilation, any depth multiplier, stride, padding |
| `FULLY_CONNECTED` | int8 in/out, int8 weights `[out, in]` (zero point 0, per-tensor or per-channel), int32 bias | input flattened to `[batches, in]`, depths up to 65535, default weight format |
| `AVERAGE_POOL_2D`, `MAX_POOL_2D` | int8 | rank 4, same input/output quantization |
| `SOFTMAX` | int8 in/out | output zero point -128, scale 1/256 |
| `ADD`, `MUL` | int8 | inputs and output of the same shape (no broadcasting) |

Fused activations: none, `RELU`, `RELU6`, `RELU_N1_TO_1`. Dimensions are limited to 65535 (ESP-NN takes them as `uint16_t`), strides, filter sizes and paddings to 32767.

ESP-NN's other kernels are not used: its `relu6`, `hard_swish` and `logistic` do not compute TFLite's quantized result (`logistic` uses `expf`), and `mean` and the channel-broadcast `mul` have no CMSIS-NN counterpart for the fallback below.

## Concurrency: the lock and the CMSIS-NN fallback

ESP-NN keeps the scratch buffers of its kernels in global variables, so it is not reentrant. Each evaluation takes a global try-lock ([`try_lock`], an atomic flag; no critical section, never blocks), sets the scratch pointer and calls ESP-NN. If the lock is taken, because another thread, task, interrupt handler or core is running ESP-NN, the operator runs the [CMSIS-NN](https://crates.io/crates/nanoglia-cmsis-nn) kernel instead: both compute the TFLite result, bit for bit. The backend therefore claims an operator only if CMSIS-NN supports it too, and prepares both kernels (the persistent buffer holds both parameter sets, the scratch buffer fits either). Code that calls ESP-NN directly next to nanoglia should hold the lock too. The lock is a global rather than a field of the backend because the state it guards is: ESP-NN's C globals exist once per binary, so two backend values (or code outside nanoglia) must share one lock.

On targets without atomic compare-and-swap (Cortex-M0, `riscv32imc`) the lock can never be taken: the backend claims nothing ([`ENABLED`] is `false`).

## Rounding

ESP-NN requantizes with the double rounding of TensorFlow Lite Micro (gemmlowp's `SaturatingRoundingDoublingHighMul` then `RoundingDivideByPOT`; `CONFIG_NN_SKIP_NUDGE` is never set). It has no single-rounding mode: with the `single-rounding` feature of `nanoglia-core`, the backend claims nothing.

The crate depends on [`nanoglia-esp-nn-sys`](https://crates.io/crates/nanoglia-esp-nn-sys), which compiles ESP-NN with a C compiler for the target.
