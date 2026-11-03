# nanoglia

**Run neural networks loaded at runtime on microcontrollers, with one `no_std` Rust API and the best kernels of each chip.**

nanoglia reads TensorFlow Lite (`.tflite`) models straight from flash or RAM, without copying their weights, and runs them with:

- [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) on Arm Cortex-M (DSP and Helium extensions), and as a portable C fallback everywhere else;
- [ESP-NN](https://github.com/espressif/esp-nn) on ESP32-S3 and ESP32-P4 (planned);
- pure Rust kernels for the operators these libraries lack.

It needs no RTOS, no HAL and no vendor SDK. Every byte it uses comes from the allocator you give it: the Rust global allocator, your own allocator, or a static buffer.

> Status: early development. 123 of the 126 golden cases (per-operator int8, int16 and float32 cases, and the four MLPerf Tiny int8 models) run bit-exact with the TFLite reference kernels; the anomaly-detection model is within one or two units, because LiteRT's reference fully-connected kernel requantizes in floating point (see `crates/nanoglia/tests/golden.rs`).

## Crates

| Crate | Role |
|---|---|
| [`nanoglia`](crates/nanoglia) | The library: load a model, create execution contexts, compute |
| [`nanoglia-core`](crates/nanoglia-core) | Tensor, operator and error types, kernel-backend interface, quantization math, arena allocator |
| [`nanoglia-tflite`](crates/nanoglia-tflite) | Zero-copy, allocation-free, panic-free TFLite model verifier and reader |
| [`nanoglia-cmsis-nn-sys`](crates/nanoglia-cmsis-nn-sys) | Vendored CMSIS-NN v8.0.0 (integer kernels) built with `cc`, raw FFI bindings |
| [`nanoglia-cmsis-nn`](crates/nanoglia-cmsis-nn) | CMSIS-NN kernel backend: int8 convolutions, fully-connected layers, pooling, softmax, add and mul, bit-exact with TFLite |
| [`nanoglia-kernels`](crates/nanoglia-kernels) | Pure-Rust kernels bit-exact with TFLite's reference kernels: shape and quantization operators, int8 gaps of CMSIS-NN, float32 operators |

## License

Apache-2.0, see [`LICENSE`](LICENSE). The vendored CMSIS-NN sources are Apache-2.0 as well.
