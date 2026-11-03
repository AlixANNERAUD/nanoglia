# nanonn

**Run neural networks loaded at runtime on microcontrollers, with one `no_std` Rust API and the best kernels of each chip.**

nanonn reads TensorFlow Lite (`.tflite`) models straight from flash or RAM, without copying their weights, and runs them with:

- [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) on Arm Cortex-M (DSP and Helium extensions), and as a portable C fallback everywhere else;
- [ESP-NN](https://github.com/espressif/esp-nn) on ESP32-S3 and ESP32-P4 (planned);
- pure Rust kernels for the operators these libraries lack.

It needs no RTOS, no HAL and no vendor SDK. Every byte it uses comes from the allocator you give it: the Rust global allocator, your own allocator, or a static buffer.

> Status: early development.

## Crates

| Crate | Role |
|---|---|
| [`nanonn-core`](crates/nanonn-core) | Tensor, operator and error types, kernel-backend interface, quantization math, arena allocator |

## License

MIT
