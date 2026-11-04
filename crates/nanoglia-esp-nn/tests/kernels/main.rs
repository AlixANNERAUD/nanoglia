//! Bit-exactness of the ESP-NN backend against Rust ports of the TFLite reference kernels, on
//! deterministic pseudo-random data. Every evaluation runs twice: on ESP-NN, then with the ESP-NN
//! lock held, on the CMSIS-NN fallback; both must give the same bytes.
//!
//! On the host this runs ESP-NN's ANSI C kernels; the ESP32-S3 kernels need the hardware.

mod harness;
mod reference;

mod conv;
mod depthwise;
mod elementwise;
mod fully_connected;
mod pooling;
mod softmax;
