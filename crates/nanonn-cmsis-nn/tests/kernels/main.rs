//! Bit-exactness of the CMSIS-NN backend against Rust ports of the TFLite reference kernels, on
//! deterministic pseudo-random data.

mod harness;
mod reference;

mod conv;
mod depthwise;
mod elementwise;
mod fully_connected;
mod pooling;
mod selection;
mod softmax;
