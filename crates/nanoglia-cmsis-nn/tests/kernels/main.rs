//! Bit-exactness of the CMSIS-NN backend against Rust ports of the TFLite reference kernels, on
//! deterministic pseudo-random data.

mod harness;
mod reference;

mod conv;
mod depthwise;
mod elementwise;
mod fully_connected;
mod int16;
mod minmax;
mod pad;
mod pooling;
mod selection;
mod softmax;
mod transpose_conv;
