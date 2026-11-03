#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc as _;

mod context;
mod graph;
mod planner;
mod registry;
mod tensor;

pub use context::*;
pub use graph::*;
pub use registry::*;
pub use tensor::*;

pub use nanoglia_core as core;
#[cfg(feature = "alloc")]
pub use nanoglia_core::allocator_api2::alloc::Global;
pub use nanoglia_core::{
    Activation, AllocError, Allocator, Arena, DataType, Detail, Dims, Error, ErrorCode, OpCode,
    Quantization, Result, TensorDesc,
};

/// How the model bytes are encoded, as in wasi-nn's `graph-encoding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Encoding {
    /// A TensorFlow Lite / LiteRT flatbuffer (`.tflite`).
    TensorFlowLite,
    /// Detect the encoding from the bytes.
    Autodetect,
}

/// Where the graph should run, as in wasi-nn's `execution-target`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Target {
    /// CPU kernels only (with the SIMD/DSP extensions of the chip); delegates are ignored.
    Cpu,
    /// A neural accelerator: loading fails unless at least one operator runs on it.
    Npu,
    /// The best kernels available for each operator.
    #[default]
    Auto,
}
