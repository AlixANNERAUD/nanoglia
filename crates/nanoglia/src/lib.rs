#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc as _;

// A bundled backend without any operator runs no model. Before the `op-*` features, turning the
// default features off kept every operator: such a build must now name `all-ops` or its
// operators, and fails here rather than with `UnsupportedOp` on the device. The test
// `op_features` of `nanoglia-tools` checks this list against the manifest.
#[cfg(all(
    any(feature = "cmsis-nn", feature = "esp-nn", feature = "kernels"),
    not(any(
        feature = "op-abs",
        feature = "op-add",
        feature = "op-arg-max",
        feature = "op-arg-min",
        feature = "op-average-pool-2d",
        feature = "op-batch-to-space-nd",
        feature = "op-cast",
        feature = "op-concatenation",
        feature = "op-conv-2d",
        feature = "op-depthwise-conv-2d",
        feature = "op-depth-to-space",
        feature = "op-dequantize",
        feature = "op-div",
        feature = "op-equal",
        feature = "op-exp",
        feature = "op-expand-dims",
        feature = "op-fully-connected",
        feature = "op-gather",
        feature = "op-greater",
        feature = "op-greater-equal",
        feature = "op-hard-swish",
        feature = "op-leaky-relu",
        feature = "op-less",
        feature = "op-less-equal",
        feature = "op-log",
        feature = "op-logistic",
        feature = "op-maximum",
        feature = "op-max-pool-2d",
        feature = "op-mean",
        feature = "op-minimum",
        feature = "op-mul",
        feature = "op-neg",
        feature = "op-not-equal",
        feature = "op-pack",
        feature = "op-pad",
        feature = "op-padv2",
        feature = "op-prelu",
        feature = "op-quantize",
        feature = "op-reduce-max",
        feature = "op-reduce-min",
        feature = "op-relu",
        feature = "op-relu6",
        feature = "op-relu-n1-to-1",
        feature = "op-reshape",
        feature = "op-resize-bilinear",
        feature = "op-resize-nearest-neighbor",
        feature = "op-rsqrt",
        feature = "op-select-v2",
        feature = "op-slice",
        feature = "op-softmax",
        feature = "op-space-to-batch-nd",
        feature = "op-space-to-depth",
        feature = "op-split",
        feature = "op-split-v",
        feature = "op-sqrt",
        feature = "op-squared-difference",
        feature = "op-squeeze",
        feature = "op-strided-slice",
        feature = "op-sub",
        feature = "op-sum",
        feature = "op-tanh",
        feature = "op-transpose",
        feature = "op-transpose-conv",
        feature = "op-unpack",
    ))
))]
compile_error!(
    "nanoglia: a backend is enabled without any operator. Add the feature `all-ops`, or the \
     `op-*` feature of each operator of your models (for example `op-conv-2d`), next to \
     `cmsis-nn` / `kernels`: see documentation/how-to-guides/reduce-code-size.md"
);

mod check;
mod context;
mod graph;
mod ownership;
mod planner;
mod registry;
mod tensor;

pub use check::*;
pub use context::*;
pub use graph::*;
pub use ownership::*;
pub use registry::*;
pub use tensor::*;

pub use nanoglia_core as core;
#[cfg(feature = "alloc")]
pub use nanoglia_core::allocator_api2::alloc::Global;
pub use nanoglia_core::{
    Activation, AllocError, Allocator, Arena, BuiltinOp, DataType, Detail, Dims, Error, ErrorCode,
    OpCode, Quantization, Result, TensorDesc,
};

/// How the model bytes are encoded, as in wasi-nn's `graph-encoding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Encoding {
    /// A TensorFlow Lite / LiteRT flatbuffer (`.tflite`).
    TensorFlowLite,
    /// Detect the encoding from the bytes.
    Autodetect,
}

/// Where the graph should run, as in wasi-nn's `execution-target`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
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
