#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
// Kernels left out by the `float` and `int16` features are still compiled, only not offered
// by the backend, so the linker drops them. So are the helpers shared by several operators
// (`util`, `math`, `strided`) and the kernels of a module compiled for another of its operators,
// when the `op-*` features of the operators using them are off.
#![cfg_attr(
    not(all(feature = "float", feature = "int16", feature = "all-ops")),
    allow(dead_code)
)]
// Without any operator, the macros that declare kernels have no user either.
#![cfg_attr(not(feature = "all-ops"), allow(unused_macros, unused_imports))]

#[cfg(any(
    feature = "op-relu",
    feature = "op-relu6",
    feature = "op-relu-n1-to-1",
    feature = "op-leaky-relu",
    feature = "op-logistic",
    feature = "op-tanh",
    feature = "op-hard-swish"
))]
mod activation;
#[cfg(any(feature = "op-arg-max", feature = "op-arg-min"))]
mod arg_min_max;
#[cfg(feature = "op-cast")]
mod cast;
#[cfg(any(
    feature = "op-equal",
    feature = "op-not-equal",
    feature = "op-less",
    feature = "op-less-equal",
    feature = "op-greater",
    feature = "op-greater-equal",
    feature = "op-select-v2"
))]
mod compare;
#[cfg(any(feature = "op-concatenation", feature = "op-pack"))]
mod concat;
#[cfg(any(feature = "op-conv-2d", feature = "op-depthwise-conv-2d"))]
mod conv;
#[cfg(any(
    feature = "op-reshape",
    feature = "op-squeeze",
    feature = "op-expand-dims"
))]
mod copy;
#[cfg(any(feature = "op-space-to-depth", feature = "op-depth-to-space"))]
mod depth_space;
#[cfg(any(
    feature = "op-add",
    feature = "op-sub",
    feature = "op-mul",
    feature = "op-maximum",
    feature = "op-minimum",
    feature = "op-div",
    feature = "op-squared-difference"
))]
mod elementwise;
#[cfg(feature = "op-fully-connected")]
mod fully_connected;
#[cfg(feature = "op-gather")]
mod gather;
mod math;
#[cfg(any(feature = "op-pad", feature = "op-padv2"))]
mod pad;
#[cfg(any(feature = "op-average-pool-2d", feature = "op-max-pool-2d"))]
mod pool;
#[cfg(feature = "op-prelu")]
mod prelu;
#[cfg(any(feature = "op-quantize", feature = "op-dequantize"))]
mod quantize;
#[cfg(any(
    feature = "op-mean",
    feature = "op-sum",
    feature = "op-reduce-max",
    feature = "op-reduce-min"
))]
mod reduce;
#[cfg(any(feature = "op-resize-nearest-neighbor", feature = "op-resize-bilinear"))]
mod resize;
#[cfg(any(feature = "op-slice", feature = "op-strided-slice"))]
mod slice;
#[cfg(feature = "op-softmax")]
mod softmax;
#[cfg(any(feature = "op-space-to-batch-nd", feature = "op-batch-to-space-nd"))]
mod space_batch;
#[cfg(any(feature = "op-split", feature = "op-split-v", feature = "op-unpack"))]
mod split;
mod strided;
#[cfg(feature = "op-transpose")]
mod transpose;
#[cfg(feature = "op-transpose-conv")]
mod transpose_conv;
#[cfg(any(
    feature = "op-abs",
    feature = "op-neg",
    feature = "op-sqrt",
    feature = "op-rsqrt",
    feature = "op-exp",
    feature = "op-log"
))]
mod unary;
mod util;

// The tests run every operator.
#[cfg(all(test, feature = "all-ops"))]
mod tests;

#[cfg(test)]
extern crate std;

use nanoglia_core::backend::{Backend, BackendKind, FnKernel, Kernel};
use nanoglia_core::{BuiltinOp, OpDesc};

/// The pure-Rust kernel backend (name `"rust"`).
///
/// It claims an operator only when it fully supports its types, shapes and options; the
/// interpreter tries it after the backends with optimized kernels.
#[derive(Clone, Copy, Debug, Default)]
pub struct RustKernels;

/// The pure-Rust kernel backend, ready to register.
pub static BACKEND: RustKernels = RustKernels;

/// A static list of kernels.
macro_rules! kernels {
    ($($(#[$attr:meta])* $kernel:path),* $(,)?) => {{
        const KERNELS: &[&FnKernel] = &[$($(#[$attr])* &$kernel),*];
        KERNELS
    }};
}

/// The table of operators: each operator, behind its `op-*` feature, with the kernels that may
/// run it in order of preference. Operators sharing kernels are listed together, but each one is
/// an arm of its own, compiled only with its own feature.
macro_rules! candidates {
    ($op:expr; $($($feature:literal $name:ident),+ => $kernels:expr;)*) => {
        match $op {
            $($(
                #[cfg(feature = $feature)]
                BuiltinOp::$name => Some($kernels),
            )+)*
            _ => None,
        }
    };
}

impl RustKernels {
    /// Whether this build of the crate has the kernels of `op`: the operator has a feature
    /// ([`BuiltinOp::feature`]) and it is on. A kernel may still refuse a given operator, for its
    /// types (the `float` and `int16` features), shapes or options: only
    /// [`select`](Backend::select) tells.
    pub fn has_operator(op: BuiltinOp) -> bool {
        Self::candidates(op).is_some()
    }

    /// The kernels that may run `op`, in order of preference; `None` without the feature of the
    /// operator.
    #[rustfmt::skip]
    fn candidates(op: BuiltinOp) -> Option<&'static [&'static FnKernel]> {
        candidates! { op;
            "op-reshape" RESHAPE, "op-squeeze" SQUEEZE, "op-expand-dims" EXPAND_DIMS =>
                kernels!(copy::RESHAPE);
            "op-quantize" QUANTIZE => kernels!(quantize::QUANTIZE, quantize::REQUANTIZE);
            "op-dequantize" DEQUANTIZE => kernels!(quantize::DEQUANTIZE);
            "op-relu" RELU, "op-relu6" RELU6, "op-relu-n1-to-1" RELU_N1_TO_1 => kernels!(
                #[cfg(feature = "float")]
                activation::RELU_F32,
                activation::RELU_Q
            );
            "op-leaky-relu" LEAKY_RELU => kernels!(
                #[cfg(feature = "float")]
                activation::LEAKY_RELU_F32,
                activation::LEAKY_RELU_Q
            );
            "op-logistic" LOGISTIC => kernels!(
                #[cfg(feature = "float")]
                activation::LOGISTIC_F32,
                activation::LUT_Q,
                #[cfg(feature = "int16")]
                activation::LOGISTIC_I16
            );
            "op-tanh" TANH => kernels!(
                #[cfg(feature = "float")]
                activation::TANH_F32,
                activation::LUT_Q,
                #[cfg(feature = "int16")]
                activation::TANH_I16
            );
            "op-hard-swish" HARD_SWISH => kernels!(
                #[cfg(feature = "float")]
                activation::HARD_SWISH_F32,
                activation::HARD_SWISH_Q
            );
            "op-mean" MEAN => kernels!(reduce::MEAN);
            "op-sum" SUM => kernels!(reduce::SUM);
            "op-reduce-max" REDUCE_MAX, "op-reduce-min" REDUCE_MIN =>
                kernels!(reduce::REDUCE_MIN_MAX);
            "op-pack" PACK => kernels!(concat::PACK);
            "op-unpack" UNPACK => kernels!(split::UNPACK);
            "op-space-to-depth" SPACE_TO_DEPTH => kernels!(depth_space::SPACE_TO_DEPTH);
            "op-depth-to-space" DEPTH_TO_SPACE => kernels!(depth_space::DEPTH_TO_SPACE);
            "op-space-to-batch-nd" SPACE_TO_BATCH_ND => kernels!(space_batch::SPACE_TO_BATCH_ND);
            "op-batch-to-space-nd" BATCH_TO_SPACE_ND => kernels!(space_batch::BATCH_TO_SPACE_ND);
            "op-prelu" PRELU => kernels!(
                #[cfg(feature = "float")]
                prelu::PRELU_F32,
                prelu::PRELU_Q
            );
            "op-abs" ABS, "op-neg" NEG, "op-sqrt" SQRT, "op-rsqrt" RSQRT, "op-exp" EXP,
            "op-log" LOG => kernels!(
                #[cfg(feature = "float")]
                unary::UNARY_F32,
                unary::UNARY_INT,
                unary::UNARY_Q
            );
            "op-pad" PAD, "op-padv2" PADV2 => kernels!(pad::PAD);
            "op-concatenation" CONCATENATION => kernels!(concat::CONCATENATION);
            "op-transpose" TRANSPOSE => kernels!(transpose::TRANSPOSE);
            "op-slice" SLICE => kernels!(slice::SLICE);
            "op-strided-slice" STRIDED_SLICE => kernels!(slice::STRIDED_SLICE);
            "op-gather" GATHER => kernels!(gather::GATHER);
            "op-split" SPLIT, "op-split-v" SPLIT_V => kernels!(split::SPLIT);
            "op-arg-max" ARG_MAX, "op-arg-min" ARG_MIN => kernels!(arg_min_max::ARG_MIN_MAX);
            "op-resize-nearest-neighbor" RESIZE_NEAREST_NEIGHBOR =>
                kernels!(resize::RESIZE_NEAREST_NEIGHBOR);
            "op-resize-bilinear" RESIZE_BILINEAR => kernels!(
                #[cfg(feature = "float")]
                resize::RESIZE_BILINEAR_F32,
                resize::RESIZE_BILINEAR_Q
            );
            "op-add" ADD, "op-sub" SUB, "op-mul" MUL,
            "op-squared-difference" SQUARED_DIFFERENCE =>
                kernels!(elementwise::BINARY, elementwise::QUANTIZED_BINARY);
            "op-maximum" MAXIMUM, "op-minimum" MINIMUM, "op-div" DIV =>
                kernels!(elementwise::BINARY);
            "op-equal" EQUAL, "op-not-equal" NOT_EQUAL, "op-less" LESS,
            "op-less-equal" LESS_EQUAL, "op-greater" GREATER,
            "op-greater-equal" GREATER_EQUAL => kernels!(compare::COMPARISON);
            "op-select-v2" SELECT_V2 => kernels!(compare::SELECT_V2);
            "op-cast" CAST => kernels!(cast::CAST);
            "op-conv-2d" CONV_2D => kernels!(
                #[cfg(feature = "float")]
                conv::CONV_2D_F32
            );
            "op-transpose-conv" TRANSPOSE_CONV => kernels!(
                #[cfg(feature = "float")]
                transpose_conv::TRANSPOSE_CONV_F32,
                transpose_conv::TRANSPOSE_CONV_S8
            );
            "op-depthwise-conv-2d" DEPTHWISE_CONV_2D => kernels!(
                #[cfg(feature = "float")]
                conv::DEPTHWISE_CONV_2D_F32
            );
            "op-fully-connected" FULLY_CONNECTED => kernels!(
                #[cfg(feature = "float")]
                fully_connected::FULLY_CONNECTED_F32,
                #[cfg(feature = "int16")]
                fully_connected::FULLY_CONNECTED_16X8
            );
            "op-average-pool-2d" AVERAGE_POOL_2D, "op-max-pool-2d" MAX_POOL_2D => kernels!(
                #[cfg(feature = "float")]
                pool::POOL_F32
            );
            "op-softmax" SOFTMAX => kernels!(
                #[cfg(feature = "float")]
                softmax::SOFTMAX_F32,
                #[cfg(feature = "int16")]
                softmax::SOFTMAX_I16
            );
        }
    }
}

impl Backend for RustKernels {
    fn name(&self) -> &'static str {
        "rust"
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn select<'s>(&'s self, op: &OpDesc<'_, '_>) -> Option<&'s dyn Kernel> {
        // A kernel supports an operator exactly when it can plan it.
        let kernel: &'static FnKernel = Self::candidates(op.builtin()?)?
            .iter()
            .copied()
            .find(|kernel| (kernel.plan)(op).is_ok())?;
        Some(kernel)
    }
}
