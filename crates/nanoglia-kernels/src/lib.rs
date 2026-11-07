#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
// Kernels left out by the `float` and `int16` features are still compiled, only not offered
// by the backend, so the linker drops them.
#![cfg_attr(not(all(feature = "float", feature = "int16")), allow(dead_code))]

mod activation;
mod arg_min_max;
mod concat;
mod conv;
mod copy;
mod depth_space;
mod elementwise;
mod fully_connected;
mod gather;
mod math;
mod pad;
mod pool;
mod prelu;
mod quantize;
mod reduce;
mod resize;
mod slice;
mod softmax;
mod space_batch;
mod split;
mod strided;
mod transpose;
mod transpose_conv;
mod unary;
mod util;

#[cfg(test)]
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
    ($($(#[$attr:meta])* $kernel:path),+ $(,)?) => {{
        const KERNELS: &[&FnKernel] = &[$($(#[$attr])* &$kernel),+];
        KERNELS
    }};
}

impl RustKernels {
    /// The kernels that may run `op`, in order of preference.
    fn candidates(op: BuiltinOp) -> &'static [&'static FnKernel] {
        match op {
            BuiltinOp::RESHAPE | BuiltinOp::SQUEEZE | BuiltinOp::EXPAND_DIMS => {
                kernels!(copy::RESHAPE)
            }
            BuiltinOp::QUANTIZE => kernels!(quantize::QUANTIZE, quantize::REQUANTIZE),
            BuiltinOp::DEQUANTIZE => kernels!(quantize::DEQUANTIZE),
            BuiltinOp::RELU | BuiltinOp::RELU6 | BuiltinOp::RELU_N1_TO_1 => {
                kernels!(
                    #[cfg(feature = "float")]
                    activation::RELU_F32,
                    activation::RELU_Q
                )
            }
            BuiltinOp::LEAKY_RELU => kernels!(
                #[cfg(feature = "float")]
                activation::LEAKY_RELU_F32,
                activation::LEAKY_RELU_Q
            ),
            BuiltinOp::LOGISTIC => kernels!(
                #[cfg(feature = "float")]
                activation::LOGISTIC_F32,
                activation::LUT_Q,
                #[cfg(feature = "int16")]
                activation::LOGISTIC_I16
            ),
            BuiltinOp::TANH => kernels!(
                #[cfg(feature = "float")]
                activation::TANH_F32,
                activation::LUT_Q,
                #[cfg(feature = "int16")]
                activation::TANH_I16
            ),
            BuiltinOp::HARD_SWISH => kernels!(
                #[cfg(feature = "float")]
                activation::HARD_SWISH_F32,
                activation::HARD_SWISH_Q
            ),
            BuiltinOp::MEAN => kernels!(reduce::MEAN),
            BuiltinOp::SUM => kernels!(reduce::SUM),
            BuiltinOp::REDUCE_MAX | BuiltinOp::REDUCE_MIN => kernels!(reduce::REDUCE_MIN_MAX),
            BuiltinOp::PACK => kernels!(concat::PACK),
            BuiltinOp::UNPACK => kernels!(split::UNPACK),
            BuiltinOp::SPACE_TO_DEPTH => kernels!(depth_space::SPACE_TO_DEPTH),
            BuiltinOp::DEPTH_TO_SPACE => kernels!(depth_space::DEPTH_TO_SPACE),
            BuiltinOp::SPACE_TO_BATCH_ND => kernels!(space_batch::SPACE_TO_BATCH_ND),
            BuiltinOp::BATCH_TO_SPACE_ND => kernels!(space_batch::BATCH_TO_SPACE_ND),
            BuiltinOp::PRELU => kernels!(
                #[cfg(feature = "float")]
                prelu::PRELU_F32,
                prelu::PRELU_Q
            ),
            BuiltinOp::ABS
            | BuiltinOp::NEG
            | BuiltinOp::SQRT
            | BuiltinOp::RSQRT
            | BuiltinOp::EXP
            | BuiltinOp::LOG => kernels!(
                #[cfg(feature = "float")]
                unary::UNARY_F32,
                unary::UNARY_INT,
                unary::UNARY_Q
            ),
            BuiltinOp::PAD | BuiltinOp::PADV2 => kernels!(pad::PAD),
            BuiltinOp::CONCATENATION => kernels!(concat::CONCATENATION),
            BuiltinOp::TRANSPOSE => kernels!(transpose::TRANSPOSE),
            BuiltinOp::SLICE => kernels!(slice::SLICE),
            BuiltinOp::STRIDED_SLICE => kernels!(slice::STRIDED_SLICE),
            BuiltinOp::GATHER => kernels!(gather::GATHER),
            BuiltinOp::SPLIT | BuiltinOp::SPLIT_V => kernels!(split::SPLIT),
            BuiltinOp::ARG_MAX | BuiltinOp::ARG_MIN => kernels!(arg_min_max::ARG_MIN_MAX),
            BuiltinOp::RESIZE_NEAREST_NEIGHBOR => kernels!(resize::RESIZE_NEAREST_NEIGHBOR),
            BuiltinOp::RESIZE_BILINEAR => kernels!(
                #[cfg(feature = "float")]
                resize::RESIZE_BILINEAR_F32,
                resize::RESIZE_BILINEAR_Q
            ),
            BuiltinOp::ADD | BuiltinOp::SUB | BuiltinOp::MUL => {
                kernels!(elementwise::BINARY, elementwise::QUANTIZED_BINARY)
            }
            BuiltinOp::MAXIMUM | BuiltinOp::MINIMUM => kernels!(elementwise::BINARY),
            BuiltinOp::CONV_2D => kernels!(
                #[cfg(feature = "float")]
                conv::CONV_2D_F32
            ),
            BuiltinOp::TRANSPOSE_CONV => kernels!(
                #[cfg(feature = "float")]
                transpose_conv::TRANSPOSE_CONV_F32,
                transpose_conv::TRANSPOSE_CONV_S8
            ),
            BuiltinOp::DEPTHWISE_CONV_2D => kernels!(
                #[cfg(feature = "float")]
                conv::DEPTHWISE_CONV_2D_F32
            ),
            BuiltinOp::FULLY_CONNECTED => kernels!(
                #[cfg(feature = "float")]
                fully_connected::FULLY_CONNECTED_F32,
                #[cfg(feature = "int16")]
                fully_connected::FULLY_CONNECTED_16X8
            ),
            BuiltinOp::AVERAGE_POOL_2D | BuiltinOp::MAX_POOL_2D => kernels!(
                #[cfg(feature = "float")]
                pool::POOL_F32
            ),
            BuiltinOp::SOFTMAX => kernels!(
                #[cfg(feature = "float")]
                softmax::SOFTMAX_F32,
                #[cfg(feature = "int16")]
                softmax::SOFTMAX_I16
            ),
            _ => &[],
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
        let kernel: &'static FnKernel = Self::candidates(op.builtin()?)
            .iter()
            .copied()
            .find(|kernel| (kernel.plan)(op).is_ok())?;
        Some(kernel)
    }
}
