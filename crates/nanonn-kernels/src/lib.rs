#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod activation;
mod concat;
mod conv;
mod copy;
mod elementwise;
mod fully_connected;
mod math;
mod pad;
mod pool;
mod quantize;
mod reduce;
mod softmax;
mod util;

#[cfg(test)]
mod tests;

#[cfg(test)]
extern crate std;

use nanonn_core::backend::{Backend, BackendKind, Kernel};
use nanonn_core::{BuiltinOp, OpDesc};

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
    ($($kernel:path),+) => {{
        const KERNELS: &[&Kernel] = &[$(&$kernel),+];
        KERNELS
    }};
}

impl RustKernels {
    /// The kernels that may run `op`, in order of preference.
    fn candidates(op: BuiltinOp) -> &'static [&'static Kernel] {
        match op {
            BuiltinOp::RESHAPE | BuiltinOp::SQUEEZE | BuiltinOp::EXPAND_DIMS => {
                kernels!(copy::RESHAPE)
            }
            BuiltinOp::QUANTIZE => kernels!(quantize::QUANTIZE, quantize::REQUANTIZE),
            BuiltinOp::DEQUANTIZE => kernels!(quantize::DEQUANTIZE),
            BuiltinOp::RELU | BuiltinOp::RELU6 | BuiltinOp::RELU_N1_TO_1 => {
                kernels!(activation::RELU_F32, activation::RELU_Q)
            }
            BuiltinOp::LEAKY_RELU => kernels!(activation::LEAKY_RELU_F32, activation::LEAKY_RELU_Q),
            BuiltinOp::LOGISTIC => kernels!(activation::LOGISTIC_F32, activation::LUT_Q),
            BuiltinOp::TANH => kernels!(activation::TANH_F32, activation::LUT_Q),
            BuiltinOp::HARD_SWISH => kernels!(activation::HARD_SWISH_F32, activation::HARD_SWISH_Q),
            BuiltinOp::MEAN => kernels!(reduce::MEAN),
            BuiltinOp::PAD | BuiltinOp::PADV2 => kernels!(pad::PAD),
            BuiltinOp::CONCATENATION => kernels!(concat::CONCATENATION),
            BuiltinOp::ADD | BuiltinOp::SUB | BuiltinOp::MUL => {
                kernels!(elementwise::BINARY, elementwise::QUANTIZED_BINARY)
            }
            BuiltinOp::MAXIMUM | BuiltinOp::MINIMUM => kernels!(elementwise::BINARY),
            BuiltinOp::CONV_2D => kernels!(conv::CONV_2D_F32),
            BuiltinOp::DEPTHWISE_CONV_2D => kernels!(conv::DEPTHWISE_CONV_2D_F32),
            BuiltinOp::FULLY_CONNECTED => kernels!(fully_connected::FULLY_CONNECTED_F32),
            BuiltinOp::AVERAGE_POOL_2D | BuiltinOp::MAX_POOL_2D => kernels!(pool::POOL_F32),
            BuiltinOp::SOFTMAX => kernels!(softmax::SOFTMAX_F32, softmax::SOFTMAX_I16),
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

    fn select(&self, op: &OpDesc<'_, '_>) -> Option<&'static Kernel> {
        // A kernel supports an operator exactly when it can plan it.
        Self::candidates(op.builtin()?)
            .iter()
            .copied()
            .find(|kernel| (kernel.plan)(op).is_ok())
    }
}
