#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

use nanoglia_core::backend::{Backend, BackendKind, Kernel};
use nanoglia_core::{BuiltinOp, OpDesc};

mod common;
mod conv;
mod depthwise;
mod elementwise;
mod fully_connected;
mod layout;
mod minmax;
mod pad;
mod pooling;
mod softmax;

/// Whether the int16 (16x8) kernels are offered (feature `int16`). When they are not, the
/// branches that select them are dead code and the linker drops the C kernels.
pub(crate) const INT16: bool = cfg!(feature = "int16");

/// The CMSIS-NN backend. Use [`BACKEND`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CmsisNn;

/// The CMSIS-NN backend, to register with the interpreter.
pub static BACKEND: CmsisNn = CmsisNn;

/// The int8 kernels [`BACKEND`] selects, for backends that run CMSIS-NN in their place (the
/// ESP-NN backend does when another context is using ESP-NN). Compare the result of
/// [`Backend::select`] with these to know which kernel an operator gets.
pub mod kernels {
    /// `CONV_2D`, int8.
    pub use crate::conv::KERNEL as CONV_2D_S8;
    /// `DEPTHWISE_CONV_2D`, int8.
    pub use crate::depthwise::KERNEL as DEPTHWISE_CONV_2D_S8;
    /// `ADD`, int8.
    pub use crate::elementwise::ADD as ADD_S8;
    /// `MUL`, int8.
    pub use crate::elementwise::MUL as MUL_S8;
    /// `FULLY_CONNECTED`, int8.
    pub use crate::fully_connected::KERNEL as FULLY_CONNECTED_S8;
    /// `AVERAGE_POOL_2D`, int8.
    pub use crate::pooling::AVERAGE as AVERAGE_POOL_2D_S8;
    /// `MAX_POOL_2D`, int8.
    pub use crate::pooling::MAX as MAX_POOL_2D_S8;
    /// `SOFTMAX`, int8.
    pub use crate::softmax::KERNEL as SOFTMAX_S8;
}

impl Backend for CmsisNn {
    fn name(&self) -> &'static str {
        common::NAME
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn select(&self, op: &OpDesc<'_, '_>) -> Option<&'static Kernel> {
        let (kernel, supported): (&'static Kernel, fn(&OpDesc<'_, '_>) -> bool) =
            match op.builtin()? {
                BuiltinOp::CONV_2D => return conv::select(op),
                BuiltinOp::DEPTHWISE_CONV_2D => (&depthwise::KERNEL, depthwise::supports),
                BuiltinOp::FULLY_CONNECTED => return fully_connected::select(op),
                BuiltinOp::AVERAGE_POOL_2D => (&pooling::AVERAGE, pooling::supports),
                BuiltinOp::MAX_POOL_2D => (&pooling::MAX, pooling::supports),
                BuiltinOp::SOFTMAX => (&softmax::KERNEL, softmax::supports),
                BuiltinOp::ADD | BuiltinOp::MUL => return elementwise::select(op),
                BuiltinOp::MAXIMUM => (&minmax::MAXIMUM, minmax::supports),
                BuiltinOp::MINIMUM => (&minmax::MINIMUM, minmax::supports),
                BuiltinOp::PAD | BuiltinOp::PADV2 => (&pad::KERNEL, pad::supports),
                _ => return None,
            };
        supported(op).then_some(kernel)
    }
}
