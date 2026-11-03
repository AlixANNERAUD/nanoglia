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
