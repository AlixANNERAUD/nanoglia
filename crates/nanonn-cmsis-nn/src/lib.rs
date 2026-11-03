#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

use nanonn_core::backend::{Backend, BackendKind, Kernel};
use nanonn_core::{BuiltinOp, OpDesc};

mod common;
mod conv;
mod depthwise;
mod elementwise;
mod fully_connected;
mod layout;
mod pooling;
mod softmax;

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
                BuiltinOp::CONV_2D => (&conv::KERNEL, conv::supports),
                BuiltinOp::DEPTHWISE_CONV_2D => (&depthwise::KERNEL, depthwise::supports),
                BuiltinOp::FULLY_CONNECTED => (&fully_connected::KERNEL, fully_connected::supports),
                BuiltinOp::AVERAGE_POOL_2D => (&pooling::AVERAGE, pooling::supports),
                BuiltinOp::MAX_POOL_2D => (&pooling::MAX, pooling::supports),
                BuiltinOp::SOFTMAX => (&softmax::KERNEL, softmax::supports),
                BuiltinOp::ADD => (&elementwise::ADD, elementwise::supports),
                BuiltinOp::MUL => (&elementwise::MUL, elementwise::supports),
                _ => return None,
            };
        supported(op).then_some(kernel)
    }
}
