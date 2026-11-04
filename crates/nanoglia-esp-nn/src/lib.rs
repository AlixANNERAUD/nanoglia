#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

use nanoglia_core::backend::{Backend, BackendKind, Kernel};
use nanoglia_core::{BuiltinOp, OpDesc, quant};

mod common;
mod conv;
mod depthwise;
mod elementwise;
mod fallback;
mod fully_connected;
mod layout;
mod pooling;
mod softmax;

pub use nanoglia_esp_nn_sys::{ScratchLock, try_lock};

/// The ESP-NN backend. Use [`BACKEND`].
#[derive(Clone, Copy, Debug, Default)]
pub struct EspNn;

/// The ESP-NN backend, to register with the interpreter (before the CMSIS-NN one).
pub static BACKEND: EspNn = EspNn;

/// Whether [`BACKEND`] claims any operator: ESP-NN only requantizes with double rounding (feature
/// `single-rounding` of `nanoglia-core` turns the backend off), and the scratch lock needs atomic
/// compare-and-swap (without it every operator would run its CMSIS-NN fallback anyway).
pub const ENABLED: bool = !quant::SINGLE_ROUNDING && nanoglia_esp_nn_sys::LOCK_AVAILABLE;

/// Whether the kernels run ESP-NN's ESP32-S3 optimized code (`xtensa-esp32s3-*` targets) rather
/// than its ANSI C reference code.
pub const OPTIMIZED: bool = nanoglia_esp_nn_sys::OPTIMIZED_ESP32S3;

impl Backend for EspNn {
    fn name(&self) -> &'static str {
        common::NAME
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn select(&self, op: &OpDesc<'_, '_>) -> Option<&'static Kernel> {
        if !ENABLED {
            return None;
        }
        match op.builtin()? {
            BuiltinOp::CONV_2D => fallback::select::<conv::Conv>(op, &conv::KERNEL),
            BuiltinOp::DEPTHWISE_CONV_2D => {
                fallback::select::<depthwise::Depthwise>(op, &depthwise::KERNEL)
            }
            BuiltinOp::FULLY_CONNECTED => {
                fallback::select::<fully_connected::FullyConnected>(op, &fully_connected::KERNEL)
            }
            BuiltinOp::AVERAGE_POOL_2D => {
                fallback::select::<pooling::Pool<false>>(op, &pooling::AVERAGE)
            }
            BuiltinOp::MAX_POOL_2D => fallback::select::<pooling::Pool<true>>(op, &pooling::MAX),
            BuiltinOp::SOFTMAX => fallback::select::<softmax::Softmax>(op, &softmax::KERNEL),
            BuiltinOp::ADD => fallback::select::<elementwise::Binary<false>>(op, &elementwise::ADD),
            BuiltinOp::MUL => fallback::select::<elementwise::Binary<true>>(op, &elementwise::MUL),
            _ => None,
        }
    }
}
