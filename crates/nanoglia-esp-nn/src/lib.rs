#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
// The kernels of a module compiled for another of its operators, and the helpers shared by
// several operators, are still compiled when the `op-*` features of the operators using them are
// off: nothing references them and the linker drops them, with the ESP-NN functions they call.
#![cfg_attr(not(feature = "all-ops"), allow(dead_code, unused_imports))]

use nanoglia_core::backend::{Backend, BackendKind, Kernel};
use nanoglia_core::{BuiltinOp, OpDesc, quant};

mod common;
#[cfg(feature = "op-conv-2d")]
mod conv;
#[cfg(feature = "op-depthwise-conv-2d")]
mod depthwise;
#[cfg(any(feature = "op-add", feature = "op-mul"))]
mod elementwise;
mod fallback;
#[cfg(feature = "op-fully-connected")]
mod fully_connected;
mod layout;
#[cfg(any(feature = "op-average-pool-2d", feature = "op-max-pool-2d"))]
mod pooling;
#[cfg(feature = "op-softmax")]
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

    fn select<'s>(&'s self, op: &OpDesc<'_, '_>) -> Option<&'s dyn Kernel> {
        if !ENABLED {
            return None;
        }
        selector(op.builtin()?)?(op)
    }
}

impl EspNn {
    /// Whether this build of the crate has the kernels of `op`: the backend has kernels for the
    /// operator and its feature ([`BuiltinOp::feature`]) is on. A kernel may still refuse a given
    /// operator, for its shapes or options, and the backend claims nothing when it is not
    /// [`ENABLED`]: only [`select`](Backend::select) tells.
    pub fn has_operator(op: BuiltinOp) -> bool {
        selector(op).is_some()
    }
}

/// Picks the kernel of an operator, `None` if ESP-NN or its CMSIS-NN fallback does not support it.
type Selector = fn(&OpDesc<'_, '_>) -> Option<&'static dyn Kernel>;

/// The table of operators: how the kernel of each is picked, behind its `op-*` feature (which
/// turns on the same feature of `nanoglia-cmsis-nn`, for the fallback kernel).
fn selector(op: BuiltinOp) -> Option<Selector> {
    match op {
        #[cfg(feature = "op-conv-2d")]
        BuiltinOp::CONV_2D => Some(|op| fallback::select::<conv::Conv>(op, &conv::KERNEL)),
        #[cfg(feature = "op-depthwise-conv-2d")]
        BuiltinOp::DEPTHWISE_CONV_2D => {
            Some(|op| fallback::select::<depthwise::Depthwise>(op, &depthwise::KERNEL))
        }
        #[cfg(feature = "op-fully-connected")]
        BuiltinOp::FULLY_CONNECTED => Some(|op| {
            fallback::select::<fully_connected::FullyConnected>(op, &fully_connected::KERNEL)
        }),
        #[cfg(feature = "op-average-pool-2d")]
        BuiltinOp::AVERAGE_POOL_2D => {
            Some(|op| fallback::select::<pooling::Pool<false>>(op, &pooling::AVERAGE))
        }
        #[cfg(feature = "op-max-pool-2d")]
        BuiltinOp::MAX_POOL_2D => {
            Some(|op| fallback::select::<pooling::Pool<true>>(op, &pooling::MAX))
        }
        #[cfg(feature = "op-softmax")]
        BuiltinOp::SOFTMAX => Some(|op| fallback::select::<softmax::Softmax>(op, &softmax::KERNEL)),
        #[cfg(feature = "op-add")]
        BuiltinOp::ADD => {
            Some(|op| fallback::select::<elementwise::Binary<false>>(op, &elementwise::ADD))
        }
        #[cfg(feature = "op-mul")]
        BuiltinOp::MUL => {
            Some(|op| fallback::select::<elementwise::Binary<true>>(op, &elementwise::MUL))
        }
        _ => None,
    }
}
