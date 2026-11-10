#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
// The kernels of a module compiled for another of its operators, and the helpers shared by
// several operators, are still compiled when the `op-*` features of the operators using them are
// off: nothing references them and the linker drops them, with the CMSIS-NN functions they call.
#![cfg_attr(not(feature = "all-ops"), allow(dead_code, unused_imports))]

use nanoglia_core::backend::{Backend, BackendKind, FnKernel, Kernel};
use nanoglia_core::{BuiltinOp, OpDesc};

mod common;
#[cfg(feature = "op-conv-2d")]
mod conv;
#[cfg(feature = "op-depthwise-conv-2d")]
mod depthwise;
#[cfg(any(feature = "op-add", feature = "op-mul"))]
mod elementwise;
#[cfg(feature = "op-fully-connected")]
mod fully_connected;
mod layout;
#[cfg(any(feature = "op-maximum", feature = "op-minimum"))]
mod minmax;
#[cfg(any(feature = "op-pad", feature = "op-padv2"))]
mod pad;
#[cfg(any(feature = "op-average-pool-2d", feature = "op-max-pool-2d"))]
mod pooling;
#[cfg(feature = "op-softmax")]
mod softmax;
#[cfg(feature = "op-transpose-conv")]
mod transpose_conv;

/// Whether the int16 (16x8) kernels are offered (feature `int16`). When they are not, the
/// branches that select them are dead code and the linker drops the C kernels.
pub(crate) const INT16: bool = cfg!(feature = "int16");

/// The CMSIS-NN backend. Use [`BACKEND`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CmsisNn;

/// The CMSIS-NN backend, to register with the interpreter.
pub static BACKEND: CmsisNn = CmsisNn;

/// The int8 kernels [`BACKEND`] selects, for backends that run CMSIS-NN in their place (the
/// ESP-NN backend does when another context is using ESP-NN). Compare the address of the kernel
/// [`Backend::select`] returns with these (`core::ptr::addr_eq`) to know which kernel an operator
/// gets.
///
/// Each kernel is there with the `op-*` feature of its operator; `ADD_S8` and `MUL_S8` with
/// either of `op-add` and `op-mul`, the two pooling kernels with either of `op-average-pool-2d`
/// and `op-max-pool-2d` (they share their code). [`BACKEND`] only selects the kernel of an
/// operator whose own feature is on.
pub mod kernels {
    /// `CONV_2D`, int8.
    #[cfg(feature = "op-conv-2d")]
    pub use crate::conv::KERNEL as CONV_2D_S8;
    /// `DEPTHWISE_CONV_2D`, int8.
    #[cfg(feature = "op-depthwise-conv-2d")]
    pub use crate::depthwise::KERNEL as DEPTHWISE_CONV_2D_S8;
    /// `ADD`, int8.
    #[cfg(any(feature = "op-add", feature = "op-mul"))]
    pub use crate::elementwise::ADD as ADD_S8;
    /// `MUL`, int8.
    #[cfg(any(feature = "op-add", feature = "op-mul"))]
    pub use crate::elementwise::MUL as MUL_S8;
    /// `FULLY_CONNECTED`, int8.
    #[cfg(feature = "op-fully-connected")]
    pub use crate::fully_connected::KERNEL as FULLY_CONNECTED_S8;
    /// `AVERAGE_POOL_2D`, int8.
    #[cfg(any(feature = "op-average-pool-2d", feature = "op-max-pool-2d"))]
    pub use crate::pooling::AVERAGE as AVERAGE_POOL_2D_S8;
    /// `MAX_POOL_2D`, int8.
    #[cfg(any(feature = "op-average-pool-2d", feature = "op-max-pool-2d"))]
    pub use crate::pooling::MAX as MAX_POOL_2D_S8;
    /// `SOFTMAX`, int8.
    #[cfg(feature = "op-softmax")]
    pub use crate::softmax::KERNEL as SOFTMAX_S8;
}

impl Backend for CmsisNn {
    fn name(&self) -> &'static str {
        common::NAME
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn select<'s>(&'s self, op: &OpDesc<'_, '_>) -> Option<&'s dyn Kernel> {
        let kernel: &'static FnKernel = select(op)?;
        Some(kernel)
    }
}

impl CmsisNn {
    /// Whether this build of the crate has the kernels of `op`: the backend has kernels for the
    /// operator and its feature ([`BuiltinOp::feature`]) is on. A kernel may still refuse a given
    /// operator, for its types (the `int16` feature), shapes or options: only
    /// [`select`](Backend::select) tells.
    pub fn has_operator(op: BuiltinOp) -> bool {
        selector(op).is_some()
    }
}

/// Picks the kernel of an operator, `None` if no kernel supports it.
type Selector = fn(&OpDesc<'_, '_>) -> Option<&'static FnKernel>;

/// The kernel [`BACKEND`] selects for `op`.
fn select(op: &OpDesc<'_, '_>) -> Option<&'static FnKernel> {
    selector(op.builtin()?)?(op)
}

/// The table of operators: how the kernel of each is picked, behind its `op-*` feature.
fn selector(op: BuiltinOp) -> Option<Selector> {
    match op {
        #[cfg(feature = "op-conv-2d")]
        BuiltinOp::CONV_2D => Some(conv::select),
        #[cfg(feature = "op-depthwise-conv-2d")]
        BuiltinOp::DEPTHWISE_CONV_2D => Some(depthwise::select),
        #[cfg(feature = "op-fully-connected")]
        BuiltinOp::FULLY_CONNECTED => Some(fully_connected::select),
        #[cfg(feature = "op-average-pool-2d")]
        BuiltinOp::AVERAGE_POOL_2D => Some(pooling::select_average),
        #[cfg(feature = "op-max-pool-2d")]
        BuiltinOp::MAX_POOL_2D => Some(pooling::select_max),
        #[cfg(feature = "op-softmax")]
        BuiltinOp::SOFTMAX => Some(|op| softmax::supports(op).then_some(&softmax::KERNEL)),
        #[cfg(feature = "op-add")]
        BuiltinOp::ADD => Some(elementwise::select_add),
        #[cfg(feature = "op-mul")]
        BuiltinOp::MUL => Some(elementwise::select_mul),
        #[cfg(feature = "op-maximum")]
        BuiltinOp::MAXIMUM => Some(|op| minmax::supports(op).then_some(&minmax::MAXIMUM)),
        #[cfg(feature = "op-minimum")]
        BuiltinOp::MINIMUM => Some(|op| minmax::supports(op).then_some(&minmax::MINIMUM)),
        #[cfg(feature = "op-pad")]
        BuiltinOp::PAD => Some(|op| pad::supports(op).then_some(&pad::KERNEL)),
        #[cfg(feature = "op-padv2")]
        BuiltinOp::PADV2 => Some(|op| pad::supports(op).then_some(&pad::KERNEL)),
        #[cfg(feature = "op-transpose-conv")]
        BuiltinOp::TRANSPOSE_CONV => {
            Some(|op| transpose_conv::supports(op).then_some(&transpose_conv::KERNEL))
        }
        _ => None,
    }
}
