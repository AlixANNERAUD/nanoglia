#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc as _;

pub use allocator_api2;
pub use allocator_api2::alloc::{AllocError, Allocator};

mod arena;
/// The kernel-backend interface: how a backend claims an operator and runs it.
pub mod backend;
mod block;
mod error;
/// Little-endian views over raw model bytes, readable without any alignment requirement.
pub mod le;
mod op;
/// Plain-old-data casts between byte buffers and typed slices.
pub mod pod;
/// TFLite-exact fixed-point quantization math.
pub mod quant;
mod tensor;

pub use arena::*;
pub use block::*;
pub use error::*;
pub use op::*;
pub use tensor::*;

/// The alignment, in bytes, of every block nanonn allocates and of every tensor in an arena.
///
/// 16 bytes covers every scalar type and the vector loads of CMSIS-NN (Helium) and ESP-NN (PIE).
pub const ALIGN: usize = 16;

/// Rounds `n` up to the next multiple of [`ALIGN`], or `None` on overflow.
#[inline]
pub const fn align_up(n: usize) -> Option<usize> {
    match n.checked_add(ALIGN - 1) {
        Some(v) => Some(v & !(ALIGN - 1)),
        None => None,
    }
}
