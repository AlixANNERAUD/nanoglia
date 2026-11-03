//! A `Backend` looks at each operator of a graph and, if it can run it, hands back a
//! `Kernel`: three plain function pointers (`plan`, `prepare`, `eval`) and no `dyn`, so the
//! interpreter stores the selection in its graph block without allocating.
//!
//! Lifecycle of a kernel, per operator:
//!
//! 1. `plan(op)` reports how many bytes of persistent parameters and of scratch memory it needs.
//! 2. `prepare(op, persist)` writes its parameters (shapes, multipliers...) into `persist`,
//!    a zeroed, [`crate::ALIGN`]-aligned buffer of exactly the planned size. After this, the
//!    kernel never looks at the model again.
//! 3. `eval(ctx)` computes the outputs from the inputs, the parameters and the scratch buffer.
//!    It must not allocate.

use crate::pod::{Pod, cast_slice, cast_slice_mut};
use crate::{OpDesc, Result};

/// Where a backend computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum BackendKind {
    /// Kernels running on the CPU (possibly with SIMD/DSP instructions).
    Cpu,
    /// A neural accelerator (NPU), used through delegate operators.
    Npu,
}

/// A source of kernels.
pub trait Backend: Sync {
    /// Short name, used in error details and diagnostics.
    fn name(&self) -> &'static str;

    /// Where the backend computes.
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    /// Returns a kernel able to run `op`, or `None` if this backend does not support it
    /// (operator, types, shapes or options).
    fn select(&self, op: &OpDesc<'_, '_>) -> Option<&'static Kernel>;
}

/// Memory needs of a prepared kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelPlan {
    /// Bytes of persistent parameters, written once by `prepare`.
    pub persistent: usize,
    /// Bytes of scratch memory needed during `eval`; the scratch buffer is shared by all
    /// operators and its contents are undefined on entry.
    pub scratch: usize,
}

/// Plans the memory of a kernel for an operator.
pub type PlanFn = fn(&OpDesc<'_, '_>) -> Result<KernelPlan>;
/// Writes the persistent parameters of a kernel for an operator.
pub type PrepareFn = fn(&OpDesc<'_, '_>, &mut [u8]) -> Result<()>;
/// Runs a prepared kernel.
pub type EvalFn = fn(&mut EvalCtx<'_, '_>) -> Result<()>;

/// A kernel: the functions that prepare and run one operator.
#[derive(Debug)]
pub struct Kernel {
    /// Name, for diagnostics (for example `"cmsis-nn/conv2d_s8"`).
    pub name: &'static str,
    /// See [`PlanFn`].
    pub plan: PlanFn,
    /// See [`PrepareFn`].
    pub prepare: PrepareFn,
    /// See [`EvalFn`].
    pub eval: EvalFn,
}

/// The buffers an operator runs on.
///
/// Inputs and outputs are in operator order. An omitted optional input is an empty slice.
/// Constant inputs point into the model; the others point into the context's arena. Every
/// buffer is aligned to [`crate::ALIGN`] except constants, whose alignment the interpreter has
/// checked for the types the kernel declared in `plan`... in practice the TFLite converter aligns
/// them to 16 bytes and the interpreter rejects misaligned models.
pub struct EvalCtx<'c, 'd> {
    /// Input tensors.
    pub inputs: &'c [&'d [u8]],
    /// Output tensors.
    pub outputs: &'c mut [&'d mut [u8]],
    /// Persistent parameters written by `prepare`.
    pub persist: &'d [u8],
    /// Shared scratch memory of the planned size.
    pub scratch: &'d mut [u8],
}

impl EvalCtx<'_, '_> {
    /// Input `i` as bytes (empty if omitted or out of range).
    pub fn input(&self, i: usize) -> &[u8] {
        self.inputs.get(i).copied().unwrap_or(&[])
    }

    /// Input `i` as a typed slice.
    pub fn input_as<T: Pod>(&self, i: usize) -> Result<&[T]> {
        cast_slice(self.input(i))
    }

    /// Output `i` as bytes (empty if out of range).
    pub fn output(&mut self, i: usize) -> &mut [u8] {
        match self.outputs.get_mut(i) {
            Some(out) => out,
            None => &mut [],
        }
    }

    /// Output `i` as a typed slice.
    pub fn output_as<T: Pod>(&mut self, i: usize) -> Result<&mut [T]> {
        cast_slice_mut(self.output(i))
    }
}
