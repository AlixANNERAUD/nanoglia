//! Running an ESP-NN kernel, or its CMSIS-NN counterpart while another context uses ESP-NN.
//!
//! ESP-NN keeps its scratch buffers in global variables, so two contexts must not run it at the
//! same time. Each evaluation takes the ESP-NN lock ([`nanoglia_esp_nn_sys::try_lock`], an
//! atomic flag, never blocking): with it, the kernel sets the scratch pointer and calls ESP-NN;
//! without it (another thread, task, interrupt handler or core is inside ESP-NN), the operator
//! runs the CMSIS-NN kernel instead. Both compute the TFLite result, bit for bit.
//!
//! An operator is claimed only if CMSIS-NN supports it too, and both kernels are prepared: the
//! persistent buffer holds a [`Header`], the ESP-NN parameters, then the CMSIS-NN ones; the
//! scratch buffer is as large as the larger of the two needs.

use nanoglia_core::backend::Backend;
use nanoglia_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanoglia_core::pod::{Pod, read, write};
use nanoglia_core::{ALIGN, Detail, Error, ErrorCode, OpDesc, Result};

use crate::common;

/// An operator ESP-NN runs.
pub(crate) trait EspOp {
    /// Kernel name, for diagnostics.
    const NAME: &'static str;

    /// The CMSIS-NN kernel run when ESP-NN is busy.
    fn fallback() -> &'static Kernel;

    /// The persistent and scratch bytes of the ESP-NN kernel, `None` if ESP-NN cannot run `op`
    /// exactly like TFLite.
    fn plan(op: &OpDesc<'_, '_>) -> Option<KernelPlan>;

    /// Writes the ESP-NN parameters in `persist` (zeroed, of the planned size).
    fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()>;

    /// Runs ESP-NN, the lock held; `ctx.persist` holds the ESP-NN parameters only.
    fn run(ctx: &mut EvalCtx<'_, '_>) -> Result<()>;
}

/// The kernel of an [`EspOp`].
pub(crate) const fn kernel<T: EspOp>() -> Kernel {
    Kernel {
        name: T::NAME,
        plan: plan::<T>,
        prepare: prepare::<T>,
        eval: eval::<T>,
    }
}

/// Whether the backend claims `op` with `T`: ESP-NN supports it, and so does the CMSIS-NN kernel
/// it falls back to.
pub(crate) fn select<T: EspOp>(
    op: &OpDesc<'_, '_>,
    kernel: &'static Kernel,
) -> Option<&'static Kernel> {
    T::plan(op)?;
    let fallback = nanoglia_cmsis_nn::BACKEND.select(op)?;
    core::ptr::eq(fallback, T::fallback()).then_some(kernel)
}

/// Where the two parameter sets are in the persistent buffer. Written at its start.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Header {
    /// Bytes of ESP-NN parameters, from offset [`ALIGN`].
    esp_len: u32,
    /// Offset of the CMSIS-NN parameters, a multiple of [`ALIGN`].
    fallback_offset: u32,
    /// Bytes of CMSIS-NN parameters.
    fallback_len: u32,
    _reserved: u32,
}

// SAFETY: `#[repr(C)]` struct of `u32` fields only: no padding, any bit pattern is valid.
unsafe impl Pod for Header {}

const _: () = assert!(core::mem::size_of::<Header>() <= ALIGN);

impl Header {
    fn new(esp_len: usize, fallback_len: usize) -> Result<Self> {
        let too_large = || Error::new(ErrorCode::TooLarge, Detail::None);
        let fallback_offset = ALIGN
            .checked_add(esp_len)
            .and_then(|end| end.checked_next_multiple_of(ALIGN))
            .ok_or_else(too_large)?;
        fallback_offset
            .checked_add(fallback_len)
            .filter(|&end| u32::try_from(end).is_ok())
            .ok_or_else(too_large)?;
        Ok(Self {
            esp_len: esp_len as u32,
            fallback_offset: fallback_offset as u32,
            fallback_len: fallback_len as u32,
            _reserved: 0,
        })
    }

    fn total(&self) -> usize {
        self.fallback_offset as usize + self.fallback_len as usize
    }

    fn esp_range(&self) -> core::ops::Range<usize> {
        ALIGN..ALIGN + self.esp_len as usize
    }

    fn fallback_range(&self) -> core::ops::Range<usize> {
        self.fallback_offset as usize..self.total()
    }
}

/// The plans of both kernels.
fn plans<T: EspOp>(op: &OpDesc<'_, '_>) -> Result<(KernelPlan, KernelPlan)> {
    let esp = T::plan(op).ok_or_else(|| common::unsupported(op))?;
    let fallback = (T::fallback().plan)(op)?;
    Ok((esp, fallback))
}

fn plan<T: EspOp>(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    let (esp, fallback) = plans::<T>(op)?;
    Ok(KernelPlan {
        persistent: Header::new(esp.persistent, fallback.persistent)?.total(),
        scratch: esp.scratch.max(fallback.scratch),
    })
}

fn prepare<T: EspOp>(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let (esp, fallback) = plans::<T>(op)?;
    let header = Header::new(esp.persistent, fallback.persistent)?;
    write(persist, header)?;
    T::prepare(op, slice_mut(persist, header.esp_range())?)?;
    (T::fallback().prepare)(op, slice_mut(persist, header.fallback_range())?)
}

fn eval<T: EspOp>(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let persist = ctx.persist;
    let header: Header = *read(persist)?;
    let esp = slice(persist, header.esp_range())?;
    let fallback = slice(persist, header.fallback_range())?;
    let result = match nanoglia_esp_nn_sys::try_lock() {
        Some(_lock) => {
            ctx.persist = esp;
            T::run(ctx)
        }
        None => {
            ctx.persist = fallback;
            (T::fallback().eval)(ctx)
        }
    };
    ctx.persist = persist;
    result
}

fn slice(bytes: &[u8], range: core::ops::Range<usize>) -> Result<&[u8]> {
    bytes.get(range).ok_or_else(crate::layout::inconsistent)
}

fn slice_mut(bytes: &mut [u8], range: core::ops::Range<usize>) -> Result<&mut [u8]> {
    bytes.get_mut(range).ok_or_else(crate::layout::inconsistent)
}
