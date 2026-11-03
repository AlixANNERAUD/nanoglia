//! `MAXIMUM` and `MINIMUM` of int8 tensors of rank up to 4, through `arm_maximum_s8` and
//! `arm_minimum_s8`, when both inputs have the output's shape or one of them is a single value.
//!
//! Like LiteRT and TFLite Micro, the raw values are compared without requantization.

use nanonn_cmsis_nn_sys as sys;
use nanonn_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanonn_core::pod::Pod;
use nanonn_core::{BuiltinOp, DataType, Dims, OpDesc, Result};

use crate::common::{self, nhwc};
use crate::layout::{read_params, write_params};

/// The `MAXIMUM` kernel.
pub(crate) static MAXIMUM: Kernel = Kernel {
    name: "cmsis-nn/maximum_s8",
    plan,
    prepare,
    eval: eval_maximum,
};

/// The `MINIMUM` kernel.
pub(crate) static MINIMUM: Kernel = Kernel {
    name: "cmsis-nn/minimum_s8",
    plan,
    prepare,
    eval: eval_minimum,
};

/// Persistent parameters: shapes extended to rank 4 with leading ones.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    input1: sys::cmsis_nn_dims,
    input2: sys::cmsis_nn_dims,
    output: sys::cmsis_nn_dims,
}

// SAFETY: `#[repr(C)]` struct of `i32` fields only: no padding, any bit pattern is valid.
unsafe impl Pod for Params {}

/// TFLite `RuntimeShape::ExtendedShape(4, ...)`.
fn extended(dims: &Dims) -> Option<sys::cmsis_nn_dims> {
    let rank = dims.rank();
    if rank > 4 {
        return None;
    }
    common::element_count(dims)?;
    let mut d = [1i32; 4];
    d[4 - rank..].copy_from_slice(dims.as_slice());
    Some(nhwc(d[0], d[1], d[2], d[3]))
}

fn analyze(op: &OpDesc<'_, '_>) -> Option<Params> {
    if !matches!(op.builtin()?, BuiltinOp::MAXIMUM | BuiltinOp::MINIMUM)
        || op.inputs.len() != 2
        || op.outputs.len() != 1
    {
        return None;
    }
    let input1 = op.input(0)?;
    let input2 = op.input(1)?;
    let output = op.output(0)?;
    if [input1.dtype, input2.dtype, output.dtype] != [DataType::I8; 3] {
        return None;
    }
    let (d1, d2, out) = (
        extended(&input1.dims)?,
        extended(&input2.dims)?,
        extended(&output.dims)?,
    );
    let same = |d: &sys::cmsis_nn_dims| *d == out;
    let scalar = |d: &sys::cmsis_nn_dims| common::dims_len(d) == 1;
    let supported =
        (same(&d1) && same(&d2)) || (same(&d1) && scalar(&d2)) || (scalar(&d1) && same(&d2));
    supported.then_some(Params {
        input1: d1,
        input2: d2,
        output: out,
    })
}

/// Whether this kernel runs `op`.
pub(crate) fn supports(op: &OpDesc<'_, '_>) -> bool {
    analyze(op).is_some()
}

fn plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    analyze(op).ok_or_else(|| common::unsupported(op))?;
    Ok(KernelPlan {
        persistent: core::mem::size_of::<Params>(),
        scratch: 0,
    })
}

fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let p = analyze(op).ok_or_else(|| common::unsupported(op))?;
    write_params(persist, p)
}

type MinMaxFn = unsafe extern "C" fn(
    *const sys::cmsis_nn_context,
    *const i8,
    *const sys::cmsis_nn_dims,
    *const i8,
    *const sys::cmsis_nn_dims,
    *mut i8,
    *const sys::cmsis_nn_dims,
) -> sys::arm_cmsis_nn_status;

fn eval_with(ctx: &mut EvalCtx<'_, '_>, function: MinMaxFn) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let input1 = common::input(ctx.inputs, 0);
    let input2 = common::input(ctx.inputs, 1);
    common::check_len(input1, common::dims_len(&p.input1))?;
    common::check_len(input2, common::dims_len(&p.input2))?;
    let output = ctx.output(0);
    common::check_len(output, common::dims_len(&p.output))?;
    let context = sys::cmsis_nn_context {
        buf: core::ptr::null_mut(),
        size: 0,
    };
    // SAFETY: the buffers hold the elements of their shapes, which are equal or a single value.
    let status = unsafe {
        function(
            &context,
            input1.as_ptr().cast(),
            &p.input1,
            input2.as_ptr().cast(),
            &p.input2,
            output.as_mut_ptr().cast(),
            &p.output,
        )
    };
    common::check_status(status)
}

fn eval_maximum(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    eval_with(ctx, sys::arm_maximum_s8)
}

fn eval_minimum(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    eval_with(ctx, sys::arm_minimum_s8)
}
