//! `ADD` and `MUL` of two int8 tensors of the same shape, through `arm_elementwise_add_s8` and
//! `arm_elementwise_mul_s8`. Broadcasting is left to other backends.
//!
//! Port of TFLite Micro's `kernels/cmsis_nn/{add,mul}.cc` with `CalculateOpDataAdd` (8-bit
//! general path, `left_shift = 20`) and `CalculateOpDataMul`.

use nanonn_cmsis_nn_sys as sys;
use nanonn_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanonn_core::pod::Pod;
use nanonn_core::quant;
use nanonn_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result};

use crate::common;
use crate::layout::{read_params, write_params};

/// The `ADD` kernel.
pub(crate) static ADD: Kernel = Kernel {
    name: "cmsis-nn/add_s8",
    plan,
    prepare,
    eval: eval_add,
};

/// The `MUL` kernel.
pub(crate) static MUL: Kernel = Kernel {
    name: "cmsis-nn/mul_s8",
    plan,
    prepare,
    eval: eval_mul,
};

/// Persistent parameters (the input multipliers are only used by `ADD`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Params {
    len: i32,
    input1_offset: i32,
    input1_multiplier: i32,
    input1_shift: i32,
    input2_offset: i32,
    input2_multiplier: i32,
    input2_shift: i32,
    left_shift: i32,
    output_offset: i32,
    output_multiplier: i32,
    output_shift: i32,
    activation_min: i32,
    activation_max: i32,
}

// SAFETY: `#[repr(C)]` struct of `i32` fields only: no padding, any bit pattern is valid.
unsafe impl Pod for Params {}

/// The left shift applied to 8-bit inputs before rescaling them (TFLite `ADD`).
const ADD_LEFT_SHIFT: i32 = 20;

fn analyze(op: &OpDesc<'_, '_>) -> Option<Params> {
    let add = match op.builtin()? {
        BuiltinOp::ADD => true,
        BuiltinOp::MUL => false,
        _ => return None,
    };
    let OpOptions::Elementwise { activation } = op.options else {
        return None;
    };
    if op.inputs.len() != 2 || op.outputs.len() != 1 {
        return None;
    }
    let input1 = op.input(0)?;
    let input2 = op.input(1)?;
    let output = op.output(0)?;
    let (scale1, zero_point1) = common::activation_quant(input1, DataType::I8)?;
    let (scale2, zero_point2) = common::activation_quant(input2, DataType::I8)?;
    let (output_scale, output_zero_point) = common::activation_quant(output, DataType::I8)?;
    if input1.dims != input2.dims || input1.dims != output.dims {
        return None;
    }
    let len = common::element_count(&output.dims)?;
    let (activation_min, activation_max) =
        common::activation_range(activation, DataType::I8, output_scale, output_zero_point)?;

    let mut p = Params {
        len,
        input1_offset: -zero_point1,
        input2_offset: -zero_point2,
        output_offset: output_zero_point,
        activation_min,
        activation_max,
        ..Params::default()
    };
    if add {
        let twice_max_input_scale = 2.0 * f64::from(scale1.max(scale2));
        let real_input1 = f64::from(scale1) / twice_max_input_scale;
        let real_input2 = f64::from(scale2) / twice_max_input_scale;
        let real_output =
            twice_max_input_scale / (f64::from(1i32 << ADD_LEFT_SHIFT) * f64::from(output_scale));
        let m1 = quant::quantize_multiplier_smaller_than_one(real_input1)?;
        let m2 = quant::quantize_multiplier_smaller_than_one(real_input2)?;
        let mo = quant::quantize_multiplier_smaller_than_one(real_output)?;
        p.left_shift = ADD_LEFT_SHIFT;
        (p.input1_multiplier, p.input1_shift) = (m1.multiplier, m1.shift);
        (p.input2_multiplier, p.input2_shift) = (m2.multiplier, m2.shift);
        (p.output_multiplier, p.output_shift) = (mo.multiplier, mo.shift);
    } else {
        let real = f64::from(scale1) * f64::from(scale2) / f64::from(output_scale);
        let m = quant::quantize_multiplier(real);
        (p.output_multiplier, p.output_shift) = (m.multiplier, m.shift);
    }
    Some(p)
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

/// The checked buffers of a binary elementwise operator.
fn buffers(ctx: &mut EvalCtx<'_, '_>, p: &Params) -> Result<(*const i8, *const i8, *mut i8)> {
    let input1 = common::input(ctx.inputs, 0);
    let input2 = common::input(ctx.inputs, 1);
    common::check_len(input1, p.len)?;
    common::check_len(input2, p.len)?;
    let output = ctx.output(0);
    common::check_len(output, p.len)?;
    Ok((
        input1.as_ptr().cast(),
        input2.as_ptr().cast(),
        output.as_mut_ptr().cast(),
    ))
}

fn eval_add(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let (input1, input2, output) = buffers(ctx, &p)?;
    // SAFETY: the three buffers hold `len` elements.
    let status = unsafe {
        sys::arm_elementwise_add_s8(
            input1,
            input2,
            p.input1_offset,
            p.input1_multiplier,
            p.input1_shift,
            p.input2_offset,
            p.input2_multiplier,
            p.input2_shift,
            p.left_shift,
            output,
            p.output_offset,
            p.output_multiplier,
            p.output_shift,
            p.activation_min,
            p.activation_max,
            p.len,
        )
    };
    common::check_status(status)
}

fn eval_mul(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let (input1, input2, output) = buffers(ctx, &p)?;
    // SAFETY: the three buffers hold `len` elements.
    let status = unsafe {
        sys::arm_elementwise_mul_s8(
            input1,
            input2,
            p.input1_offset,
            p.input2_offset,
            output,
            p.output_offset,
            p.output_multiplier,
            p.output_shift,
            p.activation_min,
            p.activation_max,
            p.len,
        )
    };
    common::check_status(status)
}
