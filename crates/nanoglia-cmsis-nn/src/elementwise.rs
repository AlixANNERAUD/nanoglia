//! `ADD` and `MUL` of two int8 or int16 tensors of the same shape, through
//! `arm_elementwise_{add,mul}_{s8,s16}`. Broadcasting is left to other backends.
//!
//! Port of TFLite Micro's `kernels/cmsis_nn/{add,mul}.cc` with `CalculateOpDataAdd` (general
//! path, `left_shift = 20` for int8 and `15` for int16) and `CalculateOpDataMul`. int16 tensors
//! are symmetric (zero point 0). TFLite (not TFLite Micro) runs int16 `ADD` through another,
//! power-of-two kernel when all scales are powers of two and the `pot_scale_int16` option is set:
//! such operators are refused.

use nanoglia_cmsis_nn_sys as sys;
use nanoglia_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanoglia_core::pod::{Pod, cast_slice, cast_slice_mut};
use nanoglia_core::quant;
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result};

use crate::common;
use crate::layout::{read_params, write_params};

/// The `ADD` kernel.
pub static ADD: Kernel = Kernel {
    name: "cmsis-nn/add_s8",
    plan,
    prepare,
    eval: eval_add,
};

/// The `MUL` kernel.
pub static MUL: Kernel = Kernel {
    name: "cmsis-nn/mul_s8",
    plan,
    prepare,
    eval: eval_mul,
};

/// The int16 `ADD` kernel.
pub(crate) static ADD_S16: Kernel = Kernel {
    name: "cmsis-nn/add_s16",
    plan,
    prepare,
    eval: eval_add_s16,
};

/// The int16 `MUL` kernel.
pub(crate) static MUL_S16: Kernel = Kernel {
    name: "cmsis-nn/mul_s16",
    plan,
    prepare,
    eval: eval_mul_s16,
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
/// The left shift applied to 16-bit inputs before rescaling them (TFLite `ADD`).
const ADD_LEFT_SHIFT_S16: i32 = 15;

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
    let dtype = output.dtype;
    let (scale1, zero_point1) = common::activation_quant(input1, dtype)?;
    let (scale2, zero_point2) = common::activation_quant(input2, dtype)?;
    let (output_scale, output_zero_point) = common::activation_quant(output, dtype)?;
    if input1.dims != input2.dims || input1.dims != output.dims {
        return None;
    }
    let left_shift = match dtype {
        DataType::I8 => ADD_LEFT_SHIFT,
        // CMSIS-NN ignores the offsets of int16 tensors.
        DataType::I16 if zero_point1 == 0 && zero_point2 == 0 && output_zero_point == 0 => {
            if add
                && [scale1, scale2, output_scale]
                    .into_iter()
                    .all(quant::is_near_power_of_two)
            {
                return None;
            }
            ADD_LEFT_SHIFT_S16
        }
        _ => return None,
    };
    let len = common::element_count(&output.dims)?;
    let (activation_min, activation_max) =
        common::activation_range(activation, dtype, output_scale, output_zero_point)?;

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
            twice_max_input_scale / (f64::from(1i32 << left_shift) * f64::from(output_scale));
        let m1 = quant::quantize_multiplier_smaller_than_one(real_input1)?;
        let m2 = quant::quantize_multiplier_smaller_than_one(real_input2)?;
        let mo = quant::quantize_multiplier_smaller_than_one(real_output)?;
        p.left_shift = left_shift;
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

/// The kernel running `op`, if any.
pub(crate) fn select(op: &OpDesc<'_, '_>) -> Option<&'static Kernel> {
    analyze(op)?;
    let int16 = op.output(0)?.dtype == DataType::I16;
    if int16 && !crate::INT16 {
        return None;
    }
    Some(match (op.builtin()?, int16) {
        (BuiltinOp::ADD, false) => &ADD,
        (BuiltinOp::ADD, true) => &ADD_S16,
        (BuiltinOp::MUL, false) => &MUL,
        _ => &MUL_S16,
    })
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

/// The checked buffers of a binary elementwise operator on `T`.
fn buffers<T: Pod>(ctx: &mut EvalCtx<'_, '_>, p: &Params) -> Result<(*const T, *const T, *mut T)> {
    let bytes = p.len.saturating_mul(core::mem::size_of::<T>() as i32);
    let input1 = common::input(ctx.inputs, 0);
    let input2 = common::input(ctx.inputs, 1);
    common::check_len(input1, bytes)?;
    common::check_len(input2, bytes)?;
    let input1 = cast_slice::<T>(input1)?.as_ptr();
    let input2 = cast_slice::<T>(input2)?.as_ptr();
    let output = ctx.output(0);
    common::check_len(output, bytes)?;
    Ok((input1, input2, cast_slice_mut::<T>(output)?.as_mut_ptr()))
}

fn eval_add(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let (input1, input2, output) = buffers::<i8>(ctx, &p)?;
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
    let (input1, input2, output) = buffers::<i8>(ctx, &p)?;
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

fn eval_add_s16(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let (input1, input2, output) = buffers::<i16>(ctx, &p)?;
    // SAFETY: the three buffers hold `len` aligned elements.
    let status = unsafe {
        sys::arm_elementwise_add_s16(
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

fn eval_mul_s16(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let (input1, input2, output) = buffers::<i16>(ctx, &p)?;
    // SAFETY: the three buffers hold `len` aligned elements.
    let status = unsafe {
        sys::arm_elementwise_mul_s16(
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
