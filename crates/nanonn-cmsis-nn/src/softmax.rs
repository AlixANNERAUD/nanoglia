//! `SOFTMAX` from int8 to int8 over the last dimension, through `arm_softmax_s8`.
//!
//! Port of TFLite Micro's `kernels/cmsis_nn/softmax.cc` and `CalculateSoftmaxParams` (int8 path
//! with `kScaledDiffIntegerBits = 5`): `PreprocessSoftmaxScaling` gives the input multiplier and
//! left shift, `CalculateInputRadius` the smallest difference to the row maximum that still
//! contributes. The output must be quantized with zero point -128 and a scale of 1/256 (LiteRT's
//! tolerance of 0.1%).

use nanonn_cmsis_nn_sys as sys;
use nanonn_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanonn_core::pod::Pod;
use nanonn_core::quant;
use nanonn_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result};

use crate::common;
use crate::layout::{read_params, write_params};

/// The `SOFTMAX` kernel.
pub(crate) static KERNEL: Kernel = Kernel {
    name: "cmsis-nn/softmax_s8",
    plan,
    prepare,
    eval,
};

/// Persistent parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    num_rows: i32,
    row_size: i32,
    multiplier: i32,
    left_shift: i32,
    diff_min: i32,
}

// SAFETY: `#[repr(C)]` struct of `i32` fields only: no padding, any bit pattern is valid.
unsafe impl Pod for Params {}

const SCALED_DIFF_INTEGER_BITS: i32 = 5;

/// TFLite `PreprocessSoftmaxScaling` (double rounding): the Q31 multiplier and left shift of
/// `beta * input_scale * 2^(31 - input_integer_bits)`, capped to `2^31 - 1`.
fn preprocess_softmax_scaling(
    beta: f64,
    input_scale: f64,
    input_integer_bits: i32,
) -> Option<(i32, i32)> {
    let max_real_multiplier = ((1i64 << 31) - 1) as f64;
    let real =
        (beta * input_scale * (1i64 << (31 - input_integer_bits)) as f64).min(max_real_multiplier);
    let m = quant::quantize_multiplier_greater_than_one(real)?;
    Some((m.multiplier, m.shift))
}

/// TFLite `CalculateInputRadius` with 31 total signed bits.
fn calculate_input_radius(input_integer_bits: i32, input_left_shift: i32) -> Option<i32> {
    let max_input_rescaled = ((1i64 << input_integer_bits) - 1) as f64
        * (1i64 << (31 - input_integer_bits)) as f64
        / (1i64 << input_left_shift) as f64;
    // floor() of a positive value.
    let radius = max_input_rescaled as i64;
    i32::try_from(radius).ok()
}

fn analyze(op: &OpDesc<'_, '_>) -> Option<Params> {
    if op.builtin()? != BuiltinOp::SOFTMAX || op.inputs.len() != 1 || op.outputs.len() != 1 {
        return None;
    }
    let OpOptions::Softmax { beta } = op.options else {
        return None;
    };
    let input = op.input(0)?;
    let output = op.output(0)?;
    let (input_scale, _) = common::activation_quant(input, DataType::I8)?;
    let (output_scale, output_zero_point) = common::activation_quant(output, DataType::I8)?;
    if output_zero_point != -128 || (output_scale - 1.0 / 256.0).abs() > 0.001 / 256.0 {
        return None;
    }
    if input.dims != output.dims || input.dims.rank() == 0 || !beta.is_finite() {
        return None;
    }
    let len = common::element_count(&input.dims)?;
    let row_size = input.dims.get(input.dims.rank() - 1)?;
    let (multiplier, left_shift) = preprocess_softmax_scaling(
        f64::from(beta),
        f64::from(input_scale),
        SCALED_DIFF_INTEGER_BITS,
    )?;
    if !(0..=31).contains(&left_shift) {
        return None;
    }
    let diff_min = -calculate_input_radius(SCALED_DIFF_INTEGER_BITS, left_shift)?;
    Some(Params {
        num_rows: len / row_size,
        row_size,
        multiplier,
        left_shift,
        diff_min,
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

fn eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let len = p.num_rows.saturating_mul(p.row_size);
    let input = common::input(ctx.inputs, 0);
    common::check_len(input, len)?;
    let output = ctx.output(0);
    common::check_len(output, len)?;
    // SAFETY: input and output hold `num_rows * row_size` elements.
    unsafe {
        sys::arm_softmax_s8(
            input.as_ptr().cast(),
            p.num_rows,
            p.row_size,
            p.multiplier,
            p.left_shift,
            p.diff_min,
            output.as_mut_ptr().cast(),
        );
    }
    Ok(())
}
