//! `PRELU` (TensorFlow Lite Micro `prelu.cc` and `prelu_common.cc`, reference
//! `BroadcastPrelu4DSlow`), with `alpha` broadcast to the input's shape, up to rank 6.
//!
//! The quantized kernel requantizes positive inputs with `input_scale / output_scale` and
//! negative ones, multiplied by `alpha`, with `input_scale * alpha_scale / output_scale`: two
//! `QuantizeMultiplier` multipliers computed in `f64`, as TensorFlow Lite Micro does. LiteRT
//! computes both ratios in `f32` before widening them, which can round a multiplier differently.

use crate::util::{
    Broadcast, check_aligned, expect_arity, fixed_kernel, in_slice, input, int_range, out_slice,
    output, per_tensor, pod_struct, shape_mismatch, type_code, unsupported, unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod::{self, Pod};
use nanoglia_core::quant::{Multiplier, multiply_by_quantized_multiplier, quantize_multiplier};
use nanoglia_core::{DataType, OpDesc, Result, TensorDesc};

/// Checks the arity and the shapes: `alpha` broadcasts to the input, which has the output's
/// shape (TFLite requires it).
fn prelu_shapes(op: &OpDesc<'_, '_>) -> Result<Broadcast> {
    expect_arity(op, 2, 1)?;
    let (x, alpha, y) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    if x.dims != y.dims {
        return Err(shape_mismatch(y));
    }
    let broadcast = Broadcast::new(x, alpha, y)?;
    check_aligned(alpha, alpha.dtype.size().unwrap_or(1))?;
    Ok(broadcast)
}

fn prelu_f32_params(op: &OpDesc<'_, '_>) -> Result<Broadcast> {
    for i in 0..2 {
        if input(op, i)?.dtype != DataType::F32 {
            return Err(unsupported_type(input(op, i)?));
        }
    }
    if output(op, 0)?.dtype != DataType::F32 {
        return Err(unsupported_type(output(op, 0)?));
    }
    prelu_shapes(op)
}

fn prelu_f32_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let b: Broadcast = *pod::read(ctx.persist)?;
    let x = in_slice::<f32>(ctx.inputs, 0, b.len1 as usize)?;
    let alpha = in_slice::<f32>(ctx.inputs, 1, b.len2 as usize)?;
    let y = out_slice::<f32>(ctx.outputs, 0, b.len as usize)?;
    b.for_each(|o, i, j| {
        let v = x[i];
        y[o] = if v >= 0.0 { v } else { v * alpha[j] };
    });
    Ok(())
}

fixed_kernel!(
    /// `PRELU` on `f32`, with broadcasting.
    PRELU_F32,
    "rust/prelu_f32",
    prelu_f32_params,
    prelu_f32_eval
);

pod_struct! {
    /// Parameters of a quantized `PRELU` (TFLite `PreluParams`).
    pub(crate) struct PreluParams {
        broadcast: Broadcast,
        /// Type of the input and output.
        dtype: u32,
        /// Negated zero point of the input.
        input_offset: i32,
        /// Negated zero point of `alpha`.
        alpha_offset: i32,
        /// Zero point of the output.
        output_offset: i32,
        /// Requantization of non-negative inputs.
        positive: Multiplier,
        /// Requantization of negative inputs times `alpha`.
        negative: Multiplier,
        min: i32,
        max: i32,
    }
}

/// The scale and zero point of a quantized tensor of `dtype`, with a zero point in the range of
/// the type (0 for int16 activations, which are symmetric).
fn quantization(tensor: &TensorDesc<'_>, symmetric: bool) -> Result<(f32, i32)> {
    let (scale, zero_point) = per_tensor(tensor)?;
    let (min, max) = int_range(tensor.dtype).ok_or_else(|| unsupported_type(tensor))?;
    if !(min..=max).contains(&zero_point) || (symmetric && zero_point != 0) {
        return Err(unsupported_type(tensor));
    }
    Ok((scale, zero_point))
}

fn prelu_q_params(op: &OpDesc<'_, '_>) -> Result<PreluParams> {
    let broadcast = prelu_shapes(op)?;
    let (x, alpha, y) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    // int8 and uint8 take an `alpha` of their type, int16 an int8 one.
    let alpha_type = match x.dtype {
        DataType::I8 | DataType::I16 => DataType::I8,
        DataType::U8 => DataType::U8,
        _ => return Err(unsupported_type(x)),
    };
    if y.dtype != x.dtype {
        return Err(unsupported_type(y));
    }
    if alpha.dtype != alpha_type {
        return Err(unsupported_type(alpha));
    }
    let int16 = x.dtype == DataType::I16;
    let (input_scale, input_zero) = quantization(x, int16)?;
    let (alpha_scale, alpha_zero) = quantization(alpha, false)?;
    let (output_scale, output_zero) = quantization(y, int16)?;
    let (input_scale, alpha_scale, output_scale) = (
        f64::from(input_scale),
        f64::from(alpha_scale),
        f64::from(output_scale),
    );
    let positive = quantize_multiplier(input_scale / output_scale);
    let negative = quantize_multiplier(input_scale * alpha_scale / output_scale);
    // `MultiplyByQuantizedMultiplier` shifts an `int32_t` left by positive shifts.
    if positive.shift > 30 || negative.shift > 30 {
        return Err(unsupported(op));
    }
    let (min, max) = int_range(x.dtype).ok_or_else(|| unsupported_type(x))?;
    Ok(PreluParams {
        broadcast,
        dtype: type_code(x.dtype),
        input_offset: -input_zero,
        alpha_offset: -alpha_zero,
        output_offset: output_zero,
        positive,
        negative,
        min,
        max,
    })
}

/// Quantized `PRELU` with inputs of `T` and `alpha` of `U`; `narrow` truncates a clamped result.
fn prelu_q<T: Pod + Into<i32>, U: Pod + Into<i32>>(
    ctx: &mut EvalCtx<'_, '_>,
    p: &PreluParams,
    narrow: fn(i32) -> T,
) -> Result<()> {
    let b = &p.broadcast;
    let x = in_slice::<T>(ctx.inputs, 0, b.len1 as usize)?;
    let alpha = in_slice::<U>(ctx.inputs, 1, b.len2 as usize)?;
    let y = out_slice::<T>(ctx.outputs, 0, b.len as usize)?;
    b.for_each(|o, i, j| {
        let input = p.input_offset + x[i].into();
        let value = if input >= 0 {
            multiply_by_quantized_multiplier(input, p.positive)
        } else {
            let alpha = p.alpha_offset + alpha[j].into();
            multiply_by_quantized_multiplier(input * alpha, p.negative)
        };
        y[o] = narrow(value.wrapping_add(p.output_offset).clamp(p.min, p.max));
    });
    Ok(())
}

fn prelu_q_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: PreluParams = *pod::read(ctx.persist)?;
    match p.dtype {
        t if t == type_code(DataType::I8) => prelu_q::<i8, i8>(ctx, &p, |v| v as i8),
        t if t == type_code(DataType::U8) => prelu_q::<u8, u8>(ctx, &p, |v| v as u8),
        _ => prelu_q::<i16, i8>(ctx, &p, |v| v as i16),
    }
}

fixed_kernel!(
    /// `PRELU` on int8, uint8 and int16 (with an int8 `alpha`), with broadcasting.
    PRELU_Q,
    "rust/prelu_q",
    prelu_q_params,
    prelu_q_eval
);
