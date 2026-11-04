//! `SOFTMAX` over the last dimension: on `f32` (TFLite reference `Softmax`) and on int16
//! (`reference_ops::SoftmaxInt16`).
//!
//! The int16 kernel ports LiteRT's `SoftmaxPrepare`: its two 513-entry lookup tables, `exp(x)`
//! on `[-10, 0]` and `1 / (1 + x)` on `[0, 1]`, are computed in `f64` by `LUTPopulate<double>`.
//! TensorFlow Lite Micro computes them in `f32` instead, which changes two entries of the
//! `1 / (1 + x)` table; nanoglia follows LiteRT, its reference.

use crate::util::{
    cmax, expect_arity, expect_type, fixed_kernel, in_slice, input, out_slice, output, per_tensor,
    pod_struct, same_shape, shape_mismatch, unsupported, unsupported_type,
};
use nanoglia_core::backend::{EvalCtx, FnKernel, KernelPlan};
use nanoglia_core::pod::{self, Layout, Region};
use nanoglia_core::quant::{self, Multiplier, multiply_by_quantized_multiplier, round_f64};
use nanoglia_core::{DataType, OpDesc, OpOptions, Result};

pod_struct! {
    /// Parameters of a float softmax.
    pub(crate) struct SoftmaxParams {
        /// Number of rows.
        outer: u32,
        /// Size of the last dimension.
        depth: u32,
        beta: f32,
    }
}

fn softmax_params(op: &OpDesc<'_, '_>) -> Result<SoftmaxParams> {
    let OpOptions::Softmax { beta } = op.options else {
        return Err(unsupported(op));
    };
    expect_arity(op, 1, 1)?;
    let (input, output) = (input(op, 0)?, output(op, 0)?);
    expect_type(input, DataType::F32)?;
    expect_type(output, DataType::F32)?;
    let len = same_shape(input, output)?;
    let depth = match input.dims.as_slice().last() {
        Some(&d) if d > 0 => d as usize,
        _ => return Err(shape_mismatch(input)),
    };
    Ok(SoftmaxParams {
        outer: (len / depth) as u32,
        depth: depth as u32,
        beta,
    })
}

fn softmax_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: SoftmaxParams = *pod::read(ctx.persist)?;
    let (outer, depth) = (p.outer as usize, p.depth as usize);
    let input = in_slice::<f32>(ctx.inputs, 0, outer * depth)?;
    let output = out_slice::<f32>(ctx.outputs, 0, outer * depth)?;
    for (x, y) in input
        .chunks_exact(depth)
        .zip(output.chunks_exact_mut(depth))
    {
        let max = x.iter().fold(f32::MIN, |m, &v| cmax(m, v));
        let mut sum = 0.0f32;
        for (o, &v) in y.iter_mut().zip(x) {
            let e = crate::math::expf((v - max) * p.beta);
            *o = e;
            sum += e;
        }
        y.iter_mut().for_each(|o| *o /= sum);
    }
    Ok(())
}

fixed_kernel!(
    /// `SOFTMAX` on `f32`.
    SOFTMAX_F32,
    "rust/softmax_f32",
    softmax_params,
    softmax_eval
);

/// Entries of an int16 lookup table: 512 intervals and the end of the last one.
pub(crate) const LUT_SIZE: usize = 513;

pod_struct! {
    /// Parameters of an int16 softmax, followed by its two lookup tables.
    pub(crate) struct SoftmaxI16Params {
        /// Number of rows.
        outer: u32,
        /// Size of the last dimension.
        depth: u32,
        /// Rescaling of the differences to the row maximum, so that `[-65535, 0]` covers
        /// `[-10, 0]`.
        input: Multiplier,
    }
}

/// The persistent buffer of an int16 softmax: parameters, `exp` table, `1 / (1 + x)` table.
fn softmax_i16_layout() -> (Region<SoftmaxI16Params>, Region<i16>, Region<i16>, usize) {
    let mut layout = Layout::new();
    let params = layout.push::<SoftmaxI16Params>(1);
    let exp = layout.push::<i16>(LUT_SIZE);
    let one_over_one_plus_x = layout.push::<i16>(LUT_SIZE);
    (params, exp, one_over_one_plus_x, layout.size())
}

fn softmax_i16_params(op: &OpDesc<'_, '_>) -> Result<SoftmaxI16Params> {
    let OpOptions::Softmax { beta } = op.options else {
        return Err(unsupported(op));
    };
    expect_arity(op, 1, 1)?;
    let (input, output) = (input(op, 0)?, output(op, 0)?);
    expect_type(input, DataType::I16)?;
    expect_type(output, DataType::I16)?;
    let len = same_shape(input, output)?;
    let depth = match input.dims.as_slice().last() {
        Some(&d) if d > 0 => d as usize,
        _ => return Err(shape_mismatch(input)),
    };
    let (input_scale, input_zero_point) = per_tensor(input)?;
    let (output_scale, output_zero_point) = per_tensor(output)?;
    if input_zero_point != 0 {
        return Err(unsupported_type(input));
    }
    let expected_scale = 1.0 / 32768.0;
    if output_zero_point != 0 || (output_scale - expected_scale).abs() > 0.001 * expected_scale {
        return Err(unsupported_type(output));
    }
    // `input->params.scale * params->beta` is a `float` product.
    let rescale = f64::from(input_scale * beta) / (10.0 / 65535.0);
    if !rescale.is_finite() {
        return Err(unsupported(op));
    }
    let multiplier = quant::quantize_multiplier(rescale);
    // `MultiplyByQuantizedMultiplier` shifts an `int32_t` left by positive shifts.
    if multiplier.shift > 30 {
        return Err(unsupported(op));
    }
    Ok(SoftmaxI16Params {
        outer: (len / depth) as u32,
        depth: depth as u32,
        input: multiplier,
    })
}

fn softmax_i16_plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    softmax_i16_params(op)?;
    Ok(KernelPlan {
        persistent: softmax_i16_layout().3,
        scratch: 0,
    })
}

fn softmax_i16_prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let params = softmax_i16_params(op)?;
    let (header, exp, one_over_one_plus_x, _) = softmax_i16_layout();
    header.get_mut(persist)?[0] = params;
    let range = f64::from(i16::MAX) - f64::from(i16::MIN);
    lut_populate_i16(
        (10.0 / range, i32::from(i16::MAX)),
        2.0 / range,
        libm::exp,
        exp.get_mut(persist)?,
    );
    lut_populate_i16(
        (1.0 / range, i32::from(i16::MIN)),
        2.0 / range,
        |x| 1.0 / (1.0 + x),
        one_over_one_plus_x.get_mut(persist)?,
    );
    Ok(())
}

/// TFLite `LUTPopulateInt16<double>` with an output zero point of 0: samples `transform` at the
/// 513 interval bounds of the int16 input range and biases each entry by half the error of the
/// linear interpolation at the middle of its interval.
pub(crate) fn lut_populate_i16(
    (input_scale, input_zero_point): (f64, i32),
    output_scale: f64,
    transform: impl Fn(f64) -> f64,
    lut: &mut [i16],
) {
    let (min, max) = (i32::from(i16::MIN), i32::from(i16::MAX));
    let input_min = input_scale * f64::from(min - input_zero_point);
    let input_max = input_scale * f64::from(max - input_zero_point);
    let output_min = output_scale * f64::from(min);
    let output_max = output_scale * f64::from(max);
    let steps = lut.len().saturating_sub(1);
    let step = (input_max - input_min) / steps as f64;
    let half_step = step / 2.0;
    let output_scaling_inv = f64::from(max - min + 1) / (output_max - output_min);
    let clamp = |x: f64| {
        // `std::min(std::max(x, table_min), table_max)`.
        let x = if x < f64::from(min) {
            f64::from(min)
        } else {
            x
        };
        (if f64::from(max) < x {
            f64::from(max)
        } else {
            x
        }) as i16
    };
    for (i, entry) in lut.iter_mut().take(steps).enumerate() {
        let x = input_min + i as f64 * step;
        let val = transform(x);
        let val_midpoint = transform(x + half_step);
        let val_next = transform(input_min + (i + 1) as f64 * step);
        let sample_val = round_f64(val * output_scaling_inv);
        let midpoint_interp_val =
            round_f64((val_next * output_scaling_inv + round_f64(val * output_scaling_inv)) / 2.0);
        let midpoint_val = round_f64(val_midpoint * output_scaling_inv);
        let midpoint_err = midpoint_interp_val - midpoint_val;
        let bias = round_f64(midpoint_err / 2.0);
        *entry = clamp(sample_val - bias);
    }
    if let Some(last) = lut.get_mut(steps) {
        *last = clamp(round_f64(transform(input_max) * output_scaling_inv));
    }
}

/// TFLite `LUTLookup` for int16: linear interpolation between the entries of a 513-entry table.
#[inline]
fn lut_lookup(value: i16, lut: &[i16; LUT_SIZE]) -> i16 {
    let index = (256 + (value >> 7)) as usize;
    let offset = i32::from(value & 0x7f);
    let base = lut[index];
    let slope = lut[index + 1].wrapping_sub(base);
    let delta = (i32::from(slope) * offset + 64) >> 7;
    (i32::from(base) + delta) as i16
}

fn softmax_i16_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let (header, exp, one_over_one_plus_x, _) = softmax_i16_layout();
    let p = header.get(ctx.persist)?[0];
    let exp_lut: &[i16; LUT_SIZE] = exp
        .get(ctx.persist)?
        .try_into()
        .map_err(|_| unsupported_lut())?;
    let one_over_lut: &[i16; LUT_SIZE] = one_over_one_plus_x
        .get(ctx.persist)?
        .try_into()
        .map_err(|_| unsupported_lut())?;
    let (outer, depth) = (p.outer as usize, p.depth as usize);
    if depth == 0 {
        return Ok(());
    }
    let input = in_slice::<i16>(ctx.inputs, 0, outer * depth)?;
    let output = out_slice::<i16>(ctx.outputs, 0, outer * depth)?;
    for (x, y) in input
        .chunks_exact(depth)
        .zip(output.chunks_exact_mut(depth))
    {
        let max = x.iter().copied().fold(i16::MIN, i16::max);
        // exp(x - max) in Q0.15, cached in the output, and their sum in Q16.15.
        let mut sum_of_exps = 0i32;
        for (o, &v) in y.iter_mut().zip(x) {
            let diff = i32::from(v) - i32::from(max);
            let scaled = multiply_by_quantized_multiplier(diff, p.input);
            let symmetric = scaled.saturating_add(32767).clamp(-32768, 32767) as i16;
            *o = lut_lookup(symmetric, exp_lut);
            sum_of_exps = sum_of_exps.wrapping_add(i32::from(*o));
        }
        // 1 / sum_of_exps through the 1 / (1 + x) table, with x in [0, 1] recentered on
        // [-32768, 32767].
        let headroom_plus_one = (sum_of_exps as u32).leading_zeros().clamp(1, 31);
        let shifted_sum =
            (((i64::from(sum_of_exps) << (headroom_plus_one - 1)) + (1 << 13)) >> 14) as i32;
        let symmetric_sum = shifted_sum
            .wrapping_sub((1 << 15) + (1 << 16))
            .clamp(-32768, 32767) as i16;
        let reciprocal = i64::from(lut_lookup(symmetric_sum, one_over_lut));
        let right_shift = 31 - headroom_plus_one;
        let round = 1i64 << right_shift.saturating_sub(1);
        for o in y.iter_mut() {
            let result = ((i64::from(*o) * reciprocal + round) >> right_shift) as i32;
            *o = result.clamp(0, 32767) as i16;
        }
    }
    Ok(())
}

/// The lookup tables of the persistent buffer do not have the expected size.
fn unsupported_lut() -> nanoglia_core::Error {
    nanoglia_core::Error::invalid_argument(nanoglia_core::Detail::None)
}

/// `SOFTMAX` on int16.
pub(crate) static SOFTMAX_I16: FnKernel = FnKernel {
    name: "rust/softmax_i16",
    plan: softmax_i16_plan,
    prepare: softmax_i16_prepare,
    eval: softmax_i16_eval,
};
