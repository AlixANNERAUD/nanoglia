//! `FULLY_CONNECTED` on `f32` (TFLite reference `FullyConnected`) and 16x8
//! (`reference_integer_ops::FullyConnected` and `FullyConnectedPerChannel` with int16
//! activations, int8 weights and an int64 bias).
//!
//! The 16x8 kernel requantizes the 64-bit accumulator in fixed point (64-bit
//! `MultiplyByQuantizedMultiplier`), like TensorFlow Lite Micro and CMSIS-NN. LiteRT's reference
//! kernel requantizes in floating point (`round(acc * scale)`), which differs on rare ties.
//! int32 biases are refused: LiteRT reads the bias according to its `quantized_bias_type`
//! option, which nanoglia does not decode, and as int64 by default.

use crate::math::mul_add;
use crate::util::{
    check_aligned, clamp_f32, elements, expect_type, fixed_kernel, float_activation, in_slice,
    input, out_slice, output, per_tensor, pod_struct, quantized_activation, shape_mismatch,
    unsupported, unsupported_type,
};
use nanoglia_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanoglia_core::pod::{self, Layout, Region};
use nanoglia_core::quant::{self, Multiplier};
use nanoglia_core::{DataType, OpDesc, OpOptions, Result, TensorDesc};

pod_struct! {
    /// Parameters of a float fully-connected layer.
    pub(crate) struct FullyConnectedParams {
        batches: u32,
        /// Input values per batch.
        depth: u32,
        /// Output values per batch.
        units: u32,
        has_bias: u32,
        min: f32,
        max: f32,
    }
}

fn fc_params(op: &OpDesc<'_, '_>) -> Result<FullyConnectedParams> {
    let OpOptions::FullyConnected(o) = op.options else {
        return Err(unsupported(op));
    };
    if !o.default_weights_format || !(2..=3).contains(&op.inputs.len()) || op.outputs.len() != 1 {
        return Err(unsupported(op));
    }
    let (input, weights, output) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    for t in [input, weights, output] {
        expect_type(t, DataType::F32)?;
    }
    check_aligned(weights, 4)?;
    let w = weights.dims.as_slice();
    if w.len() != 2 || w[0] <= 0 || w[1] <= 0 {
        return Err(shape_mismatch(weights));
    }
    let (units, depth) = (w[0] as usize, w[1] as usize);
    let out_len = elements(output)?;
    if output.dims.as_slice().last().copied() != Some(w[0]) {
        return Err(shape_mismatch(output));
    }
    let batches = out_len / units;
    if elements(input)? != batches * depth {
        return Err(shape_mismatch(input));
    }
    let has_bias = match op.input(2) {
        Some(bias) => {
            expect_type(bias, DataType::F32)?;
            check_aligned(bias, 4)?;
            if elements(bias)? != units {
                return Err(shape_mismatch(bias));
            }
            true
        }
        None => false,
    };
    let (min, max) = float_activation(op, o.activation)?;
    Ok(FullyConnectedParams {
        batches: batches as u32,
        depth: depth as u32,
        units: units as u32,
        has_bias: u32::from(has_bias),
        min,
        max,
    })
}

fn fc_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: FullyConnectedParams = *pod::read(ctx.persist)?;
    let (batches, depth, units) = (p.batches as usize, p.depth as usize, p.units as usize);
    let input = in_slice::<f32>(ctx.inputs, 0, batches * depth)?;
    let weights = in_slice::<f32>(ctx.inputs, 1, units * depth)?;
    let bias = if p.has_bias != 0 {
        Some(in_slice::<f32>(ctx.inputs, 2, units)?)
    } else {
        None
    };
    let output = out_slice::<f32>(ctx.outputs, 0, batches * units)?;
    for b in 0..batches {
        let x = &input[b * depth..(b + 1) * depth];
        for c in 0..units {
            let w = &weights[c * depth..(c + 1) * depth];
            let mut total = 0.0f32;
            for (a, b) in x.iter().zip(w) {
                total = mul_add(*a, *b, total);
            }
            let bias = bias.map_or(0.0, |b| b[c]);
            output[b * units + c] = clamp_f32(total + bias, p.min, p.max);
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `FULLY_CONNECTED` on `f32`.
    FULLY_CONNECTED_F32,
    "rust/fully_connected_f32",
    fc_params,
    fc_eval
);

pod_struct! {
    /// Parameters of a 16x8 fully-connected layer, followed by one multiplier per output.
    pub(crate) struct FullyConnected16x8Params {
        batches: u32,
        /// Input values per batch.
        depth: u32,
        /// Output values per batch.
        units: u32,
        has_bias: u32,
        min: i32,
        max: i32,
    }
}

/// The persistent buffer of a 16x8 fully-connected layer with `units` outputs.
fn fc_16x8_layout(units: usize) -> (Region<FullyConnected16x8Params>, Region<Multiplier>, usize) {
    let mut layout = Layout::new();
    let params = layout.push::<FullyConnected16x8Params>(1);
    let multipliers = layout.push::<Multiplier>(units);
    (params, multipliers, layout.size())
}

/// The scale of output channel `c` of `weights`, quantized per tensor or per channel.
fn weight_scale(weights: &TensorDesc<'_>, c: usize) -> f32 {
    let scales = weights.quant.scale;
    scales
        .get(if scales.len() == 1 { 0 } else { c })
        .unwrap_or(0.0)
}

/// Validates a 16x8 `FULLY_CONNECTED`: its parameters and the requantization of each output.
fn fc_16x8_params(
    op: &OpDesc<'_, '_>,
    mut multiplier: impl FnMut(usize, Multiplier),
) -> Result<FullyConnected16x8Params> {
    let OpOptions::FullyConnected(o) = op.options else {
        return Err(unsupported(op));
    };
    if !o.default_weights_format || !(2..=3).contains(&op.inputs.len()) || op.outputs.len() != 1 {
        return Err(unsupported(op));
    }
    let (input, weights, output) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    expect_type(input, DataType::I16)?;
    expect_type(weights, DataType::I8)?;
    expect_type(output, DataType::I16)?;
    let w = weights.dims.as_slice();
    if w.len() != 2 || w[0] <= 0 || w[1] <= 0 {
        return Err(shape_mismatch(weights));
    }
    let (units, depth) = (w[0] as usize, w[1] as usize);
    let out_len = elements(output)?;
    if output.dims.as_slice().last().copied() != Some(w[0]) {
        return Err(shape_mismatch(output));
    }
    let batches = out_len / units;
    if elements(input)? != batches * depth {
        return Err(shape_mismatch(input));
    }
    let has_bias = match op.input(2) {
        Some(bias) => {
            expect_type(bias, DataType::I64)?;
            check_aligned(bias, 8)?;
            if elements(bias)? != units {
                return Err(shape_mismatch(bias));
            }
            true
        }
        None => false,
    };
    let (input_scale, input_zero_point) = per_tensor(input)?;
    let output_quant = per_tensor(output)?;
    if input_zero_point != 0 {
        return Err(unsupported_type(input));
    }
    if output_quant.1 != 0 {
        return Err(unsupported_type(output));
    }
    // Symmetric weights, per tensor or per output channel.
    let q = &weights.quant;
    let scales = q.scale.len();
    if !(scales == 1 || (scales == units && q.axis == 0))
        || !q.zero_point.iter().all(|z| z == 0)
        || !q.scale.iter().all(|s| s.is_finite() && s >= 0.0)
    {
        return Err(unsupported_type(weights));
    }
    for c in 0..units {
        let real =
            quant::conv_effective_scale(input_scale, weight_scale(weights, c), output_quant.0);
        let m = quant::quantize_multiplier(real);
        // The 64-bit `MultiplyByQuantizedMultiplier` shifts right by `15 - shift`.
        if m.shift > 14 {
            return Err(unsupported(op));
        }
        multiplier(c, m);
    }
    let (min, max) = quantized_activation(op, o.activation, output, output_quant)?;
    Ok(FullyConnected16x8Params {
        batches: batches as u32,
        depth: depth as u32,
        units: units as u32,
        has_bias: u32::from(has_bias),
        min,
        max,
    })
}

fn fc_16x8_plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    let p = fc_16x8_params(op, |_, _| {})?;
    Ok(KernelPlan {
        persistent: fc_16x8_layout(p.units as usize).2,
        scratch: 0,
    })
}

fn fc_16x8_prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let units = output(op, 0)?.dims.as_slice().last().copied().unwrap_or(0);
    let (header, multipliers, _) = fc_16x8_layout(units.max(0) as usize);
    let table = multipliers.get_mut(persist)?;
    let mut fits = true;
    let p = fc_16x8_params(op, |c, m| match table.get_mut(c) {
        Some(entry) => *entry = m,
        None => fits = false,
    })?;
    if !fits {
        return Err(shape_mismatch(output(op, 0)?));
    }
    header.get_mut(persist)?[0] = p;
    Ok(())
}

fn fc_16x8_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let header = fc_16x8_layout(0).0;
    let p = header.get(ctx.persist)?[0];
    let (batches, depth, units) = (p.batches as usize, p.depth as usize, p.units as usize);
    if depth == 0 || units == 0 {
        return Ok(());
    }
    let multipliers = fc_16x8_layout(units).1.get(ctx.persist)?;
    let input = in_slice::<i16>(ctx.inputs, 0, batches * depth)?;
    let weights = in_slice::<i8>(ctx.inputs, 1, units * depth)?;
    let bias = if p.has_bias != 0 {
        Some(in_slice::<i64>(ctx.inputs, 2, units)?)
    } else {
        None
    };
    let output = out_slice::<i16>(ctx.outputs, 0, batches * units)?;
    for (x, y) in input
        .chunks_exact(depth)
        .zip(output.chunks_exact_mut(units))
    {
        for (c, (o, w)) in y.iter_mut().zip(weights.chunks_exact(depth)).enumerate() {
            let mut acc: i64 = x
                .iter()
                .zip(w)
                .map(|(&a, &b)| i64::from(a) * i64::from(b))
                .sum();
            if let Some(bias) = bias {
                acc = acc.wrapping_add(bias[c]);
            }
            let scaled = quant::multiply_by_quantized_multiplier_i64(acc, multipliers[c]);
            *o = scaled.max(p.min).min(p.max) as i16;
        }
    }
    Ok(())
}

/// `FULLY_CONNECTED` with int16 activations, int8 weights and an int64 bias.
pub(crate) static FULLY_CONNECTED_16X8: Kernel = Kernel {
    name: "rust/fully_connected_16x8",
    plan: fc_16x8_plan,
    prepare: fc_16x8_prepare,
    eval: fc_16x8_eval,
};
