//! `FULLY_CONNECTED` on `f32` (TFLite reference `FullyConnected`).

use crate::math::mul_add;
use crate::util::{
    check_aligned, clamp_f32, elements, expect_type, fixed_kernel, float_activation, in_slice,
    input, out_slice, output, pod_struct, shape_mismatch, unsupported,
};
use nanonn_core::backend::EvalCtx;
use nanonn_core::pod;
use nanonn_core::{DataType, OpDesc, OpOptions, Result};

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
