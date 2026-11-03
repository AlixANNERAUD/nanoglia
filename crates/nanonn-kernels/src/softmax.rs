//! `SOFTMAX` on `f32` (TFLite reference `Softmax`), over the last dimension.

use crate::util::{
    cmax, expect_arity, expect_type, fixed_kernel, in_slice, input, out_slice, output, pod_struct,
    same_shape, shape_mismatch, unsupported,
};
use nanonn_core::backend::EvalCtx;
use nanonn_core::pod;
use nanonn_core::{DataType, OpDesc, OpOptions, Result};

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
