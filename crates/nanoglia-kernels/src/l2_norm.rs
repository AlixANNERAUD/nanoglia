//! `L2_NORMALIZATION` on `f32` (TFLite `l2norm.cc`, `reference_ops::L2Normalization`): each
//! vector along the last dimension divided by its L2 norm, or by `1e-6` when the norm is smaller.
//! The sum of squares rounds as LiteRT's AArch64 build does (see [`dot_by_eight`]). TFLite
//! refuses fused activations.

use crate::math::dot_by_eight;
use crate::util::{
    cmax, elements, expect_arity, expect_type, fixed_kernel, in_slice, input, out_slice, output,
    pod_struct, same_shape, shape_mismatch, unsupported,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod;
use nanoglia_core::{Activation, DataType, OpDesc, OpOptions, Result};

/// The smallest norm a vector is divided by (`1e-6f` in `l2norm.cc`).
const EPSILON: f32 = 1e-6;

pod_struct! {
    /// Parameters of a float L2 normalization.
    pub(crate) struct L2NormParams {
        /// Number of vectors.
        outer: u32,
        /// Length of each vector (the last dimension).
        depth: u32,
    }
}

fn l2_norm_params(op: &OpDesc<'_, '_>) -> Result<L2NormParams> {
    let OpOptions::L2Norm { activation } = op.options else {
        return Err(unsupported(op));
    };
    if activation != Activation::None {
        return Err(unsupported(op));
    }
    expect_arity(op, 1, 1)?;
    let (input, output) = (input(op, 0)?, output(op, 0)?);
    expect_type(input, DataType::F32)?;
    expect_type(output, DataType::F32)?;
    let len = same_shape(input, output)?;
    let depth = match input.dims.as_slice().last() {
        Some(&d) if d >= 0 => d as usize,
        _ => return Err(shape_mismatch(input)),
    };
    elements(input)?;
    Ok(L2NormParams {
        outer: len.checked_div(depth).unwrap_or(0) as u32,
        depth: depth as u32,
    })
}

fn l2_norm_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: L2NormParams = *pod::read(ctx.persist)?;
    let (outer, depth) = (p.outer as usize, p.depth as usize);
    let input = in_slice::<f32>(ctx.inputs, 0, outer * depth)?;
    let output = out_slice::<f32>(ctx.outputs, 0, outer * depth)?;
    if depth == 0 {
        return Ok(());
    }
    for (x, y) in input
        .chunks_exact(depth)
        .zip(output.chunks_exact_mut(depth))
    {
        let squared = dot_by_eight(depth, |c| (x[c], x[c]));
        let norm = cmax(libm::sqrtf(squared), EPSILON);
        for (y, &v) in y.iter_mut().zip(x) {
            *y = v / norm;
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `L2_NORMALIZATION` on `f32`.
    L2_NORMALIZATION_F32,
    "rust/l2_normalization_f32",
    l2_norm_params,
    l2_norm_eval
);
