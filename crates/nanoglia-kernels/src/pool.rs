//! `AVERAGE_POOL_2D` and `MAX_POOL_2D` on `f32` (TFLite reference `AveragePool` and `MaxPool`).

use crate::util::{
    clamp_f32, cmax, dims4, expect_arity, expect_type, fixed_kernel, float_activation, in_slice,
    input, out_slice, output, pod_struct, shape_mismatch, unsupported, window,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod;
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result};

pod_struct! {
    /// Parameters of a float pooling.
    pub(crate) struct PoolParams {
        batches: i32,
        in_h: i32,
        in_w: i32,
        channels: i32,
        out_h: i32,
        out_w: i32,
        filter_h: i32,
        filter_w: i32,
        stride_h: i32,
        stride_w: i32,
        pad_h: i32,
        pad_w: i32,
        /// Whether this is a max pooling (else average).
        max_pool: u32,
        min: f32,
        max: f32,
    }
}

fn pool_params(op: &OpDesc<'_, '_>) -> Result<PoolParams> {
    let max_pool = match op.builtin() {
        Some(BuiltinOp::MAX_POOL_2D) => true,
        Some(BuiltinOp::AVERAGE_POOL_2D) => false,
        _ => return Err(unsupported(op)),
    };
    let OpOptions::Pool2D(o) = op.options else {
        return Err(unsupported(op));
    };
    expect_arity(op, 1, 1)?;
    let (input, output) = (input(op, 0)?, output(op, 0)?);
    expect_type(input, DataType::F32)?;
    expect_type(output, DataType::F32)?;
    let i = dims4(input)?;
    let out = dims4(output)?;
    if i[0] != out[0] || i[3] != out[3] {
        return Err(shape_mismatch(output));
    }
    let w = window(
        output,
        o.padding,
        (i[1], i[2]),
        (o.filter_h, o.filter_w),
        (o.stride_h, o.stride_w),
        (1, 1),
        (out[1], out[2]),
    )?;
    let (min, max) = float_activation(op, o.activation)?;
    Ok(PoolParams {
        batches: i[0],
        in_h: i[1],
        in_w: i[2],
        channels: i[3],
        out_h: out[1],
        out_w: out[2],
        filter_h: o.filter_h,
        filter_w: o.filter_w,
        stride_h: o.stride_h,
        stride_w: o.stride_w,
        pad_h: w.pad_h,
        pad_w: w.pad_w,
        max_pool: u32::from(max_pool),
        min,
        max,
    })
}

fn pool_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: PoolParams = *pod::read(ctx.persist)?;
    let input = in_slice::<f32>(
        ctx.inputs,
        0,
        (p.batches * p.in_h * p.in_w * p.channels) as usize,
    )?;
    let output = out_slice::<f32>(
        ctx.outputs,
        0,
        (p.batches * p.out_h * p.out_w * p.channels) as usize,
    )?;
    let mut o = 0;
    for b in 0..p.batches {
        for out_y in 0..p.out_h {
            for out_x in 0..p.out_w {
                for c in 0..p.channels {
                    let in_x_origin = out_x * p.stride_w - p.pad_w;
                    let in_y_origin = out_y * p.stride_h - p.pad_h;
                    let fx_start = (-in_x_origin).max(0);
                    let fx_end = p.filter_w.min(p.in_w - in_x_origin);
                    let fy_start = (-in_y_origin).max(0);
                    let fy_end = p.filter_h.min(p.in_h - in_y_origin);
                    let at = |fy: i32, fx: i32| {
                        let (y, x) = (in_y_origin + fy, in_x_origin + fx);
                        input[(((b * p.in_h + y) * p.in_w + x) * p.channels + c) as usize]
                    };
                    let value = if p.max_pool != 0 {
                        let mut max = f32::MIN;
                        for fy in fy_start..fy_end {
                            for fx in fx_start..fx_end {
                                max = cmax(max, at(fy, fx));
                            }
                        }
                        max
                    } else {
                        let mut total = 0.0f32;
                        let mut count = 0.0f32;
                        for fy in fy_start..fy_end {
                            for fx in fx_start..fx_end {
                                total += at(fy, fx);
                                count += 1.0;
                            }
                        }
                        if count == 0.0 {
                            return Err(empty_window());
                        }
                        total / count
                    };
                    output[o] = clamp_f32(value, p.min, p.max);
                    o += 1;
                }
            }
        }
    }
    Ok(())
}

/// An empty pooling window (TFLite fails the same way).
fn empty_window() -> nanoglia_core::Error {
    nanoglia_core::Error::invalid_argument(nanoglia_core::Detail::None)
}

fixed_kernel!(
    /// `AVERAGE_POOL_2D` and `MAX_POOL_2D` on `f32`.
    POOL_F32,
    "rust/pool_f32",
    pool_params,
    pool_eval
);
