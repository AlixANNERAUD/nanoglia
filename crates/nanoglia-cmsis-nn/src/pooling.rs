//! `AVERAGE_POOL_2D` and `MAX_POOL_2D` on int8 tensors, through `arm_avgpool_s8` and
//! `arm_max_pool_s8`, and on int16 tensors, through `arm_avgpool_s16` and `arm_max_pool_s16`.
//!
//! Port of TFLite Micro's `kernels/cmsis_nn/pooling.cc` and `PoolingPrepare`: the input and the
//! output share their quantization (scales within 1e-6, equal zero points; zero points 0 for
//! int16), so pooling works on the raw values and only the fused activation is applied.
//!
//! The int16 kernels compute TFLite's `reference_integer_ops::AveragePool` and `MaxPool` for
//! `int16_t`: a 32-bit sum rounded half away from zero (`(sum ± count / 2) / count`), and the
//! window maximum. Average pooling is claimed for at most 65535 filter values, so that neither
//! the sum nor its rounding overflows an `int32_t`.

use nanoglia_cmsis_nn_sys as sys;
use nanoglia_core::backend::{EvalCtx, FnKernel, KernelPlan};
use nanoglia_core::pod::{Pod, cast_slice, cast_slice_mut};
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result};

use crate::common::{self, nhwc};
use crate::layout::{read_params, write_params};

/// The `AVERAGE_POOL_2D` kernel.
pub static AVERAGE: FnKernel = FnKernel {
    name: "cmsis-nn/average_pool2d_s8",
    plan,
    prepare,
    eval: eval_average,
};

/// The `MAX_POOL_2D` kernel.
pub static MAX: FnKernel = FnKernel {
    name: "cmsis-nn/max_pool2d_s8",
    plan,
    prepare,
    eval: eval_max,
};

/// The int16 `AVERAGE_POOL_2D` kernel.
pub(crate) static AVERAGE_S16: FnKernel = FnKernel {
    name: "cmsis-nn/average_pool2d_s16",
    plan,
    prepare,
    eval: eval_average_s16,
};

/// The int16 `MAX_POOL_2D` kernel.
pub(crate) static MAX_S16: FnKernel = FnKernel {
    name: "cmsis-nn/max_pool2d_s16",
    plan,
    prepare,
    eval: eval_max_s16,
};

/// The most values an int16 average pooling window may hold: `65535 × 32768 + 65535 / 2` is the
/// largest magnitude that still fits an `int32_t`.
const S16_MAX_AVERAGE_WINDOW: i64 = 65535;

/// Persistent parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    pool: sys::cmsis_nn_pool_params,
    input: sys::cmsis_nn_dims,
    filter: sys::cmsis_nn_dims,
    output: sys::cmsis_nn_dims,
    /// Scratch bytes.
    scratch: i32,
}

// SAFETY: `#[repr(C)]` struct of `i32` fields only: no padding, any bit pattern is valid.
unsafe impl Pod for Params {}

/// The parameters of `op`, and whether its tensors are int16.
fn analyze(op: &OpDesc<'_, '_>) -> Option<(Params, bool)> {
    let average = match op.builtin()? {
        BuiltinOp::AVERAGE_POOL_2D => true,
        BuiltinOp::MAX_POOL_2D => false,
        _ => return None,
    };
    let OpOptions::Pool2D(options) = op.options else {
        return None;
    };
    if op.inputs.len() != 1 || op.outputs.len() != 1 {
        return None;
    }
    let input = op.input(0)?;
    let output = op.output(0)?;
    let dtype = input.dtype;
    let int16 = dtype == DataType::I16;
    let (input_scale, input_zero_point) = common::activation_quant(input, dtype)?;
    let (output_scale, output_zero_point) = common::activation_quant(output, dtype)?;
    if (input_scale - output_scale).abs() > 1.0e-6 || input_zero_point != output_zero_point {
        return None;
    }
    if int16 && output_zero_point != 0 {
        return None;
    }
    let [batches, in_h, in_w, channels] = common::dims4(input)?;
    let [out_n, out_h, out_w, out_c] = common::dims4(output)?;
    if out_n != batches || out_c != channels {
        return None;
    }
    let in_range = |v: i32| (1..=i16::MAX as i32).contains(&v);
    if !(in_range(options.stride_h)
        && in_range(options.stride_w)
        && in_range(options.filter_h)
        && in_range(options.filter_w))
    {
        return None;
    }
    if int16
        && average
        && i64::from(options.filter_h) * i64::from(options.filter_w) > S16_MAX_AVERAGE_WINDOW
    {
        return None;
    }
    let (computed_h, pad_h) = common::output_size_and_padding(
        options.padding,
        in_h,
        options.filter_h,
        options.stride_h,
        1,
    )?;
    let (computed_w, pad_w) = common::output_size_and_padding(
        options.padding,
        in_w,
        options.filter_w,
        options.stride_w,
        1,
    )?;
    if (computed_h, computed_w) != (out_h, out_w) {
        return None;
    }
    let (act_min, act_max) =
        common::activation_range(options.activation, dtype, output_scale, output_zero_point)?;
    let scratch = match (average, int16) {
        // SAFETY: plain integer arguments.
        (true, false) => unsafe { sys::arm_avgpool_s8_get_buffer_size(out_w, channels) },
        // SAFETY: plain integer arguments.
        (true, true) => unsafe { sys::arm_avgpool_s16_get_buffer_size(out_w, channels) },
        (false, _) => 0,
    };
    let params = Params {
        pool: sys::cmsis_nn_pool_params {
            stride: sys::cmsis_nn_tile {
                w: options.stride_w,
                h: options.stride_h,
            },
            padding: sys::cmsis_nn_tile { w: pad_w, h: pad_h },
            activation: sys::cmsis_nn_activation {
                min: act_min,
                max: act_max,
            },
        },
        input: nhwc(batches, in_h, in_w, channels),
        filter: nhwc(1, options.filter_h, options.filter_w, 1),
        output: nhwc(batches, out_h, out_w, channels),
        scratch: common::buffer_size(scratch)?,
    };
    Some((params, int16))
}

/// The kernel running the `AVERAGE_POOL_2D` operator `op`, if any.
#[cfg(feature = "op-average-pool-2d")]
pub(crate) fn select_average(op: &OpDesc<'_, '_>) -> Option<&'static FnKernel> {
    Some(if is_int16(op)? && crate::INT16 {
        &AVERAGE_S16
    } else {
        &AVERAGE
    })
}

/// The kernel running the `MAX_POOL_2D` operator `op`, if any.
#[cfg(feature = "op-max-pool-2d")]
pub(crate) fn select_max(op: &OpDesc<'_, '_>) -> Option<&'static FnKernel> {
    Some(if is_int16(op)? && crate::INT16 {
        &MAX_S16
    } else {
        &MAX
    })
}

/// Whether the supported operator `op` is an int16 one; `None` if no kernel runs it. The callers
/// test [`crate::INT16`] again, a constant, so that without the `int16` feature the compiler sees
/// that nothing references the int16 kernels.
fn is_int16(op: &OpDesc<'_, '_>) -> Option<bool> {
    let (_, int16) = analyze(op)?;
    (!int16 || crate::INT16).then_some(int16)
}

fn plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    let (p, _) = analyze(op).ok_or_else(|| common::unsupported(op))?;
    Ok(KernelPlan {
        persistent: core::mem::size_of::<Params>(),
        scratch: p.scratch as usize,
    })
}

fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let (p, _) = analyze(op).ok_or_else(|| common::unsupported(op))?;
    write_params(persist, p)
}

type PoolFn<T> = unsafe extern "C" fn(
    *const sys::cmsis_nn_context,
    *const sys::cmsis_nn_pool_params,
    *const sys::cmsis_nn_dims,
    *const T,
    *const sys::cmsis_nn_dims,
    *const sys::cmsis_nn_dims,
    *mut T,
) -> sys::arm_cmsis_nn_status;

fn eval_with(ctx: &mut EvalCtx<'_, '_>, function: PoolFn<i8>) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let input = common::input(ctx.inputs, 0);
    common::check_len(input, common::dims_len(&p.input))?;
    let scratch = common::scratch_context(ctx.scratch, p.scratch)?;
    let output = ctx.output(0);
    common::check_len(output, common::dims_len(&p.output))?;
    // SAFETY: the input and output sizes match the dimensions, the scratch buffer is as large as
    // CMSIS-NN asked for, and the output does not overlap the input (distinct borrows).
    let status = unsafe {
        function(
            &scratch,
            &p.pool,
            &p.input,
            input.as_ptr().cast(),
            &p.filter,
            &p.output,
            output.as_mut_ptr().cast(),
        )
    };
    common::check_status(status)
}

fn eval_with_s16(ctx: &mut EvalCtx<'_, '_>, function: PoolFn<i16>) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let input = common::input(ctx.inputs, 0);
    common::check_len(input, common::dims_len(&p.input).saturating_mul(2))?;
    let input = cast_slice::<i16>(input)?;
    let scratch = common::scratch_context(ctx.scratch, p.scratch)?;
    let output = ctx.output(0);
    common::check_len(output, common::dims_len(&p.output).saturating_mul(2))?;
    let output = cast_slice_mut::<i16>(output)?;
    // SAFETY: the input and output sizes match the dimensions and both are aligned for `i16`,
    // the scratch buffer is as large as CMSIS-NN asked for, and the output does not overlap the
    // input (distinct borrows).
    let status = unsafe {
        function(
            &scratch,
            &p.pool,
            &p.input,
            input.as_ptr(),
            &p.filter,
            &p.output,
            output.as_mut_ptr(),
        )
    };
    common::check_status(status)
}

fn eval_average(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    eval_with(ctx, sys::arm_avgpool_s8)
}

fn eval_max(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    eval_with(ctx, sys::arm_max_pool_s8)
}

fn eval_average_s16(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    eval_with_s16(ctx, sys::arm_avgpool_s16)
}

fn eval_max_s16(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    eval_with_s16(ctx, sys::arm_max_pool_s16)
}
