//! `AVERAGE_POOL_2D` and `MAX_POOL_2D` on int8 tensors, through `arm_avgpool_s8` and
//! `arm_max_pool_s8`.
//!
//! Port of TFLite Micro's `kernels/cmsis_nn/pooling.cc` and `PoolingPrepare`: the input and the
//! output share their quantization (scales within 1e-6, equal zero points), so pooling works on
//! the raw values and only the fused activation is applied.

use nanoglia_cmsis_nn_sys as sys;
use nanoglia_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanoglia_core::pod::Pod;
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result};

use crate::common::{self, nhwc};
use crate::layout::{read_params, write_params};

/// The `AVERAGE_POOL_2D` kernel.
pub(crate) static AVERAGE: Kernel = Kernel {
    name: "cmsis-nn/average_pool2d_s8",
    plan,
    prepare,
    eval: eval_average,
};

/// The `MAX_POOL_2D` kernel.
pub(crate) static MAX: Kernel = Kernel {
    name: "cmsis-nn/max_pool2d_s8",
    plan,
    prepare,
    eval: eval_max,
};

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

fn analyze(op: &OpDesc<'_, '_>) -> Option<Params> {
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
    let (input_scale, input_zero_point) = common::activation_quant(input, DataType::I8)?;
    let (output_scale, output_zero_point) = common::activation_quant(output, DataType::I8)?;
    if (input_scale - output_scale).abs() > 1.0e-6 || input_zero_point != output_zero_point {
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
    let (act_min, act_max) = common::activation_range(
        options.activation,
        DataType::I8,
        output_scale,
        output_zero_point,
    )?;
    let scratch = if average {
        // SAFETY: plain integer arguments.
        common::buffer_size(unsafe { sys::arm_avgpool_s8_get_buffer_size(out_w, channels) })?
    } else {
        0
    };
    Some(Params {
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
        scratch,
    })
}

/// Whether this kernel runs `op`.
pub(crate) fn supports(op: &OpDesc<'_, '_>) -> bool {
    analyze(op).is_some()
}

fn plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    let p = analyze(op).ok_or_else(|| common::unsupported(op))?;
    Ok(KernelPlan {
        persistent: core::mem::size_of::<Params>(),
        scratch: p.scratch as usize,
    })
}

fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let p = analyze(op).ok_or_else(|| common::unsupported(op))?;
    write_params(persist, p)
}

type PoolFn = unsafe extern "C" fn(
    *const sys::cmsis_nn_context,
    *const sys::cmsis_nn_pool_params,
    *const sys::cmsis_nn_dims,
    *const i8,
    *const sys::cmsis_nn_dims,
    *const sys::cmsis_nn_dims,
    *mut i8,
) -> sys::arm_cmsis_nn_status;

fn eval_with(ctx: &mut EvalCtx<'_, '_>, function: PoolFn) -> Result<()> {
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

fn eval_average(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    eval_with(ctx, sys::arm_avgpool_s8)
}

fn eval_max(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    eval_with(ctx, sys::arm_max_pool_s8)
}
