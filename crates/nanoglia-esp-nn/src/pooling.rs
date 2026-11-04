//! `AVERAGE_POOL_2D` and `MAX_POOL_2D` on int8 tensors, through `esp_nn_avg_pool_s8` and
//! `esp_nn_max_pool_s8`, one batch at a time.
//!
//! Port of esp-tflite-micro's `kernels/esp_nn/pooling.cc` with TFLite Micro's `PoolingPrepare`:
//! the input and the output share their quantization, so pooling works on the raw values and only
//! the fused activation is applied. Like esp-tflite-micro, layers whose channel count is not a
//! multiple of 4 run the ANSI C kernels (the ESP32-S3 ones need it).

use nanoglia_core::backend::{EvalCtx, FnKernel, KernelPlan};
use nanoglia_core::pod::Pod;
use nanoglia_core::{BuiltinOp, OpDesc, OpOptions, Result};
use nanoglia_esp_nn_sys as sys;

use crate::common;
use crate::fallback::{self, EspOp};
use crate::layout::{inconsistent, read_params, write_params};

/// The `AVERAGE_POOL_2D` kernel.
pub(crate) static AVERAGE: FnKernel = fallback::kernel::<Pool<false>>();

/// The `MAX_POOL_2D` kernel.
pub(crate) static MAX: FnKernel = fallback::kernel::<Pool<true>>();

/// ESP-NN parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    batches: i32,
    input_w: i32,
    input_h: i32,
    output_w: i32,
    output_h: i32,
    stride_w: i32,
    stride_h: i32,
    filter_w: i32,
    filter_h: i32,
    pad_w: i32,
    pad_h: i32,
    activation_min: i32,
    activation_max: i32,
    channels: i32,
}

// SAFETY: `#[repr(C)]` struct of `i32` fields only: no padding, any bit pattern is valid.
unsafe impl Pod for Params {}

fn analyze(op: &OpDesc<'_, '_>, max: bool) -> Option<Params> {
    let code = if max {
        BuiltinOp::MAX_POOL_2D
    } else {
        BuiltinOp::AVERAGE_POOL_2D
    };
    if op.builtin()? != code {
        return None;
    }
    let OpOptions::Pool2D(options) = op.options else {
        return None;
    };
    if op.inputs.len() != 1 || op.outputs.len() != 1 {
        return None;
    }
    let input = op.input(0)?;
    let output = op.output(0)?;
    let (input_scale, input_zero_point) = common::activation_quant(input)?;
    let (output_scale, output_zero_point) = common::activation_quant(output)?;
    if (input_scale - output_scale).abs() > 1.0e-6 || input_zero_point != output_zero_point {
        return None;
    }
    let [batches, in_h, in_w, channels] = common::dims4(input)?;
    let [out_n, out_h, out_w, out_c] = common::dims4(output)?;
    if out_n != batches || out_c != channels {
        return None;
    }
    let sizes = [
        options.stride_h,
        options.stride_w,
        options.filter_h,
        options.filter_w,
    ];
    if sizes.iter().any(|&v| v < 1) || !common::fits_u16(&sizes) {
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
    if (computed_h, computed_w) != (out_h, out_w) || !common::fits_u16(&[pad_h, pad_w]) {
        return None;
    }
    let (activation_min, activation_max) =
        common::activation_range(options.activation, output_scale, output_zero_point)?;
    Some(Params {
        batches,
        input_w: in_w,
        input_h: in_h,
        output_w: out_w,
        output_h: out_h,
        stride_w: options.stride_w,
        stride_h: options.stride_h,
        filter_w: options.filter_w,
        filter_h: options.filter_h,
        pad_w,
        pad_h,
        activation_min,
        activation_max,
        channels,
    })
}

/// The signature of the ESP-NN pooling functions.
type PoolFn = unsafe extern "C" fn(
    *const i8,
    u16,
    u16,
    *mut i8,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    i32,
    i32,
    u16,
);

/// `AVERAGE_POOL_2D` (`MAX = false`) or `MAX_POOL_2D` (`MAX = true`) on ESP-NN.
pub(crate) struct Pool<const MAX: bool>;

impl<const MAX: bool> EspOp for Pool<MAX> {
    const NAME: &'static str = if MAX {
        "esp-nn/max_pool2d_s8"
    } else {
        "esp-nn/average_pool2d_s8"
    };

    fn fallback() -> &'static FnKernel {
        if MAX {
            &nanoglia_cmsis_nn::kernels::MAX_POOL_2D_S8
        } else {
            &nanoglia_cmsis_nn::kernels::AVERAGE_POOL_2D_S8
        }
    }

    fn plan(op: &OpDesc<'_, '_>) -> Option<KernelPlan> {
        analyze(op, MAX)?;
        Some(KernelPlan {
            persistent: core::mem::size_of::<Params>(),
            scratch: 0,
        })
    }

    fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
        let p = analyze(op, MAX).ok_or_else(|| common::unsupported(op))?;
        write_params(persist, p)
    }

    fn run(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
        let p: Params = read_params(ctx.persist)?;
        let u16s = [
            p.input_w, p.input_h, p.output_w, p.output_h, p.stride_w, p.stride_h, p.filter_w,
            p.filter_h, p.pad_w, p.pad_h, p.channels,
        ]
        .map(|v| u16::try_from(v).unwrap_or(0));
        let [
            input_w,
            input_h,
            output_w,
            output_h,
            stride_w,
            stride_h,
            filter_w,
            filter_h,
            pad_w,
            pad_h,
            channels,
        ] = u16s;
        let input_len = usize::from(input_w) * usize::from(input_h) * usize::from(channels);
        let output_len = usize::from(output_w) * usize::from(output_h) * usize::from(channels);
        if input_len == 0 || output_len == 0 || stride_w == 0 || stride_h == 0 {
            return Err(inconsistent());
        }
        // The ESP32-S3 kernels need a multiple of 4 channels; the ANSI C ones take any count.
        let function: PoolFn = match (MAX, channels % 4 == 0) {
            (true, true) => sys::nanoglia_esp_nn_max_pool_s8,
            (true, false) => sys::esp_nn_max_pool_s8_ansi,
            (false, true) => sys::nanoglia_esp_nn_avg_pool_s8,
            (false, false) => sys::esp_nn_avg_pool_s8_ansi,
        };

        let input = common::input(ctx.inputs, 0);
        common::check_len(input, (input_len as i32).saturating_mul(p.batches))?;
        let output = ctx.output(0);
        common::check_len(output, (output_len as i32).saturating_mul(p.batches))?;
        for (input, output) in input
            .chunks_exact(input_len)
            .zip(output.chunks_exact_mut(output_len))
        {
            // SAFETY: one batch of the input and of the output, of the sizes ESP-NN derives from
            // the dimensions; distinct buffers.
            unsafe {
                function(
                    input.as_ptr().cast(),
                    input_w,
                    input_h,
                    output.as_mut_ptr().cast(),
                    output_w,
                    output_h,
                    stride_w,
                    stride_h,
                    filter_w,
                    filter_h,
                    pad_w,
                    pad_h,
                    p.activation_min,
                    p.activation_max,
                    channels,
                );
            }
        }
        Ok(())
    }
}
