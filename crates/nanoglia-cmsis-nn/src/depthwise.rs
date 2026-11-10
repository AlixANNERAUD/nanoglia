//! `DEPTHWISE_CONV_2D` with int8 activations, int8 weights (per-tensor or per-output-channel) and
//! int32 bias, through `arm_depthwise_conv_wrapper_s8`, and with int16 activations (16x8: zero
//! points 0, int8 weights, int64 bias) through `arm_depthwise_conv_wrapper_s16`.
//!
//! Port of TFLite Micro's `kernels/cmsis_nn/depthwise_conv.cc` and of
//! `CalculateOpDataDepthwiseConv`. The depth multiplier is derived from the shapes (output
//! channels / input channels) like LiteRT does, rather than read from the options.
//!
//! The int16 wrapper runs `arm_depthwise_conv_fast_s16` for a depth multiplier of 1, no dilation
//! and fewer than 512 filter values per channel (with the DSP or Helium extension; elsewhere it
//! runs the generic code), `arm_depthwise_conv_s16` otherwise. TensorFlow Lite Micro calls
//! `arm_depthwise_conv_s16` directly; both compute TFLite's reference result
//! (`reference_integer_ops::DepthwiseConvPerChannel` with a 64-bit accumulator and
//! `MultiplyByQuantizedMultiplier` for `int64_t`) under these 16x8 restrictions:
//!
//! - int32 biases are refused: LiteRT reads the bias of a 16x8 depthwise convolution as int64;
//! - requantization shifts must stay below 8 (TFLite's 64-bit `MultiplyByQuantizedMultiplier`);
//! - CMSIS-NN's `arm_nn_requantize_s64` truncates `acc * multiplier >> (14 - shift)` to 32 bits
//!   before its last rounding shift, where TFLite keeps 64 bits (issue #38): the kernel is claimed
//!   only when the largest possible accumulator of each channel (`2^15 × Σ|weight| + |bias|`, see
//!   `common::s16_accumulators_exact`) keeps that value in an `int32_t` and the product in 62
//!   bits, so that both agree. A bias computed at run time cannot be bounded and is refused;
//! - CMSIS-NN keeps the top-left input coordinate of each window in an `int16_t`, so
//!   `(output size - 1) × stride - padding` must fit one.

use nanoglia_cmsis_nn_sys as sys;
use nanoglia_core::backend::{EvalCtx, FnKernel, KernelPlan};
use nanoglia_core::pod::{Pod, cast_slice, cast_slice_mut};
use nanoglia_core::quant::{self, Multiplier};
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result, TensorDesc};

use crate::common::{self, nhwc};
use crate::layout::{ChannelLayout, read_params};

/// The `DEPTHWISE_CONV_2D` kernel.
pub static KERNEL: FnKernel = FnKernel {
    name: "cmsis-nn/depthwise_conv2d_s8",
    plan,
    prepare,
    eval,
};

/// The 16x8 `DEPTHWISE_CONV_2D` kernel.
pub(crate) static KERNEL_S16: FnKernel = FnKernel {
    name: "cmsis-nn/depthwise_conv2d_s16",
    plan,
    prepare,
    eval: eval_s16,
};

/// Persistent parameters, followed by the per-channel multipliers and shifts.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    conv: sys::cmsis_nn_dw_conv_params,
    input: sys::cmsis_nn_dims,
    filter: sys::cmsis_nn_dims,
    output: sys::cmsis_nn_dims,
    /// Number of output channels.
    channels: i32,
    /// Whether the bias input is present.
    has_bias: i32,
    /// Scratch bytes.
    scratch: i32,
}

// SAFETY: `#[repr(C)]` struct of `i32` fields only: no padding, any bit pattern is valid.
unsafe impl Pod for Params {}

struct Analysis<'a> {
    params: Params,
    input: &'a TensorDesc<'a>,
    filter: &'a TensorDesc<'a>,
    output: &'a TensorDesc<'a>,
}

/// TFLite `PopulateConvolutionQuantizationParams` for output channel `channel`.
fn channel_multiplier(
    input: &TensorDesc<'_>,
    filter: &TensorDesc<'_>,
    output: &TensorDesc<'_>,
    channel: usize,
) -> Multiplier {
    quant::quantize_multiplier(quant::conv_effective_scale(
        input.quant.scale(),
        common::weight_scale(filter, channel),
        output.quant.scale(),
    ))
}

fn analyze<'a>(op: &'a OpDesc<'a, '_>) -> Option<Analysis<'a>> {
    if op.builtin()? != BuiltinOp::DEPTHWISE_CONV_2D || op.outputs.len() != 1 {
        return None;
    }
    let OpOptions::DepthwiseConv2D(options) = op.options else {
        return None;
    };
    if !(2..=3).contains(&op.inputs.len()) {
        return None;
    }
    let input = op.input(0)?;
    let filter = op.input(1)?;
    let bias = op.input(2);
    let output = op.output(0)?;

    let dtype = input.dtype;
    let int16 = dtype == DataType::I16;
    let (_, input_zero_point) = common::activation_quant(input, dtype)?;
    let (output_scale, output_zero_point) = common::activation_quant(output, dtype)?;
    if filter.dtype != DataType::I8 || (int16 && (input_zero_point != 0 || output_zero_point != 0))
    {
        return None;
    }
    let [batches, in_h, in_w, in_c] = common::dims4(input)?;
    let [filter_n, filter_h, filter_w, out_c] = common::dims4(filter)?;
    let [out_n, out_h, out_w, out_c2] = common::dims4(output)?;
    if filter_n != 1 || out_n != batches || out_c2 != out_c || out_c % in_c != 0 {
        return None;
    }
    let depth_multiplier = out_c / in_c;
    common::weight_quant(filter, out_c as usize, 3)?;
    let bias_type = if int16 { DataType::I64 } else { DataType::I32 };
    let has_bias = match bias {
        Some(b) => {
            if b.dtype != bias_type || common::element_count(&b.dims)? != out_c {
                return None;
            }
            true
        }
        None => false,
    };

    let in_range = |v: i32| (1..=i16::MAX as i32).contains(&v);
    let (stride_h, stride_w) = (options.stride_h, options.stride_w);
    let (dilation_h, dilation_w) = (options.dilation_h, options.dilation_w);
    if !(in_range(stride_h) && in_range(stride_w) && in_range(dilation_h) && in_range(dilation_w)) {
        return None;
    }
    let (computed_h, pad_h) =
        common::output_size_and_padding(options.padding, in_h, filter_h, stride_h, dilation_h)?;
    let (computed_w, pad_w) =
        common::output_size_and_padding(options.padding, in_w, filter_w, stride_w, dilation_w)?;
    if (computed_h, computed_w) != (out_h, out_w) {
        return None;
    }
    // CMSIS-NN keeps the window origin `out * stride - pad` in an `int16_t`
    // (`arm_depthwise_conv_s8.c`, `_s8_opt.c`, `_s16.c` and `_fast_s16.c`).
    let fits_i16 = |out: i32, stride: i32, pad: i32| {
        let last = i64::from(out - 1) * i64::from(stride) - i64::from(pad);
        pad <= i16::MAX as i32 && last <= i64::from(i16::MAX)
    };
    if !(fits_i16(out_h, stride_h, pad_h) && fits_i16(out_w, stride_w, pad_w)) {
        return None;
    }
    if int16 {
        // Workaround for CMSIS-NN v8.0.0's 32-bit requantization intermediate (issue #38, see
        // `common::s16_requantize_exact`): remove once a fixed CMSIS-NN release is vendored.
        let (channels, taps) = (out_c as usize, (filter_h * filter_w) as usize);
        common::s16_accumulators_exact(
            filter,
            bias,
            channels,
            taps,
            |c, t| t * channels + c,
            |c| channel_multiplier(input, filter, output, c),
        )?;
    }
    let (act_min, act_max) =
        common::activation_range(options.activation, dtype, output_scale, output_zero_point)?;

    let conv = sys::cmsis_nn_dw_conv_params {
        input_offset: -input_zero_point,
        output_offset: output_zero_point,
        ch_mult: depth_multiplier,
        stride: sys::cmsis_nn_tile {
            w: stride_w,
            h: stride_h,
        },
        padding: sys::cmsis_nn_tile { w: pad_w, h: pad_h },
        dilation: sys::cmsis_nn_tile {
            w: dilation_w,
            h: dilation_h,
        },
        activation: sys::cmsis_nn_activation {
            min: act_min,
            max: act_max,
        },
    };
    let input_dims = nhwc(batches, in_h, in_w, in_c);
    let filter_dims = nhwc(filter_n, filter_h, filter_w, out_c);
    let output_dims = nhwc(batches, out_h, out_w, out_c);
    // SAFETY: the pointers are to live locals; the functions only read them.
    let scratch = unsafe {
        if int16 {
            sys::arm_depthwise_conv_wrapper_s16_get_buffer_size(
                &conv,
                &input_dims,
                &filter_dims,
                &output_dims,
            )
        } else {
            sys::arm_depthwise_conv_wrapper_s8_get_buffer_size(
                &conv,
                &input_dims,
                &filter_dims,
                &output_dims,
            )
        }
    };
    let params = Params {
        conv,
        input: input_dims,
        filter: filter_dims,
        output: output_dims,
        channels: out_c,
        has_bias: has_bias as i32,
        scratch: common::buffer_size(scratch)?,
    };
    Some(Analysis {
        params,
        input,
        filter,
        output,
    })
}

/// The kernel running `op`, if any.
pub(crate) fn select(op: &OpDesc<'_, '_>) -> Option<&'static FnKernel> {
    let a = analyze(op)?;
    Some(if a.input.dtype == DataType::I16 {
        if !crate::INT16 {
            return None;
        }
        &KERNEL_S16
    } else {
        &KERNEL
    })
}

fn plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    let a = analyze(op).ok_or_else(|| common::unsupported(op))?;
    Ok(KernelPlan {
        persistent: ChannelLayout::<Params>::new(a.params.channels as usize, 0).size,
        scratch: a.params.scratch as usize,
    })
}

fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let a = analyze(op).ok_or_else(|| common::unsupported(op))?;
    let layout = ChannelLayout::<Params>::new(a.params.channels as usize, 0);
    let multipliers =
        (0..a.params.channels as usize).map(|c| channel_multiplier(a.input, a.filter, a.output, c));
    layout.write(persist, a.params, multipliers)
}

fn eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let layout = ChannelLayout::<Params>::new(p.channels as usize, 0);
    let multiplier = layout.multiplier.get(ctx.persist)?;
    let shift = layout.shift.get(ctx.persist)?;
    let quant_params = sys::cmsis_nn_per_channel_quant_params {
        // CMSIS-NN only reads the arrays despite the non-const pointers.
        multiplier: multiplier.as_ptr().cast_mut(),
        shift: shift.as_ptr().cast_mut(),
    };

    let input = common::input(ctx.inputs, 0);
    let filter = common::input(ctx.inputs, 1);
    common::check_len(input, common::dims_len(&p.input))?;
    common::check_len(filter, common::dims_len(&p.filter))?;
    let bias = common::optional_i32(common::input(ctx.inputs, 2), p.has_bias != 0, p.channels)?;
    let bias_dims = nhwc(1, 1, 1, p.channels);
    let scratch = common::scratch_context(ctx.scratch, p.scratch)?;
    let output = ctx.output(0);
    common::check_len(output, common::dims_len(&p.output))?;

    // SAFETY: every buffer was checked against the sizes CMSIS-NN derives from the dimensions,
    // the bias is aligned, the scratch buffer is as large as CMSIS-NN asked for, and the
    // per-channel arrays have `output.c` entries.
    let status = unsafe {
        sys::arm_depthwise_conv_wrapper_s8(
            &scratch,
            &p.conv,
            &quant_params,
            &p.input,
            input.as_ptr().cast(),
            &p.filter,
            filter.as_ptr().cast(),
            &bias_dims,
            bias,
            &p.output,
            output.as_mut_ptr().cast(),
        )
    };
    common::check_status(status)
}

fn eval_s16(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let layout = ChannelLayout::<Params>::new(p.channels as usize, 0);
    let multiplier = layout.multiplier.get(ctx.persist)?;
    let shift = layout.shift.get(ctx.persist)?;
    let quant_params = sys::cmsis_nn_per_channel_quant_params {
        // CMSIS-NN only reads the arrays despite the non-const pointers.
        multiplier: multiplier.as_ptr().cast_mut(),
        shift: shift.as_ptr().cast_mut(),
    };

    let input = common::input(ctx.inputs, 0);
    let filter = common::input(ctx.inputs, 1);
    common::check_len(input, common::dims_len(&p.input).saturating_mul(2))?;
    let input = cast_slice::<i16>(input)?;
    common::check_len(filter, common::dims_len(&p.filter))?;
    let bias = if p.has_bias != 0 {
        let bias = common::input(ctx.inputs, 2);
        common::check_len(bias, p.channels.saturating_mul(8))?;
        cast_slice::<i64>(bias)?.as_ptr()
    } else {
        core::ptr::null()
    };
    let bias_dims = nhwc(1, 1, 1, p.channels);
    let scratch = common::scratch_context(ctx.scratch, p.scratch)?;
    let output = ctx.output(0);
    common::check_len(output, common::dims_len(&p.output).saturating_mul(2))?;
    let output = cast_slice_mut::<i16>(output)?;

    // SAFETY: every buffer was checked against the sizes CMSIS-NN derives from the dimensions and
    // is aligned for its type, the scratch buffer is as large as CMSIS-NN asked for, and the
    // per-channel arrays have `output.c` entries.
    let status = unsafe {
        sys::arm_depthwise_conv_wrapper_s16(
            &scratch,
            &p.conv,
            &quant_params,
            &p.input,
            input.as_ptr(),
            &p.filter,
            filter.as_ptr().cast(),
            &bias_dims,
            bias,
            &p.output,
            output.as_mut_ptr(),
        )
    };
    common::check_status(status)
}
