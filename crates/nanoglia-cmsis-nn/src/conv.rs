//! `CONV_2D` with int8 activations, int8 weights (per-tensor or per-output-channel) and int32
//! bias, through `arm_convolve_wrapper_s8`, and with int16 activations (16x8: zero points 0, int8
//! weights, int64 bias) through `arm_convolve_wrapper_s16`.
//!
//! Port of TFLite Micro's `kernels/cmsis_nn/conv.cc` (Prepare, EvalQuantizedPerChannel and
//! EvalQuantizedPerChannel16x8) and of `CalculateOpDataConv`/
//! `PopulateConvolutionQuantizationParams`. Grouped convolutions (input channels a multiple of the
//! filter's) are supported like in the TFLite reference kernel, for int8 only.
//!
//! 16x8 restrictions, where CMSIS-NN or LiteRT would not compute the reference result:
//!
//! - int32 biases are refused: LiteRT picks the bias type from the `quantized_bias_type` option,
//!   which nanoglia does not decode, and reads the bias as int64 by default;
//! - CMSIS-NN accumulates some outputs in 32 bits before adding the bias, so the filter may have
//!   at most 511 values per output channel (`|int8 * int16| <= 2^22`);
//! - requantization shifts must stay below 8 (TFLite's 64-bit `MultiplyByQuantizedMultiplier`);
//! - CMSIS-NN's `arm_nn_requantize_s64` truncates `acc * multiplier >> (14 - shift)` to 32 bits
//!   before its last rounding shift, where TFLite keeps 64 bits (issue #38): the kernel is claimed
//!   only when the largest possible accumulator of each output channel
//!   (`2^15 × Σ|weight| + |bias|`, see `common::s16_accumulators_exact`) keeps that value in an
//!   `int32_t`. A bias computed at run time cannot be bounded and is refused; there is no Rust
//!   16x8 convolution to fall back to, so such a model does not load.
//!
//! CMSIS-NN v8.0.0 advances the bias pointer of grouped convolutions even when it is null (and then
//! reads through it), so a grouped convolution without bias gets an all-zero bias in its
//! persistent buffer.

use nanoglia_cmsis_nn_sys as sys;
use nanoglia_core::backend::{EvalCtx, FnKernel, KernelPlan};
use nanoglia_core::pod::{Pod, cast_slice, cast_slice_mut};
use nanoglia_core::quant::{self, Multiplier};
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result, TensorDesc};

use crate::common::{self, nhwc};
use crate::layout::{ChannelLayout, read_params};

/// The `CONV_2D` kernel.
pub static KERNEL: FnKernel = FnKernel {
    name: "cmsis-nn/conv2d_s8",
    plan,
    prepare,
    eval,
};

/// The 16x8 `CONV_2D` kernel.
pub(crate) static KERNEL_S16: FnKernel = FnKernel {
    name: "cmsis-nn/conv2d_s16",
    plan,
    prepare,
    eval: eval_s16,
};

/// The most values per output channel of a 16x8 filter: CMSIS-NN may accumulate them in an
/// `int32_t` before adding the int64 bias.
const S16_MAX_FILTER_VALUES: i32 = 511;

/// Persistent parameters, followed by the per-channel multipliers and shifts.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    conv: sys::cmsis_nn_conv_params,
    input: sys::cmsis_nn_dims,
    filter: sys::cmsis_nn_dims,
    output: sys::cmsis_nn_dims,
    /// Number of output channels.
    channels: i32,
    /// Elements of the filter tensor.
    filter_len: i32,
    /// Whether the bias input is present.
    has_bias: i32,
    /// Whether an all-zero bias follows the per-channel arrays (grouped convolution without bias).
    zero_bias: i32,
    /// Scratch bytes.
    scratch: i32,
}

// SAFETY: `#[repr(C)]` struct of `i32` fields only (the CMSIS-NN structs are made of `i32`):
// no padding, any bit pattern is valid, alignment 4.
unsafe impl Pod for Params {}

/// Everything `select`, `plan` and `prepare` derive from the operator.
struct Analysis<'a> {
    params: Params,
    input: &'a TensorDesc<'a>,
    filter: &'a TensorDesc<'a>,
    output: &'a TensorDesc<'a>,
}

fn analyze<'a>(op: &'a OpDesc<'a, '_>) -> Option<Analysis<'a>> {
    if op.builtin()? != BuiltinOp::CONV_2D || op.outputs.len() != 1 {
        return None;
    }
    let OpOptions::Conv2D(options) = op.options else {
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
    let (input_scale, input_zero_point) = common::activation_quant(input, dtype)?;
    let (output_scale, output_zero_point) = common::activation_quant(output, dtype)?;
    if filter.dtype != DataType::I8 || (int16 && (input_zero_point != 0 || output_zero_point != 0))
    {
        return None;
    }
    let [batches, in_h, in_w, in_c] = common::dims4(input)?;
    let [out_c, filter_h, filter_w, filter_c] = common::dims4(filter)?;
    let [out_n, out_h, out_w, out_c2] = common::dims4(output)?;
    if out_n != batches || out_c2 != out_c || in_c % filter_c != 0 {
        return None;
    }
    let groups = in_c / filter_c;
    if out_c % groups != 0 || (int16 && groups != 1) {
        return None;
    }
    common::weight_quant(filter, out_c as usize, 0)?;
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
    if int16 {
        if filter_h * filter_w * filter_c > S16_MAX_FILTER_VALUES {
            return None;
        }
        // Shifts below 8, and the workaround for CMSIS-NN v8.0.0's 32-bit requantization
        // intermediate (issue #38, see `common::s16_requantize_exact`): remove the accumulator
        // bound once a fixed CMSIS-NN release is vendored.
        let (channels, taps) = (out_c as usize, (filter_h * filter_w * filter_c) as usize);
        common::s16_accumulators_exact(
            filter,
            bias,
            channels,
            taps,
            |c, t| c * taps + t,
            |c| {
                quant::quantize_multiplier(quant::conv_effective_scale(
                    input_scale,
                    common::weight_scale(filter, c),
                    output_scale,
                ))
            },
        )?;
    }

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
    let (act_min, act_max) =
        common::activation_range(options.activation, dtype, output_scale, output_zero_point)?;

    let conv = sys::cmsis_nn_conv_params {
        input_offset: -input_zero_point,
        output_offset: output_zero_point,
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
    // Like TFLite Micro: the output channel count is taken from `output_dims`.
    let filter_dims = nhwc(1, filter_h, filter_w, filter_c);
    let output_dims = nhwc(batches, out_h, out_w, out_c);
    // SAFETY: the pointers are to live locals; the functions only read them.
    let scratch = unsafe {
        if int16 {
            sys::arm_convolve_wrapper_s16_get_buffer_size(
                &conv,
                &input_dims,
                &filter_dims,
                &output_dims,
            )
        } else {
            sys::arm_convolve_wrapper_s8_get_buffer_size(
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
        filter_len: common::element_count(&filter.dims)?,
        has_bias: has_bias as i32,
        zero_bias: (!has_bias && groups > 1) as i32,
        scratch: common::buffer_size(scratch)?,
    };
    Some(Analysis {
        params,
        input,
        filter,
        output,
    })
}

fn layout(p: &Params) -> ChannelLayout<Params> {
    let zero_bias = if p.zero_bias != 0 { p.channels } else { 0 };
    ChannelLayout::new(p.channels as usize, zero_bias as usize)
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
        persistent: layout(&a.params).size,
        scratch: a.params.scratch as usize,
    })
}

fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let a = analyze(op).ok_or_else(|| common::unsupported(op))?;
    let layout = layout(&a.params);
    let input_scale = a.input.quant.scale();
    let output_scale = a.output.quant.scale();
    let multipliers = (0..a.params.channels as usize).map(|c| -> Multiplier {
        let filter_scale = common::weight_scale(a.filter, c);
        quant::quantize_multiplier(quant::conv_effective_scale(
            input_scale,
            filter_scale,
            output_scale,
        ))
    });
    layout.write(persist, a.params, multipliers)?;
    layout.extra.get_mut(persist)?.fill(0);
    Ok(())
}

fn eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let persist = ctx.persist;
    let p: Params = read_params(persist)?;
    let layout = layout(&p);
    let multiplier = layout.multiplier.get(persist)?;
    let shift = layout.shift.get(persist)?;
    let quant_params = sys::cmsis_nn_per_channel_quant_params {
        // CMSIS-NN only reads the arrays despite the non-const pointers.
        multiplier: multiplier.as_ptr().cast_mut(),
        shift: shift.as_ptr().cast_mut(),
    };

    let input = common::input(ctx.inputs, 0);
    let filter = common::input(ctx.inputs, 1);
    common::check_len(input, common::dims_len(&p.input))?;
    common::check_len(filter, p.filter_len)?;
    let bias = if p.zero_bias != 0 {
        layout.extra.get(persist)?.as_ptr()
    } else {
        common::optional_i32(common::input(ctx.inputs, 2), p.has_bias != 0, p.channels)?
    };
    let bias_dims = nhwc(1, 1, 1, p.channels);
    let scratch = common::scratch_context(ctx.scratch, p.scratch)?;
    let output = ctx.output(0);
    common::check_len(output, common::dims_len(&p.output))?;

    // SAFETY: every buffer was checked against the sizes CMSIS-NN derives from the dimensions,
    // the bias is aligned, the scratch buffer is as large as CMSIS-NN asked for, and the
    // per-channel arrays have `output.c` entries.
    let status = unsafe {
        sys::arm_convolve_wrapper_s8(
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
    let persist = ctx.persist;
    let p: Params = read_params(persist)?;
    let layout = layout(&p);
    let multiplier = layout.multiplier.get(persist)?;
    let shift = layout.shift.get(persist)?;
    let quant_params = sys::cmsis_nn_per_channel_quant_params {
        // CMSIS-NN only reads the arrays despite the non-const pointers.
        multiplier: multiplier.as_ptr().cast_mut(),
        shift: shift.as_ptr().cast_mut(),
    };

    let input = common::input(ctx.inputs, 0);
    let filter = common::input(ctx.inputs, 1);
    common::check_len(input, common::dims_len(&p.input).saturating_mul(2))?;
    let input = cast_slice::<i16>(input)?;
    common::check_len(filter, p.filter_len)?;
    let bias = if p.has_bias != 0 {
        let bias = common::input(ctx.inputs, 2);
        common::check_len(bias, p.channels.saturating_mul(8))?;
        cast_slice::<i64>(bias)?.as_ptr()
    } else {
        core::ptr::null()
    };
    let bias = sys::cmsis_nn_bias_data {
        data: bias.cast(),
        is_int32_bias: false,
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
        sys::arm_convolve_wrapper_s16(
            &scratch,
            &p.conv,
            &quant_params,
            &p.input,
            input.as_ptr(),
            &p.filter,
            filter.as_ptr().cast(),
            &bias_dims,
            &bias,
            &p.output,
            output.as_mut_ptr(),
        )
    };
    common::check_status(status)
}
