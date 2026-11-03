//! `CONV_2D` with int8 activations, int8 weights (per-tensor or per-output-channel) and int32
//! bias, through `arm_convolve_wrapper_s8`.
//!
//! Port of TFLite Micro's `kernels/cmsis_nn/conv.cc` (Prepare and EvalQuantizedPerChannel) and
//! of `CalculateOpDataConv`/`PopulateConvolutionQuantizationParams`. Grouped convolutions (input
//! channels a multiple of the filter's) are supported like in the TFLite reference kernel.
//!
//! CMSIS-NN v8.0.0 advances the bias pointer of grouped convolutions even when it is null (and then
//! reads through it), so a grouped convolution without bias gets an all-zero bias in its
//! persistent buffer.

use nanonn_cmsis_nn_sys as sys;
use nanonn_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanonn_core::pod::Pod;
use nanonn_core::quant::{self, Multiplier};
use nanonn_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result, TensorDesc};

use crate::common::{self, nhwc};
use crate::layout::{ChannelLayout, read_params};

/// The `CONV_2D` kernel.
pub(crate) static KERNEL: Kernel = Kernel {
    name: "cmsis-nn/conv2d_s8",
    plan,
    prepare,
    eval,
};

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

    let (_, input_zero_point) = common::activation_quant(input, DataType::I8)?;
    let (output_scale, output_zero_point) = common::activation_quant(output, DataType::I8)?;
    if filter.dtype != DataType::I8 {
        return None;
    }
    let [batches, in_h, in_w, in_c] = common::dims4(input)?;
    let [out_c, filter_h, filter_w, filter_c] = common::dims4(filter)?;
    let [out_n, out_h, out_w, out_c2] = common::dims4(output)?;
    if out_n != batches || out_c2 != out_c || in_c % filter_c != 0 {
        return None;
    }
    let groups = in_c / filter_c;
    if out_c % groups != 0 {
        return None;
    }
    common::weight_quant(filter, out_c as usize, 0)?;
    let has_bias = match bias {
        Some(b) => {
            if b.dtype != DataType::I32 || common::element_count(&b.dims)? != out_c {
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
    let (act_min, act_max) = common::activation_range(
        options.activation,
        DataType::I8,
        output_scale,
        output_zero_point,
    )?;

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
    // SAFETY: the pointers are to live locals; the function only reads them.
    let scratch = unsafe {
        sys::arm_convolve_wrapper_s8_get_buffer_size(&conv, &input_dims, &filter_dims, &output_dims)
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

/// Whether this kernel runs `op`.
pub(crate) fn supports(op: &OpDesc<'_, '_>) -> bool {
    analyze(op).is_some()
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
