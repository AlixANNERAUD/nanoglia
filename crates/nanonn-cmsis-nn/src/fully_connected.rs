//! `FULLY_CONNECTED` with int8 activations, int8 weights `[output_depth, accum_depth]`
//! (per-tensor or per-output-channel, zero point 0) and int32 bias.
//!
//! Port of TFLite Micro's `kernels/cmsis_nn/fully_connected.cc` and of
//! `CalculateOpDataFullyConnected`: the input is flattened to `[batches, accum_depth]`; outputs of
//! rank 3 or more whose depth is a multiple of 4 run as a 1x1 convolution
//! (`arm_convolve_1x1_s8_fast`), the others through `arm_fully_connected_wrapper_s8`. Per-tensor
//! requantization parameters are broadcast to one entry per channel at prepare time instead of in
//! a scratch buffer at each evaluation.
//!
//! When CMSIS-NN needs kernel sums (Helium), they are computed once at prepare time if the weights
//! and bias are constant, else at each evaluation in the scratch buffer.
//!
//! 16x8 layers (int16 activations with zero point 0, int8 weights, int64 bias) run through
//! `arm_fully_connected_s16`, like in TensorFlow Lite Micro: per-tensor weights only (TFLite
//! Micro refuses per-channel int16 layers), requantization shifts below 8. int32 biases are
//! refused: LiteRT picks the bias type from the `quantized_bias_type` option, which nanonn does
//! not decode, and reads the bias as int64 by default.

use nanonn_cmsis_nn_sys as sys;
use nanonn_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanonn_core::pod::{Pod, cast_slice, cast_slice_mut};
use nanonn_core::quant;
use nanonn_core::{BuiltinOp, DataType, Detail, Error, OpDesc, OpOptions, Result, TensorDesc};

use crate::common::{self, nhwc};
use crate::layout::{ChannelLayout, inconsistent, read_params};

/// The `FULLY_CONNECTED` kernel.
pub(crate) static KERNEL: Kernel = Kernel {
    name: "cmsis-nn/fully_connected_s8",
    plan,
    prepare,
    eval,
};

/// The 16x8 `FULLY_CONNECTED` kernel.
pub(crate) static KERNEL_S16: Kernel = Kernel {
    name: "cmsis-nn/fully_connected_s16",
    plan,
    prepare,
    eval: eval_s16,
};

/// Persistent parameters, followed by the per-channel multipliers and shifts, then by the kernel
/// sums when they are precomputed.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    fc: sys::cmsis_nn_fc_params,
    /// Parameters of the 1x1 convolution path.
    conv: sys::cmsis_nn_conv_params,
    /// `[batches, 1, 1, accum_depth]`.
    input: sys::cmsis_nn_dims,
    /// `[accum_depth, 1, 1, output_depth]`.
    filter: sys::cmsis_nn_dims,
    /// `[batches, 1, 1, output_depth]`.
    output: sys::cmsis_nn_dims,
    /// Output depth.
    channels: i32,
    /// Whether the bias input is present.
    has_bias: i32,
    /// Whether the weights are quantized per channel.
    per_channel: i32,
    /// Whether the layer runs as a 1x1 convolution.
    conv_path: i32,
    /// Whether the kernel sums are precomputed in the persistent buffer.
    sums_persistent: i32,
    /// Whether the kernel sums are computed at each evaluation in the scratch buffer.
    sums_in_scratch: i32,
    /// Scratch bytes.
    scratch: i32,
}

// SAFETY: `#[repr(C)]` struct of `i32` fields only: no padding, any bit pattern is valid.
unsafe impl Pod for Params {}

struct Analysis<'a> {
    params: Params,
    input: &'a TensorDesc<'a>,
    filter: &'a TensorDesc<'a>,
    bias: Option<&'a TensorDesc<'a>>,
    output: &'a TensorDesc<'a>,
}

fn analyze<'a>(op: &'a OpDesc<'a, '_>) -> Option<Analysis<'a>> {
    if op.builtin()? != BuiltinOp::FULLY_CONNECTED || op.outputs.len() != 1 {
        return None;
    }
    let OpOptions::FullyConnected(options) = op.options else {
        return None;
    };
    if !options.default_weights_format || !(2..=3).contains(&op.inputs.len()) {
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
    if filter.dtype != DataType::I8 || filter.dims.rank() != 2 || output.dims.rank() == 0 {
        return None;
    }
    if int16 && (input_zero_point != 0 || output_zero_point != 0) {
        return None;
    }
    let output_depth = filter.dims.get(0)?;
    let accum_depth = filter.dims.get(1)?;
    if output_depth <= 0 || accum_depth <= 0 {
        return None;
    }
    common::element_count(&filter.dims)?;
    let output_rank = output.dims.rank();
    if output.dims.get(output_rank - 1)? != output_depth {
        return None;
    }
    let output_len = common::element_count(&output.dims)?;
    let batches = output_len / output_depth;
    if common::element_count(&input.dims)? != batches.checked_mul(accum_depth)? {
        return None;
    }
    common::weight_quant(filter, output_depth as usize, 0)?;
    let per_channel = filter.quant.scale.len() > 1;
    let bias_type = if int16 { DataType::I64 } else { DataType::I32 };
    if let Some(b) = bias
        && (b.dtype != bias_type || common::element_count(&b.dims)? != output_depth)
    {
        return None;
    }
    if int16 {
        let real = quant::fully_connected_effective_scale(
            input_scale,
            common::weight_scale(filter, 0),
            output_scale,
        );
        if per_channel || quant::quantize_multiplier(real).shift >= 8 {
            return None;
        }
    }
    let (act_min, act_max) =
        common::activation_range(options.activation, dtype, output_scale, output_zero_point)?;
    let activation = sys::cmsis_nn_activation {
        min: act_min,
        max: act_max,
    };

    let fc = sys::cmsis_nn_fc_params {
        input_offset: -input_zero_point,
        filter_offset: 0,
        output_offset: output_zero_point,
        activation,
    };
    let unit = sys::cmsis_nn_tile { w: 1, h: 1 };
    let conv = sys::cmsis_nn_conv_params {
        input_offset: -input_zero_point,
        output_offset: output_zero_point,
        stride: unit,
        padding: sys::cmsis_nn_tile { w: 0, h: 0 },
        dilation: unit,
        activation,
    };
    let input_dims = nhwc(batches, 1, 1, accum_depth);
    let filter_dims = nhwc(accum_depth, 1, 1, output_depth);
    let output_dims = nhwc(batches, 1, 1, output_depth);

    let conv_path = !int16 && output_rank > 2 && accum_depth % 4 == 0;
    let (mut scratch, mut sums_persistent, mut sums_in_scratch) = (0, false, false);
    if int16 {
        // SAFETY: pointer to a live local, only read.
        scratch = unsafe { sys::arm_fully_connected_s16_get_buffer_size(&filter_dims) };
    } else if conv_path {
        // SAFETY: pointer to a live local, only read.
        scratch = unsafe { sys::arm_convolve_1x1_s8_fast_get_buffer_size(&input_dims) };
    } else {
        // SAFETY: pointer to a live local, only read.
        let sums = unsafe { sys::arm_fully_connected_s8_get_buffer_size(&filter_dims) };
        if sums > 0 {
            if filter.is_constant() && bias.is_none_or(TensorDesc::is_constant) {
                sums_persistent = true;
            } else {
                sums_in_scratch = true;
                scratch = sums;
            }
        }
    }
    let params = Params {
        fc,
        conv,
        input: input_dims,
        filter: filter_dims,
        output: output_dims,
        channels: output_depth,
        has_bias: bias.is_some() as i32,
        per_channel: per_channel as i32,
        conv_path: conv_path as i32,
        sums_persistent: sums_persistent as i32,
        sums_in_scratch: sums_in_scratch as i32,
        scratch: common::buffer_size(scratch)?,
    };
    Some(Analysis {
        params,
        input,
        filter,
        bias,
        output,
    })
}

fn layout(p: &Params) -> ChannelLayout<Params> {
    let sums = if p.sums_persistent != 0 {
        p.channels as usize
    } else {
        0
    };
    ChannelLayout::new(p.channels as usize, sums)
}

/// The kernel running `op`, if any.
pub(crate) fn select(op: &OpDesc<'_, '_>) -> Option<&'static Kernel> {
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
    let p = a.params;
    let layout = layout(&p);
    let input_scale = a.input.quant.scale();
    let output_scale = a.output.quant.scale();
    let multipliers = (0..p.channels as usize).map(|c| {
        quant::quantize_multiplier(quant::conv_effective_scale(
            input_scale,
            common::weight_scale(a.filter, c),
            output_scale,
        ))
    });
    layout.write(persist, p, multipliers)?;

    if p.sums_persistent != 0 {
        let filter = a.filter.data.ok_or_else(|| common::unsupported(op))?;
        common::check_len(filter, p.channels * p.input.c)?;
        let bias = match a.bias {
            Some(b) => {
                let data = b.data.ok_or_else(|| common::unsupported(op))?;
                common::check_len(data, p.channels * 4)?;
                cast_slice::<i32>(data)
                    .map_err(|_| Error::invalid_argument(Detail::Misaligned { tensor: b.index }))?
                    .as_ptr()
            }
            None => core::ptr::null(),
        };
        let sums = layout.extra.get_mut(persist)?;
        // SAFETY: `sums` has `channels` entries, the filter `channels * accum_depth` bytes and
        // the bias (if any) `channels` aligned entries.
        let status = unsafe {
            sys::arm_vector_sum_s8(
                sums.as_mut_ptr(),
                p.input.c,
                p.channels,
                filter.as_ptr().cast(),
                p.fc.input_offset,
                p.fc.filter_offset,
                bias,
            )
        };
        common::check_status(status)?;
    }
    Ok(())
}

fn eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let persist = ctx.persist;
    let p: Params = read_params(persist)?;
    let layout = layout(&p);
    let multiplier = layout.multiplier.get(persist)?;
    let shift = layout.shift.get(persist)?;

    let input = common::input(ctx.inputs, 0);
    let filter = common::input(ctx.inputs, 1);
    common::check_len(input, common::dims_len(&p.input))?;
    common::check_len(filter, p.channels * p.input.c)?;
    let bias = common::optional_i32(common::input(ctx.inputs, 2), p.has_bias != 0, p.channels)?;
    let bias_dims = nhwc(1, 1, 1, p.channels);
    let mut scratch = common::scratch_context(ctx.scratch, p.scratch)?;
    let output = ctx.output(0);
    common::check_len(output, common::dims_len(&p.output))?;

    if p.conv_path != 0 {
        let quant_params = sys::cmsis_nn_per_channel_quant_params {
            // CMSIS-NN only reads the arrays despite the non-const pointers.
            multiplier: multiplier.as_ptr().cast_mut(),
            shift: shift.as_ptr().cast_mut(),
        };
        // SAFETY: the buffers match the dimensions, the bias is aligned, the scratch buffer is
        // as large as CMSIS-NN asked for and the per-channel arrays have `channels` entries.
        let status = unsafe {
            sys::arm_convolve_1x1_s8_fast(
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
        return common::check_status(status);
    }

    if p.sums_persistent != 0 {
        let sums = layout.extra.get(persist)?;
        scratch.buf = sums.as_ptr().cast_mut().cast();
    } else if p.sums_in_scratch != 0 {
        // SAFETY: the scratch buffer holds `channels` aligned `i32` (its planned size), the
        // filter `channels * accum_depth` bytes and the bias, if any, `channels` entries.
        let status = unsafe {
            sys::arm_vector_sum_s8(
                scratch.buf.cast(),
                p.input.c,
                p.channels,
                filter.as_ptr().cast(),
                p.fc.input_offset,
                p.fc.filter_offset,
                bias,
            )
        };
        common::check_status(status)?;
    }
    let quant_params = sys::cmsis_nn_quant_params {
        // CMSIS-NN only reads the arrays despite the non-const pointers.
        multiplier: multiplier.as_ptr().cast_mut(),
        shift: shift.as_ptr().cast_mut(),
        is_per_channel: p.per_channel,
    };
    // SAFETY: as above; the kernel sums, when CMSIS-NN reads them, have `channels` entries.
    let status = unsafe {
        sys::arm_fully_connected_wrapper_s8(
            &scratch,
            &p.fc,
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
    let quant_params = sys::cmsis_nn_per_tensor_quant_params {
        multiplier: *layout
            .multiplier
            .get(persist)?
            .first()
            .ok_or_else(inconsistent)?,
        shift: *layout
            .shift
            .get(persist)?
            .first()
            .ok_or_else(inconsistent)?,
    };

    let input = common::input(ctx.inputs, 0);
    let filter = common::input(ctx.inputs, 1);
    common::check_len(input, common::dims_len(&p.input).saturating_mul(2))?;
    let input = cast_slice::<i16>(input)?;
    common::check_len(filter, p.channels * p.input.c)?;
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

    // SAFETY: the buffers match the dimensions and are aligned for their types, and the scratch
    // buffer is as large as CMSIS-NN asked for.
    let status = unsafe {
        sys::arm_fully_connected_s16(
            &scratch,
            &p.fc,
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
