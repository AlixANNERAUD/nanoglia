//! `TRANSPOSE_CONV` with int8 activations, int8 weights (per-tensor or per-output-channel) and an
//! optional int32 bias, through `arm_transpose_conv_wrapper_s8`.
//!
//! Port of TFLite Micro's `kernels/cmsis_nn/transpose_conv.cc` (Prepare, CalculateOpData and
//! EvalQuantizedPerChannel) with the checks of LiteRT's `transpose_conv.cc`. Unlike TFLite Micro,
//! which passes `kTfLiteActNone` to `PopulateConvolutionQuantizationParams`, the fused activation
//! of the options is applied, as LiteRT's reference kernel does.
//!
//! CMSIS-NN v8.0.0 runs strides up to 2 with more than 16 input channels as a convolution of the
//! upsampled input with the reversed filter, which computes every output size TFLite accepts.
//! Its other path, `arm_transpose_conv_s8`, keeps a rolling buffer of output rows and only gives
//! the reference result when:
//!
//! - the rows it writes while reading the input, `in_h * stride_h - pad_h`, are at most the
//!   output height, and at least zero (it would write past the output, or misplace its rows);
//! - the rows it writes after the input, `max(filter_h - stride_h, 0)` more, reach the end of the
//!   output (it would leave the last rows unwritten);
//! - the output columns, after `pad_w` skipped ones, fit in the `(in_w - 1) * stride_w +
//!   max(filter_w, stride_w)` columns of its buffer (it would read past them).
//!
//! The Rust backend runs the configurations refused here. `arm_transpose_conv_s8_get_buffer_size`
//! sizes the rows of that buffer with `max(filter_w, stride_h)` where the kernel uses
//! `max(filter_w, stride_w)`: the scratch buffer is the larger of the two.

use nanoglia_cmsis_nn_sys as sys;
use nanoglia_core::backend::{EvalCtx, FnKernel, KernelPlan};
use nanoglia_core::pod::Pod;
use nanoglia_core::quant::{self, Multiplier};
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Padding, Result, TensorDesc};

use crate::common::{self, nhwc};
use crate::layout::{ChannelLayout, read_params};

/// The `TRANSPOSE_CONV` kernel.
pub static KERNEL: FnKernel = FnKernel {
    name: "cmsis-nn/transpose_conv_s8",
    plan,
    prepare,
    eval,
};

/// CMSIS-NN's `REVERSE_TCOL_EFFICIENT_THRESHOLD`: above this many input channels (and for strides
/// up to 2), the wrapper runs a convolution of the upsampled input.
const REVERSE_THRESHOLD: i32 = 16;

/// Persistent parameters, followed by the per-channel multipliers and shifts.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    tconv: sys::cmsis_nn_transpose_conv_params,
    input: sys::cmsis_nn_dims,
    filter: sys::cmsis_nn_dims,
    output: sys::cmsis_nn_dims,
    /// Number of output channels.
    channels: i32,
    /// Whether the bias input is present.
    has_bias: i32,
    /// Bytes of the `ctx` buffer, at the start of the scratch buffer.
    scratch: i32,
    /// Bytes of the reversed filter, after the `ctx` buffer (rounded up to 4 bytes).
    reverse: i32,
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

/// TFLite `ComputeOutSize` and `ComputePaddingWithOffset` of the forward convolution, from the
/// output of the transpose convolution: the input size it implies, the padding before and the
/// padding offset (1 when the total padding is odd).
fn forward(padding: Padding, out_size: i32, filter: i32, stride: i32) -> Option<(i32, i32, i32)> {
    let (out, filter, stride) = (i64::from(out_size), i64::from(filter), i64::from(stride));
    let in_size = match padding {
        Padding::Same => (out + stride - 1) / stride,
        Padding::Valid => (out + stride - filter) / stride,
    };
    let total = ((in_size - 1) * stride + filter - out).max(0);
    let narrow = |v: i64| i16::try_from(v).ok().map(i32::from);
    Some((
        i32::try_from(in_size).ok()?,
        narrow(total / 2)?,
        narrow(total % 2)?,
    ))
}

fn analyze<'a>(op: &'a OpDesc<'a, '_>) -> Option<Analysis<'a>> {
    if op.builtin()? != BuiltinOp::TRANSPOSE_CONV || op.outputs.len() != 1 {
        return None;
    }
    let OpOptions::TransposeConv(options) = op.options else {
        return None;
    };
    if !(3..=4).contains(&op.inputs.len()) {
        return None;
    }
    // Inputs: output shape, weights (OHWI), input (NHWC), optional bias.
    let shape = op.input(0)?;
    let filter = op.input(1)?;
    let input = op.input(2)?;
    let bias = op.input(3);
    let output = op.output(0)?;

    let dtype = DataType::I8;
    let (_, input_zero_point) = common::activation_quant(input, dtype)?;
    let (output_scale, output_zero_point) = common::activation_quant(output, dtype)?;
    if filter.dtype != DataType::I8 {
        return None;
    }
    let [batches, in_h, in_w, in_c] = common::dims4(input)?;
    let [out_c, filter_h, filter_w, filter_c] = common::dims4(filter)?;
    let [out_n, out_h, out_w, out_c2] = common::dims4(output)?;
    if out_n != batches || out_c2 != out_c || filter_c != in_c {
        return None;
    }
    // The output shape: a constant int32 vector of four elements, which the output must have.
    if shape.dtype != DataType::I32 || shape.dims.as_slice() != [4] {
        return None;
    }
    let shape_data = shape.data?;
    for (i, &d) in [batches, out_h, out_w, out_c].iter().enumerate() {
        if nanoglia_core::le::read::<i32>(shape_data, 4 * i)? != d {
            return None;
        }
    }
    common::weight_quant(filter, out_c as usize, 0)?;
    let has_bias = match bias {
        Some(b) => {
            if b.dtype != DataType::I32
                || common::element_count(&b.dims)? != out_c
                || b.quant.zero_point() != 0
            {
                return None;
            }
            true
        }
        None => false,
    };
    let stride_range = 1..=i32::from(i16::MAX);
    let (stride_h, stride_w) = (options.stride_h, options.stride_w);
    if !(stride_range.contains(&stride_h) && stride_range.contains(&stride_w)) {
        return None;
    }
    let (computed_h, pad_h, offset_h) = forward(options.padding, out_h, filter_h, stride_h)?;
    let (computed_w, pad_w, offset_w) = forward(options.padding, out_w, filter_w, stride_w)?;
    if (computed_h, computed_w) != (in_h, in_w) {
        return None;
    }
    let (act_min, act_max) =
        common::activation_range(options.activation, dtype, output_scale, output_zero_point)?;

    // The path `arm_transpose_conv_wrapper_s8` takes, and the configurations it computes exactly.
    let (in_h64, in_w64) = (i64::from(in_h), i64::from(in_w));
    let (sh, sw) = (i64::from(stride_h), i64::from(stride_w));
    let (fh, fw) = (i64::from(filter_h), i64::from(filter_w));
    let (ph, pw) = (i64::from(pad_h), i64::from(pad_w));
    let (oh, ow) = (i64::from(out_h), i64::from(out_w));
    let reverse = stride_h <= 2 && stride_w <= 2 && in_c > REVERSE_THRESHOLD;
    if reverse {
        // `arm_convolve_s8` keeps the dimensions of the upsampled input and the padding of the
        // reversed filter in `uint16_t` variables.
        let max = i64::from(u16::MAX);
        if in_h64 * sh > max || in_w64 * sw > max || fh - 1 - ph < 0 || fw - 1 - pw < 0 {
            return None;
        }
    } else {
        let flushed = in_h64 * sh - ph;
        let rows_ok = flushed >= 0 && flushed <= oh && in_h64 * sh + (fh - sh).max(0) >= ph + oh;
        let columns_ok = pw + ow <= (in_w64 - 1) * sw + fw.max(sw);
        if !(rows_ok && columns_ok) {
            return None;
        }
    }

    let tconv = sys::cmsis_nn_transpose_conv_params {
        input_offset: -input_zero_point,
        output_offset: output_zero_point,
        stride: sys::cmsis_nn_tile {
            w: stride_w,
            h: stride_h,
        },
        padding: sys::cmsis_nn_tile { w: pad_w, h: pad_h },
        // Like TFLite Micro: `width_offset + width`.
        padding_offsets: sys::cmsis_nn_tile {
            w: offset_w + pad_w,
            h: offset_h + pad_h,
        },
        dilation: sys::cmsis_nn_tile { w: 1, h: 1 },
        activation: sys::cmsis_nn_activation {
            min: act_min,
            max: act_max,
        },
    };
    let input_dims = nhwc(batches, in_h, in_w, in_c);
    let filter_dims = nhwc(out_c, filter_h, filter_w, in_c);
    let output_dims = nhwc(batches, out_h, out_w, out_c);
    // SAFETY: the pointers are to live locals; the functions only read them.
    let (scratch, reverse_size) = unsafe {
        (
            sys::arm_transpose_conv_s8_get_buffer_size(
                &tconv,
                &input_dims,
                &filter_dims,
                &output_dims,
            ),
            sys::arm_transpose_conv_s8_get_reverse_conv_buffer_size(
                &tconv,
                &input_dims,
                &filter_dims,
            ),
        )
    };
    let mut scratch = i64::from(common::buffer_size(scratch)?);
    if !reverse {
        // The rolling buffer `arm_transpose_conv_s8` actually uses (see the module comment).
        let row = ((in_w64 - 1) * sw + fw.max(sw)) * i64::from(out_c);
        scratch = scratch.max(row * fh.max(sh) * 4);
    }
    let reverse_size = i64::from(common::buffer_size(reverse_size)?);
    // The reversed filter follows the `ctx` buffer, which stays 4-byte aligned.
    let scratch = (scratch + 3) / 4 * 4;
    i32::try_from(scratch + reverse_size).ok()?;
    let params = Params {
        tconv,
        input: input_dims,
        filter: filter_dims,
        output: output_dims,
        channels: out_c,
        has_bias: has_bias as i32,
        scratch: i32::try_from(scratch).ok()?,
        reverse: i32::try_from(reverse_size).ok()?,
    };
    Some(Analysis {
        params,
        input,
        filter,
        output,
    })
}

fn layout(p: &Params) -> ChannelLayout<Params> {
    ChannelLayout::new(p.channels as usize, 0)
}

/// Whether this backend runs `op`.
pub(crate) fn supports(op: &OpDesc<'_, '_>) -> bool {
    analyze(op).is_some()
}

fn plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    let a = analyze(op).ok_or_else(|| common::unsupported(op))?;
    Ok(KernelPlan {
        persistent: layout(&a.params).size,
        scratch: (a.params.scratch + a.params.reverse) as usize,
    })
}

fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let a = analyze(op).ok_or_else(|| common::unsupported(op))?;
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
    layout(&a.params).write(persist, a.params, multipliers)
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

    let filter = common::input(ctx.inputs, 1);
    let input = common::input(ctx.inputs, 2);
    common::check_len(input, common::dims_len(&p.input))?;
    common::check_len(filter, common::dims_len(&p.filter))?;
    let bias = common::optional_i32(common::input(ctx.inputs, 3), p.has_bias != 0, p.channels)?;
    let bias_dims = nhwc(1, 1, 1, p.channels);

    // The scratch buffer: the `ctx` buffer, then the reversed filter.
    let total = (p.scratch as usize).saturating_add(p.reverse as usize);
    if ctx.scratch.len() < total {
        return Err(nanoglia_core::Error::invalid_argument(
            nanoglia_core::Detail::SizeMismatch {
                expected: total as u32,
                actual: u32::try_from(ctx.scratch.len()).unwrap_or(u32::MAX),
            },
        ));
    }
    let (head, tail) = ctx.scratch.split_at_mut(p.scratch as usize);
    let scratch = common::scratch_context(head, p.scratch)?;
    let reverse = if p.reverse > 0 {
        sys::cmsis_nn_context {
            buf: tail.as_mut_ptr().cast(),
            size: p.reverse,
        }
    } else {
        sys::cmsis_nn_context {
            buf: core::ptr::null_mut(),
            size: 0,
        }
    };
    let output = ctx.output(0);
    common::check_len(output, common::dims_len(&p.output))?;

    // SAFETY: every buffer was checked against the sizes CMSIS-NN derives from the dimensions,
    // the bias is aligned, both scratch buffers are as large as CMSIS-NN asked for (the `ctx`
    // one 4-byte aligned), the per-channel arrays have `output.c` entries, and `select` only
    // kept configurations that CMSIS-NN computes inside these buffers.
    let status = unsafe {
        sys::arm_transpose_conv_wrapper_s8(
            &scratch,
            &reverse,
            &p.tconv,
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
