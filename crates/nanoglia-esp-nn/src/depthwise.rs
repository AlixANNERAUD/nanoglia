//! `DEPTHWISE_CONV_2D` with int8 activations, int8 weights (per-tensor or per-output-channel) and
//! int32 bias, through `esp_nn_depthwise_conv_s8`, one batch at a time.
//!
//! Port of esp-tflite-micro's `kernels/esp_nn/depthwise_conv.cc` with TFLite Micro's
//! `CalculateOpDataDepthwiseConv`. The depth multiplier is derived from the shapes (output
//! channels / input channels) like LiteRT does. ESP-NN ignores dilations: dilated convolutions
//! are left to the next backend.

use nanoglia_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanoglia_core::pod::Pod;
use nanoglia_core::quant;
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result, TensorDesc};
use nanoglia_esp_nn_sys as sys;

use crate::common::{self, dims};
use crate::fallback::{self, EspOp};
use crate::layout::{ChannelLayout, read_params};

/// The `DEPTHWISE_CONV_2D` kernel.
pub(crate) static KERNEL: Kernel = fallback::kernel::<Depthwise>();

/// ESP-NN parameters, followed by the per-channel multipliers and shifts.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    conv: sys::dw_conv_params_t,
    /// One batch of the input.
    input: sys::data_dims_t,
    filter: sys::data_dims_t,
    /// One batch of the output.
    output: sys::data_dims_t,
    batches: i32,
    /// Number of output channels.
    channels: i32,
    /// Elements of the filter tensor.
    filter_len: i32,
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

    let (_, input_zero_point) = common::activation_quant(input)?;
    let (output_scale, output_zero_point) = common::activation_quant(output)?;
    if filter.dtype != DataType::I8 {
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
    let has_bias = match bias {
        Some(b) => {
            if b.dtype != DataType::I32 || common::element_count(&b.dims)? != out_c {
                return None;
            }
            true
        }
        None => false,
    };
    // ESP-NN does not read the dilations.
    if options.dilation_h != 1 || options.dilation_w != 1 {
        return None;
    }
    let (stride_h, stride_w) = (options.stride_h, options.stride_w);
    if stride_h < 1 || stride_w < 1 || !common::fits_u16(&[stride_h, stride_w]) {
        return None;
    }
    let (computed_h, pad_h) =
        common::output_size_and_padding(options.padding, in_h, filter_h, stride_h, 1)?;
    let (computed_w, pad_w) =
        common::output_size_and_padding(options.padding, in_w, filter_w, stride_w, 1)?;
    if (computed_h, computed_w) != (out_h, out_w) || !common::fits_u16(&[pad_h, pad_w]) {
        return None;
    }
    let (act_min, act_max) =
        common::activation_range(options.activation, output_scale, output_zero_point)?;

    let conv = sys::dw_conv_params_t {
        in_offset: -input_zero_point,
        out_offset: output_zero_point,
        ch_mult: depth_multiplier,
        stride: sys::data_2d_t {
            width: stride_w,
            height: stride_h,
        },
        padding: sys::data_2d_t {
            width: pad_w,
            height: pad_h,
        },
        dilation: sys::data_2d_t {
            width: 0,
            height: 0,
        },
        activation: sys::act_params_t {
            min: act_min,
            max: act_max,
        },
    };
    let input_dims = dims(in_w, in_h, in_c, 1);
    let filter_dims = dims(filter_w, filter_h, out_c, 0);
    let output_dims = dims(out_w, out_h, out_c, 1);
    // SAFETY: the pointers are to live locals; the function only reads them.
    let scratch = unsafe {
        sys::nanoglia_esp_nn_get_depthwise_conv_scratch_size(
            &input_dims,
            &filter_dims,
            &output_dims,
            &conv,
        )
    };
    common::buffer_size(scratch)?;
    let params = Params {
        conv,
        input: input_dims,
        filter: filter_dims,
        output: output_dims,
        batches,
        channels: out_c,
        filter_len: common::element_count(&filter.dims)?,
        has_bias: has_bias as i32,
        scratch,
    };
    Some(Analysis {
        params,
        input,
        filter,
        output,
    })
}

fn layout(p: &Params) -> ChannelLayout<Params> {
    ChannelLayout::new(p.channels as usize)
}

/// `DEPTHWISE_CONV_2D` on ESP-NN.
pub(crate) struct Depthwise;

impl EspOp for Depthwise {
    const NAME: &'static str = "esp-nn/depthwise_conv2d_s8";

    fn fallback() -> &'static Kernel {
        &nanoglia_cmsis_nn::kernels::DEPTHWISE_CONV_2D_S8
    }

    fn plan(op: &OpDesc<'_, '_>) -> Option<KernelPlan> {
        let a = analyze(op)?;
        Some(KernelPlan {
            persistent: layout(&a.params).size,
            scratch: a.params.scratch as usize,
        })
    }

    fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
        let a = analyze(op).ok_or_else(|| common::unsupported(op))?;
        let input_scale = a.input.quant.scale();
        let output_scale = a.output.quant.scale();
        let multipliers = (0..a.params.channels as usize).map(|c| {
            quant::quantize_multiplier(quant::conv_effective_scale(
                input_scale,
                common::weight_scale(a.filter, c),
                output_scale,
            ))
        });
        layout(&a.params).write(persist, a.params, multipliers)
    }

    fn run(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
        let persist = ctx.persist;
        let p: Params = read_params(persist)?;
        let layout = layout(&p);
        let quant_data = sys::quant_data_t {
            // ESP-NN only reads the arrays despite the non-const pointers.
            shift: layout.shift.get(persist)?.as_ptr().cast_mut(),
            mult: layout.multiplier.get(persist)?.as_ptr().cast_mut(),
        };

        let (input_len, output_len) = common::batch_lens(&p.input, &p.output)?;
        let input = common::input(ctx.inputs, 0);
        let filter = common::input(ctx.inputs, 1);
        common::check_len(input, (input_len as i32).saturating_mul(p.batches))?;
        common::check_len(filter, p.filter_len)?;
        let bias = common::optional_i32(common::input(ctx.inputs, 2), p.has_bias != 0, p.channels)?;
        let scratch = common::scratch_ptr(ctx.scratch, p.scratch)?;
        let output = ctx.output(0);
        common::check_len(output, (output_len as i32).saturating_mul(p.batches))?;

        // SAFETY: the caller holds the ESP-NN lock, so nothing else uses the global scratch
        // pointer; the scratch buffer is as large as ESP-NN asked for and aligned. Each batch of
        // the input and the output has the size ESP-NN derives from the dimensions, the filter
        // and the bias match the output channels, and so do the per-channel arrays.
        unsafe {
            sys::nanoglia_esp_nn_set_depthwise_conv_scratch_buf(scratch);
            for (input, output) in input
                .chunks_exact(input_len)
                .zip(output.chunks_exact_mut(output_len))
            {
                sys::nanoglia_esp_nn_depthwise_conv_s8(
                    &p.input,
                    input.as_ptr().cast(),
                    &p.filter,
                    filter.as_ptr().cast(),
                    bias,
                    &p.output,
                    output.as_mut_ptr().cast(),
                    &p.conv,
                    &quant_data,
                );
            }
            sys::nanoglia_esp_nn_set_depthwise_conv_scratch_buf(core::ptr::null_mut());
        }
        Ok(())
    }
}
