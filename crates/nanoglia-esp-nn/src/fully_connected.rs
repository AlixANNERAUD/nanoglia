//! `FULLY_CONNECTED` with int8 activations, int8 weights `[output_depth, accum_depth]`
//! (per-tensor or per-output-channel, zero point 0) and int32 bias.
//!
//! Port of esp-tflite-micro's `kernels/esp_nn/fully_connected.cc` with TFLite Micro's
//! `CalculateOpDataFullyConnected`: the input is flattened to `[batches, accum_depth]`. Per-tensor
//! layers run `esp_nn_fully_connected_s8` row by row, per-channel layers
//! `esp_nn_fully_connected_per_ch_s8_batch` (a weight-stationary kernel on ESP32-S3, the
//! per-row reference kernel in a loop elsewhere). ESP-NN takes the depths as `uint16_t`.

use nanoglia_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanoglia_core::pod::Pod;
use nanoglia_core::quant;
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result, TensorDesc};
use nanoglia_esp_nn_sys as sys;

use crate::common;
use crate::fallback::{self, EspOp};
use crate::layout::{ChannelLayout, inconsistent, read_params};

/// The `FULLY_CONNECTED` kernel.
pub(crate) static KERNEL: Kernel = fallback::kernel::<FullyConnected>();

/// ESP-NN parameters, followed by the per-channel multipliers and shifts (one entry for
/// per-tensor layers).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    input_offset: i32,
    output_offset: i32,
    activation_min: i32,
    activation_max: i32,
    batches: i32,
    accum_depth: i32,
    output_depth: i32,
    /// Whether the bias input is present.
    has_bias: i32,
    /// Whether the weights are quantized per channel.
    per_channel: i32,
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

    let (_, input_zero_point) = common::activation_quant(input)?;
    let (output_scale, output_zero_point) = common::activation_quant(output)?;
    if filter.dtype != DataType::I8 || filter.dims.rank() != 2 || output.dims.rank() == 0 {
        return None;
    }
    let output_depth = filter.dims.get(0)?;
    let accum_depth = filter.dims.get(1)?;
    let in_range = |v: i32| (1..=u16::MAX as i32).contains(&v);
    if !in_range(output_depth) || !in_range(accum_depth) {
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
    let per_channel = common::weight_quant(filter, output_depth as usize, 0)?;
    if let Some(b) = bias
        && (b.dtype != DataType::I32 || common::element_count(&b.dims)? != output_depth)
    {
        return None;
    }
    let (activation_min, activation_max) =
        common::activation_range(options.activation, output_scale, output_zero_point)?;
    let params = Params {
        input_offset: -input_zero_point,
        output_offset: output_zero_point,
        activation_min,
        activation_max,
        batches,
        accum_depth,
        output_depth,
        has_bias: bias.is_some() as i32,
        per_channel: per_channel as i32,
    };
    Some(Analysis {
        params,
        input,
        filter,
        output,
    })
}

fn layout(p: &Params) -> ChannelLayout<Params> {
    let channels = if p.per_channel != 0 {
        p.output_depth as usize
    } else {
        1
    };
    ChannelLayout::new(channels)
}

/// `FULLY_CONNECTED` on ESP-NN.
pub(crate) struct FullyConnected;

impl EspOp for FullyConnected {
    const NAME: &'static str = "esp-nn/fully_connected_s8";

    fn fallback() -> &'static Kernel {
        &nanoglia_cmsis_nn::kernels::FULLY_CONNECTED_S8
    }

    fn plan(op: &OpDesc<'_, '_>) -> Option<KernelPlan> {
        let a = analyze(op)?;
        Some(KernelPlan {
            persistent: layout(&a.params).size,
            scratch: 0,
        })
    }

    fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
        let a = analyze(op).ok_or_else(|| common::unsupported(op))?;
        let layout = layout(&a.params);
        let input_scale = a.input.quant.scale();
        let output_scale = a.output.quant.scale();
        // Per-tensor: TFLite Micro's `GetQuantizedConvolutionMultipler`; per-channel: the
        // convolution formula for each channel. Both compute in double precision.
        let multipliers = (0..layout.multiplier.len()).map(|c| {
            quant::quantize_multiplier(quant::fully_connected_effective_scale(
                input_scale,
                common::weight_scale(a.filter, c),
                output_scale,
            ))
        });
        layout.write(persist, a.params, multipliers)
    }

    fn run(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
        let persist = ctx.persist;
        let p: Params = read_params(persist)?;
        let layout = layout(&p);
        let multiplier = layout.multiplier.get(persist)?;
        let shift = layout.shift.get(persist)?;
        let row_len = u16::try_from(p.accum_depth).map_err(|_| inconsistent())?;
        let out_channels = u16::try_from(p.output_depth).map_err(|_| inconsistent())?;
        if row_len == 0 || out_channels == 0 {
            return Err(inconsistent());
        }

        let input = common::input(ctx.inputs, 0);
        let filter = common::input(ctx.inputs, 1);
        common::check_len(input, p.batches.saturating_mul(p.accum_depth))?;
        common::check_len(filter, p.output_depth.saturating_mul(p.accum_depth))?;
        let bias = common::optional_i32(
            common::input(ctx.inputs, 2),
            p.has_bias != 0,
            p.output_depth,
        )?;
        let output = ctx.output(0);
        common::check_len(output, p.batches.saturating_mul(p.output_depth))?;

        if p.per_channel != 0 {
            // SAFETY: the input holds `batches` rows of `row_len` values, the output `batches`
            // rows of `out_channels`, the filter `out_channels` rows of `row_len`, and the bias
            // and the per-channel arrays `out_channels` entries.
            unsafe {
                sys::nanoglia_esp_nn_fully_connected_per_ch_s8_batch(
                    input.as_ptr().cast(),
                    p.input_offset,
                    row_len,
                    filter.as_ptr().cast(),
                    0,
                    bias,
                    output.as_mut_ptr().cast(),
                    out_channels,
                    p.output_offset,
                    shift.as_ptr(),
                    multiplier.as_ptr(),
                    p.activation_min,
                    p.activation_max,
                    p.batches,
                );
            }
            return Ok(());
        }

        let (multiplier, shift) = match (multiplier.first(), shift.first()) {
            (Some(&m), Some(&s)) => (m, s),
            _ => return Err(inconsistent()),
        };
        for (input, output) in input
            .chunks_exact(row_len as usize)
            .zip(output.chunks_exact_mut(out_channels as usize))
        {
            // SAFETY: one row of `row_len` input values and of `out_channels` outputs; the filter
            // and the bias match.
            unsafe {
                sys::nanoglia_esp_nn_fully_connected_s8(
                    input.as_ptr().cast(),
                    p.input_offset,
                    row_len,
                    filter.as_ptr().cast(),
                    0,
                    bias,
                    output.as_mut_ptr().cast(),
                    out_channels,
                    p.output_offset,
                    shift,
                    multiplier,
                    p.activation_min,
                    p.activation_max,
                );
            }
        }
        Ok(())
    }
}
