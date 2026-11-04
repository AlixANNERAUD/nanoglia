//! `TRANSPOSE_CONV` (TFLite `transpose_conv.cc`, the reference kernels its reference registration
//! runs: `reference_ops::TransposeConv` on `f32`, `reference_integer_ops::TransposeConv` on int8
//! with per-channel requantization), with a constant output shape.
//!
//! Both scatter each input value into the output region it influences, in the reference loop
//! order, then add the bias, requantize (int8) and clamp. The int8 kernel accumulates in a scratch
//! buffer of one `i32` per output value; the CMSIS-NN backend runs most int8 transpose
//! convolutions, and this kernel the configurations CMSIS-NN cannot compute exactly.

use crate::math::mul_add;
use crate::util::{
    check_aligned, clamp_f32, compute_out_size, compute_padding, constant, dims4, elements,
    float_activation, in_slice, input, out_slice, output, per_tensor, pod_struct,
    quantized_activation, shape_mismatch, unsupported, unsupported_type,
};
use nanoglia_core::backend::{EvalCtx, FnKernel, KernelPlan};
use nanoglia_core::pod::{self, Layout, Region};
use nanoglia_core::quant::{self, Multiplier};
use nanoglia_core::{DataType, OpDesc, OpOptions, Result, TensorDesc};

pod_struct! {
    /// Geometry, offsets and activation range of a transpose convolution.
    pub(crate) struct TransposeConvParams {
        batches: i32,
        in_h: i32,
        in_w: i32,
        in_c: i32,
        filter_h: i32,
        filter_w: i32,
        out_h: i32,
        out_w: i32,
        out_c: i32,
        stride_h: i32,
        stride_w: i32,
        pad_h: i32,
        pad_w: i32,
        has_bias: u32,
        /// Negated input zero point (int8).
        input_offset: i32,
        /// Output zero point (int8).
        output_offset: i32,
        /// Quantized activation range (int8).
        min_q: i32,
        max_q: i32,
        /// Float activation range (f32).
        min_f: f32,
        max_f: f32,
    }
}

impl TransposeConvParams {
    fn input_len(&self) -> usize {
        (self.batches * self.in_h * self.in_w * self.in_c) as usize
    }

    fn filter_len(&self) -> usize {
        (self.out_c * self.filter_h * self.filter_w * self.in_c) as usize
    }

    fn output_len(&self) -> usize {
        (self.batches * self.out_h * self.out_w * self.out_c) as usize
    }
}

/// The quantized tensors of a transpose convolution: input, weights, output.
struct Tensors<'o, 'a> {
    input: &'o TensorDesc<'a>,
    weights: &'o TensorDesc<'a>,
    output: &'o TensorDesc<'a>,
}

/// TFLite `Prepare` and `ComputeAndValidatePadding`, for the element type `dtype` of the input,
/// the weights and the output.
fn analyze<'o, 'a>(
    op: &'o OpDesc<'a, '_>,
    dtype: DataType,
) -> Result<(TransposeConvParams, Tensors<'o, 'a>)> {
    let OpOptions::TransposeConv(o) = op.options else {
        return Err(unsupported(op));
    };
    if !(3..=4).contains(&op.inputs.len()) || op.outputs.len() != 1 {
        return Err(unsupported(op));
    }
    // Inputs: output shape, weights (OHWI), input (NHWC), optional bias.
    let (shape, weights, data, out) = (input(op, 0)?, input(op, 1)?, input(op, 2)?, output(op, 0)?);
    let bias = op.input(3);
    for t in [weights, data, out] {
        if t.dtype != dtype {
            return Err(unsupported_type(t));
        }
    }
    // The output shape: a constant int32 vector of four elements, which the output must have.
    if shape.dtype != DataType::I32 || shape.dims.as_slice() != [4] {
        return Err(unsupported(op));
    }
    let shape_data = constant(op, shape)?;
    let mut expected = [0i32; 4];
    for (i, d) in expected.iter_mut().enumerate() {
        *d = nanoglia_core::le::read::<i32>(shape_data, 4 * i).ok_or_else(|| unsupported(op))?;
    }
    if out.dims.as_slice() != expected {
        return Err(shape_mismatch(out));
    }
    let [batches, in_h, in_w, in_c] = dims4(data)?;
    let [out_c, filter_h, filter_w, filter_c] = dims4(weights)?;
    let [out_n, out_h, out_w, out_c2] = dims4(out)?;
    let stride = 1..=i32::from(i16::MAX);
    if !stride.contains(&o.stride_h) || !stride.contains(&o.stride_w) {
        return Err(unsupported(op));
    }
    if [in_h, in_w, in_c, out_c, filter_h, filter_w, out_h, out_w]
        .iter()
        .any(|&d| d <= 0)
    {
        return Err(shape_mismatch(out));
    }
    if filter_c != in_c {
        return Err(shape_mismatch(weights));
    }
    if out_n != batches || out_c2 != out_c {
        return Err(shape_mismatch(out));
    }
    if let Some(b) = bias {
        let bias_type = if dtype == DataType::F32 {
            DataType::F32
        } else {
            DataType::I32
        };
        if b.dtype != bias_type {
            return Err(unsupported_type(b));
        }
        check_aligned(b, 4)?;
        if elements(b)? != out_c as usize {
            return Err(shape_mismatch(b));
        }
    }
    // The input is what the convolution this operator is the transpose of would compute from
    // the output; the padding is that convolution's.
    let fits_i16 = |v: i64| (i64::from(i16::MIN)..=i64::from(i16::MAX)).contains(&v);
    let mut pads = [0i32; 2];
    for (pad, (out_size, filter, stride, in_size)) in pads.iter_mut().zip([
        (out_h, filter_h, o.stride_h, in_h),
        (out_w, filter_w, o.stride_w, in_w),
    ]) {
        let computed = compute_out_size(o.padding, out_size, filter, stride, 1);
        if computed != i64::from(in_size) {
            return Err(shape_mismatch(data));
        }
        let total =
            (i64::from(in_size) - 1) * i64::from(stride) + i64::from(filter) - i64::from(out_size);
        if !fits_i16(total.max(0) / 2) {
            return Err(shape_mismatch(data));
        }
        *pad = compute_padding(stride, 1, out_size, filter, in_size);
    }
    let mut p = TransposeConvParams {
        batches,
        in_h,
        in_w,
        in_c,
        filter_h,
        filter_w,
        out_h,
        out_w,
        out_c,
        stride_h: o.stride_h,
        stride_w: o.stride_w,
        pad_h: pads[0],
        pad_w: pads[1],
        has_bias: u32::from(bias.is_some()),
        input_offset: 0,
        output_offset: 0,
        min_q: 0,
        max_q: 0,
        min_f: 0.0,
        max_f: 0.0,
    };
    if dtype == DataType::F32 {
        check_aligned(weights, 4)?;
        (p.min_f, p.max_f) = float_activation(op, o.activation)?;
    } else {
        let (_, input_zero_point) = per_tensor(data)?;
        let output_quant = per_tensor(out)?;
        p.input_offset = -input_zero_point;
        p.output_offset = output_quant.1;
        (p.min_q, p.max_q) = quantized_activation(op, o.activation, out, output_quant)?;
        // Symmetric weights, per tensor or per output channel.
        let q = &weights.quant;
        let scales = q.scale.len();
        if !(scales == 1 || (scales == out_c as usize && q.axis == 0))
            || !q.zero_point.iter().all(|z| z == 0)
            || !q.scale.iter().all(|s| s.is_finite() && s >= 0.0)
        {
            return Err(unsupported_type(weights));
        }
        if let Some(b) = bias
            && b.quant.zero_point() != 0
        {
            return Err(unsupported_type(b));
        }
    }
    Ok((
        p,
        Tensors {
            input: data,
            weights,
            output: out,
        },
    ))
}

// --------------------------------------------------------------------------------------------
// f32
// --------------------------------------------------------------------------------------------

fn f32_plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    analyze(op, DataType::F32)?;
    Ok(KernelPlan {
        persistent: core::mem::size_of::<TransposeConvParams>(),
        scratch: 0,
    })
}

fn f32_prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    pod::write(persist, analyze(op, DataType::F32)?.0)
}

/// Calls `f(output_index, input_index, filter_index)` for every product of the reference
/// scatter loop, in its order.
#[inline(always)]
fn scatter(p: &TransposeConvParams, mut f: impl FnMut(usize, usize, usize)) {
    for b in 0..p.batches {
        for in_y in 0..p.in_h {
            for in_x in 0..p.in_w {
                for ic in 0..p.in_c {
                    let out_x_origin = in_x * p.stride_w - p.pad_w;
                    let out_y_origin = in_y * p.stride_h - p.pad_h;
                    let i = (((b * p.in_h + in_y) * p.in_w + in_x) * p.in_c + ic) as usize;
                    for fy in 0..p.filter_h {
                        let out_y = out_y_origin + fy;
                        if out_y < 0 || out_y >= p.out_h {
                            continue;
                        }
                        for fx in 0..p.filter_w {
                            let out_x = out_x_origin + fx;
                            if out_x < 0 || out_x >= p.out_w {
                                continue;
                            }
                            let o = (((b * p.out_h + out_y) * p.out_w + out_x) * p.out_c) as usize;
                            for oc in 0..p.out_c {
                                let w = (((oc * p.filter_h + fy) * p.filter_w + fx) * p.in_c + ic)
                                    as usize;
                                f(o + oc as usize, i, w);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn f32_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: TransposeConvParams = *pod::read(ctx.persist)?;
    let input = in_slice::<f32>(ctx.inputs, 2, p.input_len())?;
    let filter = in_slice::<f32>(ctx.inputs, 1, p.filter_len())?;
    let bias = if p.has_bias != 0 {
        Some(in_slice::<f32>(ctx.inputs, 3, p.out_c as usize)?)
    } else {
        None
    };
    let output = out_slice::<f32>(ctx.outputs, 0, p.output_len())?;
    output.fill(0.0);
    // `output += input * filter`: Clang fuses it into a multiply-add.
    scatter(&p, |o, i, w| {
        output[o] = mul_add(input[i], filter[w], output[o])
    });
    for (n, value) in output.iter_mut().enumerate() {
        let mut acc = *value;
        if let Some(bias) = bias {
            acc += bias[n % p.out_c as usize];
        }
        *value = clamp_f32(acc, p.min_f, p.max_f);
    }
    Ok(())
}

/// `TRANSPOSE_CONV` on `f32`.
pub(crate) static TRANSPOSE_CONV_F32: FnKernel = FnKernel {
    name: "rust/transpose_conv_f32",
    plan: f32_plan,
    prepare: f32_prepare,
    eval: f32_eval,
};

// --------------------------------------------------------------------------------------------
// int8
// --------------------------------------------------------------------------------------------

/// The persistent buffer of an int8 transpose convolution with `channels` output channels.
fn s8_layout(channels: usize) -> (Region<TransposeConvParams>, Region<Multiplier>, usize) {
    let mut layout = Layout::new();
    let params = layout.push::<TransposeConvParams>(1);
    let multipliers = layout.push::<Multiplier>(channels);
    (params, multipliers, layout.size())
}

fn s8_plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    let (p, _) = analyze(op, DataType::I8)?;
    Ok(KernelPlan {
        persistent: s8_layout(p.out_c as usize).2,
        scratch: p.output_len() * 4,
    })
}

fn s8_prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let (p, t) = analyze(op, DataType::I8)?;
    let (header, multipliers, _) = s8_layout(p.out_c as usize);
    header.get_mut(persist)?[0] = p;
    let (input_scale, output_scale) = (t.input.quant.scale(), t.output.quant.scale());
    let scales = t.weights.quant.scale;
    for (c, m) in multipliers.get_mut(persist)?.iter_mut().enumerate() {
        let filter_scale = scales
            .get(if scales.len() == 1 { 0 } else { c })
            .unwrap_or(0.0);
        *m = quant::quantize_multiplier(quant::conv_effective_scale(
            input_scale,
            filter_scale,
            output_scale,
        ));
    }
    Ok(())
}

fn s8_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let header = s8_layout(0).0;
    let p = header.get(ctx.persist)?[0];
    let multipliers = s8_layout(p.out_c as usize).1.get(ctx.persist)?;
    let input = in_slice::<i8>(ctx.inputs, 2, p.input_len())?;
    let filter = in_slice::<i8>(ctx.inputs, 1, p.filter_len())?;
    let bias = if p.has_bias != 0 {
        Some(in_slice::<i32>(ctx.inputs, 3, p.out_c as usize)?)
    } else {
        None
    };
    let n = p.output_len();
    let scratch = ctx
        .scratch
        .get_mut(..n * 4)
        .ok_or_else(|| crate::util::size_mismatch(n * 4, 0))?;
    let acc = pod::cast_slice_mut::<i32>(scratch)?;
    acc.fill(0);
    scatter(&p, |o, i, w| {
        let product = (i32::from(input[i]) + p.input_offset) * i32::from(filter[w]);
        acc[o] = acc[o].wrapping_add(product);
    });
    let output = out_slice::<i8>(ctx.outputs, 0, n)?;
    let channels = p.out_c as usize;
    for (k, (y, &a)) in output.iter_mut().zip(acc.iter()).enumerate() {
        let c = k % channels;
        let mut value = a;
        if let Some(bias) = bias {
            value = value.wrapping_add(bias[c]);
        }
        let value = quant::multiply_by_quantized_multiplier(value, multipliers[c])
            .wrapping_add(p.output_offset);
        *y = value.max(p.min_q).min(p.max_q) as i8;
    }
    Ok(())
}

/// `TRANSPOSE_CONV` with int8 activations and weights (per tensor or per output channel) and an
/// int32 bias.
pub(crate) static TRANSPOSE_CONV_S8: FnKernel = FnKernel {
    name: "rust/transpose_conv_s8",
    plan: s8_plan,
    prepare: s8_prepare,
    eval: s8_eval,
};
