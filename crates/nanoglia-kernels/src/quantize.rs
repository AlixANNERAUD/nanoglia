//! `QUANTIZE` and `DEQUANTIZE` (TFLite `quantize.cc` and `dequantize.cc`, reference kernels).

use crate::util::{
    expect_arity, expect_type, fixed_kernel, in_slice, input, int_range, out_slice, output,
    per_tensor, pod_struct, same_elements, type_code, unsupported, unsupported_type,
};
use nanoglia_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanoglia_core::pod::{self, Layout, Region};
use nanoglia_core::quant::{
    Multiplier, multiply_by_quantized_multiplier, quantize_multiplier, round_f32,
};
use nanoglia_core::{DataType, OpDesc, Result};

pod_struct! {
    /// Parameters of `QUANTIZE` from `f32` (TFLite `AffineQuantize`).
    pub(crate) struct QuantizeParams {
        len: u32,
        output_type: u32,
        scale: f32,
        zero_point: i32,
        min: i32,
        max: i32,
    }
}

fn quantize_params(op: &OpDesc<'_, '_>) -> Result<QuantizeParams> {
    expect_arity(op, 1, 1)?;
    let (input, output) = (input(op, 0)?, output(op, 0)?);
    expect_type(input, DataType::F32)?;
    let (min, max) = int_range(output.dtype).ok_or_else(|| unsupported_type(output))?;
    let (scale, zero_point) = per_tensor(output)?;
    Ok(QuantizeParams {
        len: same_elements(input, output)? as u32,
        output_type: type_code(output.dtype),
        scale,
        zero_point,
        min,
        max,
    })
}

fn quantize_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: QuantizeParams = *pod::read(ctx.persist)?;
    let len = p.len as usize;
    let input = in_slice::<f32>(ctx.inputs, 0, len)?;
    // `static_cast<int32_t>(round(x / scale)) + zero_point`: the conversion saturates and the
    // addition wraps, as on Arm and AArch64 (out-of-range conversions are undefined in C++).
    let q = |x: f32| {
        (round_f32(x / p.scale) as i32)
            .wrapping_add(p.zero_point)
            .clamp(p.min, p.max)
    };
    match p.output_type {
        t if t == type_code(DataType::I8) => {
            let out = out_slice::<i8>(ctx.outputs, 0, len)?;
            out.iter_mut()
                .zip(input)
                .for_each(|(o, &x)| *o = q(x) as i8);
        }
        t if t == type_code(DataType::U8) => {
            let out = out_slice::<u8>(ctx.outputs, 0, len)?;
            out.iter_mut()
                .zip(input)
                .for_each(|(o, &x)| *o = q(x) as u8);
        }
        _ => {
            let out = out_slice::<i16>(ctx.outputs, 0, len)?;
            out.iter_mut()
                .zip(input)
                .for_each(|(o, &x)| *o = q(x) as i16);
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `QUANTIZE` from `f32` to per-tensor `i8`, `u8` or `i16`.
    QUANTIZE,
    "rust/quantize",
    quantize_params,
    quantize_eval
);

pod_struct! {
    /// Parameters of `QUANTIZE` between integer types (TFLite `Requantize`).
    pub(crate) struct RequantizeParams {
        len: u32,
        input_type: u32,
        output_type: u32,
        multiplier: Multiplier,
        input_zero_point: i32,
        output_zero_point: i32,
        min: i32,
        max: i32,
    }
}

fn requantize_params(op: &OpDesc<'_, '_>) -> Result<RequantizeParams> {
    expect_arity(op, 1, 1)?;
    let (input, output) = (input(op, 0)?, output(op, 0)?);
    let supported = matches!(
        (input.dtype, output.dtype),
        (
            DataType::I8 | DataType::U8,
            DataType::I8 | DataType::U8 | DataType::I16
        ) | (DataType::I16, DataType::I8 | DataType::I16)
            | (DataType::I32, DataType::I8 | DataType::I16)
    );
    if !supported {
        return Err(unsupported(op));
    }
    let (input_scale, input_zero_point) = per_tensor(input)?;
    let (output_scale, output_zero_point) = per_tensor(output)?;
    if input.dtype == DataType::I16
        && output.dtype == DataType::I16
        && (input_zero_point != 0 || output_zero_point != 0)
    {
        return Err(unsupported(op));
    }
    let (min, max) = int_range(output.dtype).ok_or_else(|| unsupported_type(output))?;
    Ok(RequantizeParams {
        len: same_elements(input, output)? as u32,
        input_type: type_code(input.dtype),
        output_type: type_code(output.dtype),
        multiplier: quantize_multiplier(f64::from(input_scale) / f64::from(output_scale)),
        input_zero_point,
        output_zero_point,
        min,
        max,
    })
}

/// Reads element `i` of an integer buffer of type `code` as `i32`.
fn read_int(code: u32, bytes: &[u8], i: usize) -> i32 {
    match code {
        c if c == type_code(DataType::I8) => i32::from(bytes[i] as i8),
        c if c == type_code(DataType::U8) => i32::from(bytes[i]),
        c if c == type_code(DataType::I16) => {
            i32::from(i16::from_ne_bytes([bytes[2 * i], bytes[2 * i + 1]]))
        }
        _ => i32::from_ne_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ]),
    }
}

fn type_size(code: u32) -> usize {
    match code {
        c if c == type_code(DataType::I8) || c == type_code(DataType::U8) => 1,
        c if c == type_code(DataType::I16) => 2,
        _ => 4,
    }
}

fn requantize_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: RequantizeParams = *pod::read(ctx.persist)?;
    let len = p.len as usize;
    let input = in_slice::<u8>(ctx.inputs, 0, len * type_size(p.input_type))?;
    let out = out_slice::<u8>(ctx.outputs, 0, len * type_size(p.output_type))?;
    for i in 0..len {
        // Wrapping, like the two's-complement int32 arithmetic of TFLite: an int32 input at the
        // end of its range must not panic.
        let x = read_int(p.input_type, input, i).wrapping_sub(p.input_zero_point);
        let y = multiply_by_quantized_multiplier(x, p.multiplier)
            .wrapping_add(p.output_zero_point)
            .clamp(p.min, p.max);
        match type_size(p.output_type) {
            1 => out[i] = y as u8,
            _ => out[2 * i..2 * i + 2].copy_from_slice(&(y as i16).to_ne_bytes()),
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `QUANTIZE` between per-tensor quantized integer types.
    REQUANTIZE,
    "rust/requantize",
    requantize_params,
    requantize_eval
);

pod_struct! {
    /// Parameters of `DEQUANTIZE`. Per-channel scales and zero points follow in the persistent
    /// buffer.
    struct DequantizeHeader {
        len: u32,
        input_type: u32,
        /// Number of channels (1 for per-tensor quantization).
        channels: u32,
        /// Number of elements after the quantized dimension.
        inner: u32,
    }
}

fn dequantize_layout(channels: usize) -> (Region<DequantizeHeader>, Region<f32>, Region<i32>) {
    let mut layout = Layout::new();
    let header = layout.push::<DequantizeHeader>(1);
    let scales = layout.push::<f32>(channels);
    let zero_points = layout.push::<i32>(channels);
    (header, scales, zero_points)
}

/// Validates a `DEQUANTIZE` and returns its header.
fn dequantize_header(op: &OpDesc<'_, '_>) -> Result<DequantizeHeader> {
    expect_arity(op, 1, 1)?;
    let (input, output) = (input(op, 0)?, output(op, 0)?);
    expect_type(output, DataType::F32)?;
    let len = same_elements(input, output)? as u32;
    let mut header = DequantizeHeader {
        len,
        input_type: type_code(input.dtype),
        channels: 1,
        inner: 1,
    };
    match input.dtype {
        DataType::F16 => return Ok(header),
        DataType::I8 | DataType::U8 | DataType::I16 => {}
        _ => return Err(unsupported_type(input)),
    }
    let q = &input.quant;
    if q.is_per_tensor() {
        let (_, zero_point) = per_tensor(input)?;
        if input.dtype == DataType::I16 && zero_point != 0 {
            return Err(unsupported_type(input));
        }
        return Ok(header);
    }
    // Per-channel quantization (`u8` and `i8` only, like TFLite).
    let dims = input.dims.as_slice();
    let axis = usize::try_from(q.axis).map_err(|_| unsupported_type(input))?;
    let channels = q.scale.len();
    if input.dtype == DataType::I16
        || dims.get(axis).copied() != i32::try_from(channels).ok()
        || !(q.zero_point.len() == channels || q.zero_point.len() == 1)
    {
        return Err(unsupported_type(input));
    }
    header.channels = channels as u32;
    header.inner = dims[axis + 1..].iter().map(|&d| d as u32).product();
    Ok(header)
}

fn dequantize_plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    let header = dequantize_header(op)?;
    let mut layout = Layout::new();
    layout.push::<DequantizeHeader>(1);
    layout.push::<f32>(header.channels as usize);
    layout.push::<i32>(header.channels as usize);
    Ok(KernelPlan {
        persistent: layout.size(),
        scratch: 0,
    })
}

fn dequantize_prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let header = dequantize_header(op)?;
    let (h, scales, zero_points) = dequantize_layout(header.channels as usize);
    h.get_mut(persist)?[0] = header;
    let q = &input(op, 0)?.quant;
    for (i, s) in scales.get_mut(persist)?.iter_mut().enumerate() {
        *s = q.scale.get(i).unwrap_or(0.0);
    }
    for (i, z) in zero_points.get_mut(persist)?.iter_mut().enumerate() {
        let z64 = q
            .zero_point
            .get(i)
            .or_else(|| q.zero_point.get(0))
            .unwrap_or(0);
        *z = i32::try_from(z64).map_err(|_| unsupported(op))?;
    }
    Ok(())
}

/// Converts an IEEE 754 half-precision float to `f32` (exact).
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exponent = u32::from((h >> 10) & 0x1f);
    let mantissa = u32::from(h & 0x3ff);
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalise the mantissa.
            let mut e = 127 - 15 + 1;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

fn dequantize_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let header = pod::read::<DequantizeHeader>(ctx.persist)?;
    let (_, scales, zero_points) = dequantize_layout(header.channels as usize);
    let len = header.len as usize;
    let out = out_slice::<f32>(ctx.outputs, 0, len)?;
    if header.input_type == type_code(DataType::F16) {
        let input = in_slice::<u16>(ctx.inputs, 0, len)?;
        out.iter_mut()
            .zip(input)
            .for_each(|(o, &h)| *o = f16_to_f32(h));
        return Ok(());
    }
    let size = type_size(header.input_type);
    let input = in_slice::<u8>(ctx.inputs, 0, len * size)?;
    let scales = scales.get(ctx.persist)?;
    let zero_points = zero_points.get(ctx.persist)?;
    if header.channels == 1 {
        // `reference_integer_ops::Dequantize`: the product is computed in `double`.
        let scale = f64::from(scales[0]);
        let zero_point = zero_points[0];
        for (i, o) in out.iter_mut().enumerate() {
            let v = read_int(header.input_type, input, i) - zero_point;
            *o = (scale * f64::from(v)) as f32;
        }
    } else {
        // `reference_ops::PerChannelDequantize`: the product is computed in `float`.
        let channels = header.channels as usize;
        let inner = (header.inner as usize).max(1);
        for (i, o) in out.iter_mut().enumerate() {
            let c = (i / inner) % channels;
            let v = read_int(header.input_type, input, i) - zero_points[c];
            *o = scales[c] * v as f32;
        }
    }
    Ok(())
}

/// `DEQUANTIZE` of `i8`, `u8`, `i16` (per-tensor or, for 8-bit types, per-channel) and `f16`
/// tensors to `f32`.
pub(crate) static DEQUANTIZE: Kernel = Kernel {
    name: "rust/dequantize",
    plan: dequantize_plan,
    prepare: dequantize_prepare,
    eval: dequantize_eval,
};
