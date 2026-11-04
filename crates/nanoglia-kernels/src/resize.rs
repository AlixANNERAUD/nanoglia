//! `RESIZE_NEAREST_NEIGHBOR` (TFLite `resize_nearest_neighbor.cc`, reference
//! `ResizeNearestNeighbor`) and `RESIZE_BILINEAR` (TFLite `resize_bilinear.cc`, the kernels its
//! reference registration runs: `ResizeBilinear` in `float` for `f32` and `u8`,
//! `ResizeBilinearInteger` in fixed point for `i8` and `i16`), with a constant output size.
//!
//! Both operators take an NHWC input and resize its height and width. Like TFLite, they do not
//! look at quantization: the output values are computed from the raw input values, so the input
//! and the output are expected to share their quantization, as converters write them.

use crate::math::mul_add;
use crate::util::{
    constant, dims4, fixed_kernel, in_slice, input, out_slice, output, pod_struct, shape_mismatch,
    type_code, unsupported, unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::quant::round_f32;
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result, pod};

pod_struct! {
    /// Parameters of the resize operators.
    pub(crate) struct ResizeParams {
        batches: i32,
        in_h: i32,
        in_w: i32,
        depth: i32,
        out_h: i32,
        out_w: i32,
        /// The element type ([`type_code`]).
        dtype: u32,
        /// Size of an element in bytes.
        size: u32,
        align_corners: u32,
        half_pixel_centers: u32,
    }
}

impl ResizeParams {
    fn input_len(&self) -> usize {
        (self.batches * self.in_h * self.in_w * self.depth) as usize
    }

    fn output_len(&self) -> usize {
        (self.batches * self.out_h * self.out_w * self.depth) as usize
    }
}

/// The largest input or output height and width: keeps the fixed-point arithmetic of
/// `ResizeBilinearInteger` (`(1 << 10) * size`) and every offset inside an `i32`.
const MAX_SIZE: i32 = 1 << 20;

/// The checks both operators share (TFLite `Prepare` and `ResizeOutputTensor`).
fn resize_params(op: &OpDesc<'_, '_>, code: BuiltinOp) -> Result<ResizeParams> {
    let OpOptions::Resize {
        align_corners,
        half_pixel_centers,
    } = op.options
    else {
        return Err(unsupported(op));
    };
    if op.builtin() != Some(code) || op.inputs.len() != 2 || op.outputs.len() != 1 {
        return Err(unsupported(op));
    }
    let (data, size, out) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    if out.dtype != data.dtype {
        return Err(unsupported_type(out));
    }
    let element = data.dtype.size().ok_or_else(|| unsupported_type(data))?;
    // The size: a constant int32 vector of two elements, the new height and width.
    if size.dtype != DataType::I32 || size.dims.as_slice() != [2] {
        return Err(unsupported(op));
    }
    let sizes = constant(op, size)?;
    let read =
        |i: usize| nanoglia_core::le::read::<i32>(sizes, 4 * i).ok_or_else(|| unsupported(op));
    let (out_h, out_w) = (read(0)?, read(1)?);
    let [batches, in_h, in_w, depth] = dims4(data)?;
    if out.dims.as_slice() != [batches, out_h, out_w, depth] {
        return Err(shape_mismatch(out));
    }
    dims4(out)?;
    let spatial = 0..=MAX_SIZE;
    if ![in_h, in_w, out_h, out_w]
        .iter()
        .all(|d| spatial.contains(d))
    {
        return Err(shape_mismatch(out));
    }
    // An empty input with a non-empty output: TFLite would read before the input.
    if (in_h == 0 || in_w == 0) && out_h > 0 && out_w > 0 && batches > 0 && depth > 0 {
        return Err(shape_mismatch(data));
    }
    Ok(ResizeParams {
        batches,
        in_h,
        in_w,
        depth,
        out_h,
        out_w,
        dtype: type_code(data.dtype),
        size: element as u32,
        align_corners: u32::from(align_corners),
        half_pixel_centers: u32::from(half_pixel_centers),
    })
}

// --------------------------------------------------------------------------------------------
// RESIZE_NEAREST_NEIGHBOR
// --------------------------------------------------------------------------------------------

fn nearest_params(op: &OpDesc<'_, '_>) -> Result<ResizeParams> {
    let p = resize_params(op, BuiltinOp::RESIZE_NEAREST_NEIGHBOR)?;
    // TFLite copies elements of 8, 16 or 32 bits.
    let data = input(op, 0)?;
    let supported = matches!(
        data.dtype,
        DataType::F32
            | DataType::I32
            | DataType::U8
            | DataType::I8
            | DataType::I16
            | DataType::Bool
    );
    if !supported {
        return Err(unsupported_type(data));
    }
    Ok(p)
}

/// TFLite `GetNearestNeighbor`, in `float` like the reference.
fn nearest(value: i32, in_size: i32, out_size: i32, align_corners: bool, half: bool) -> i32 {
    let scale = if align_corners && out_size > 1 {
        (in_size - 1) as f32 / (out_size - 1) as f32
    } else {
        in_size as f32 / out_size as f32
    };
    let offset = if half { 0.5f32 } else { 0.0 };
    let scaled = (value as f32 + offset) * scale;
    let index = if align_corners {
        round_f32(scaled) as i32
    } else {
        libm::floorf(scaled) as i32
    };
    let index = index.min(in_size - 1);
    // Without half-pixel centers the index is never negative; the clamp is TFLite's.
    if half { index.max(0) } else { index }
}

fn nearest_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: ResizeParams = *pod::read(ctx.persist)?;
    let size = p.size as usize;
    let input = in_slice::<u8>(ctx.inputs, 0, p.input_len() * size)?;
    let output = out_slice::<u8>(ctx.outputs, 0, p.output_len() * size)?;
    if output.is_empty() {
        return Ok(());
    }
    let (align, half) = (p.align_corners != 0, p.half_pixel_centers != 0);
    let pixel = p.depth as usize * size;
    let row = p.in_w as usize * pixel;
    let image = p.in_h as usize * row;
    let mut chunks = output.chunks_exact_mut(pixel);
    for b in 0..p.batches as usize {
        for y in 0..p.out_h {
            // `0 <= in_y < in_h` (the input is not empty when the output is not).
            let in_y = nearest(y, p.in_h, p.out_h, align, half).max(0) as usize;
            for x in 0..p.out_w {
                let in_x = nearest(x, p.in_w, p.out_w, align, half).max(0) as usize;
                let from = b * image + in_y * row + in_x * pixel;
                let src = input
                    .get(from..from + pixel)
                    .ok_or_else(|| crate::util::size_mismatch(from + pixel, input.len()))?;
                let dst = chunks
                    .next()
                    .ok_or_else(|| crate::util::size_mismatch(pixel, 0))?;
                dst.copy_from_slice(src);
            }
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `RESIZE_NEAREST_NEIGHBOR` of `f32`, `i32`, `u8`, `i8`, `i16` and `bool` values.
    RESIZE_NEAREST_NEIGHBOR,
    "rust/resize_nearest_neighbor",
    nearest_params,
    nearest_eval
);

// --------------------------------------------------------------------------------------------
// RESIZE_BILINEAR
// --------------------------------------------------------------------------------------------

fn bilinear_params(op: &OpDesc<'_, '_>, float: bool) -> Result<ResizeParams> {
    let p = resize_params(op, BuiltinOp::RESIZE_BILINEAR)?;
    let data = input(op, 0)?;
    let supported = if float {
        data.dtype == DataType::F32
    } else {
        matches!(data.dtype, DataType::U8 | DataType::I8 | DataType::I16)
    };
    if !supported {
        return Err(unsupported_type(data));
    }
    // TFLite refuses this combination when the size is constant, and a size of zero.
    if (p.align_corners != 0 && p.half_pixel_centers != 0) || p.out_h <= 0 || p.out_w <= 0 {
        return Err(unsupported(op));
    }
    Ok(p)
}

fn bilinear_params_f32(op: &OpDesc<'_, '_>) -> Result<ResizeParams> {
    bilinear_params(op, true)
}

fn bilinear_params_q(op: &OpDesc<'_, '_>) -> Result<ResizeParams> {
    bilinear_params(op, false)
}

/// The scale of one dimension of the `float` kernel.
fn float_scale(in_size: i32, out_size: i32, align_corners: bool) -> f32 {
    if align_corners && out_size > 1 {
        (in_size - 1) as f32 / (out_size - 1) as f32
    } else {
        in_size as f32 / out_size as f32
    }
}

/// TFLite `ComputeInterpolationValues`: the input coordinate of an output coordinate and the two
/// input indices around it. LiteRT is compiled with floating-point contraction, so
/// `(value + 0.5f) * scale - 0.5f` is one fused multiply-add.
fn float_interpolation(value: i32, scale: f32, half: bool, in_size: i32) -> (f32, i32, i32) {
    let scaled = if half {
        mul_add(value as f32 + 0.5, scale, -0.5)
    } else {
        value as f32 * scale
    };
    let lower = (libm::floorf(scaled) as i32).max(0);
    let upper = (libm::ceilf(scaled) as i32).min(in_size - 1);
    // In range by construction; clamped so that no model can index outside the input.
    (scaled, lower.min(in_size - 1), upper.max(0))
}

/// An element type of the `float` kernel.
trait FloatResize: Copy + nanoglia_core::pod::Pod {
    /// Added before the conversion back: `0.5` for integers (rounding), `0.0` for floats.
    const ROUNDING: f32;
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl FloatResize for f32 {
    const ROUNDING: f32 = 0.0;
    fn to_f32(self) -> f32 {
        self
    }
    fn from_f32(value: f32) -> Self {
        value
    }
}

impl FloatResize for u8 {
    const ROUNDING: f32 = 0.5;
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
    fn from_f32(value: f32) -> Self {
        // `static_cast<uint8_t>`: truncation (the value is in range: the weights sum to one).
        value as u8
    }
}

/// The reference `ResizeBilinear`.
fn bilinear_float<T: FloatResize>(p: &ResizeParams, input: &[T], output: &mut [T]) {
    let (align, half) = (p.align_corners != 0, p.half_pixel_centers != 0);
    let height_scale = float_scale(p.in_h, p.out_h, align);
    let width_scale = float_scale(p.in_w, p.out_w, align);
    let depth = p.depth as usize;
    let at = |b: usize, y: i32, x: i32| {
        ((b * p.in_h as usize + y as usize) * p.in_w as usize + x as usize) * depth
    };
    let mut o = 0;
    for b in 0..p.batches as usize {
        for y in 0..p.out_h {
            let (input_y, y0, y1) = float_interpolation(y, height_scale, half, p.in_h);
            let dy = input_y - y0 as f32;
            for x in 0..p.out_w {
                let (input_x, x0, x1) = float_interpolation(x, width_scale, half, p.in_w);
                let dx = input_x - x0 as f32;
                let (i00, i10, i01, i11) =
                    (at(b, y0, x0), at(b, y1, x0), at(b, y0, x1), at(b, y1, x1));
                for c in 0..depth {
                    // `a * (1 - dy) * (1 - dx) + b * dy * (1 - dx) + c * (1 - dy) * dx
                    // + d * dy * dx + rounding`, with the multiply-adds Clang fuses.
                    let v00 = input[i00 + c].to_f32();
                    let v10 = input[i10 + c].to_f32();
                    let v01 = input[i01 + c].to_f32();
                    let v11 = input[i11 + c].to_f32();
                    let mut acc = mul_add(v00 * (1.0 - dy), 1.0 - dx, v10 * dy * (1.0 - dx));
                    acc = mul_add(v01 * (1.0 - dy), dx, acc);
                    acc = mul_add(v11 * dy, dx, acc);
                    output[o] = T::from_f32(acc + T::ROUNDING);
                    o += 1;
                }
            }
        }
    }
}

/// TFLite `ComputeInterpolationValuesInteger`: coordinates in 1/1024 of a pixel.
fn integer_interpolation(value: i32, scale_10: i32, half: bool, in_size: i32) -> (i32, i32, i32) {
    let scaled = if half {
        value
            .wrapping_mul(scale_10)
            .wrapping_add(scale_10 / 2)
            .wrapping_sub(1 << 9)
    } else {
        value.wrapping_mul(scale_10)
    };
    let lower = (scaled / (1 << 10)).max(0);
    let upper = (scaled.wrapping_add((1 << 10) - 1) / (1 << 10)).min(in_size - 1);
    // In range by construction; clamped so that no model can index outside the input.
    (scaled, lower.min(in_size - 1), upper.max(0))
}

/// The scale of one dimension of the fixed-point kernel, in 1/1024.
fn integer_scale(in_size: i32, out_size: i32, align_corners: bool) -> i32 {
    if align_corners && out_size > 1 {
        ((1 << 10) * (in_size - 1) + (out_size - 1) / 2) / (out_size - 1)
    } else {
        ((1 << 10) * in_size + out_size / 2) / out_size
    }
}

/// An element type of the fixed-point kernel.
trait IntegerResize: Copy + nanoglia_core::pod::Pod {
    fn to_i64(self) -> i64;
    /// `static_cast<T>` of an `int64_t`: keeps the low bits.
    fn wrap(value: i64) -> Self;
}

impl IntegerResize for i8 {
    fn to_i64(self) -> i64 {
        i64::from(self)
    }
    fn wrap(value: i64) -> Self {
        value as i8
    }
}

impl IntegerResize for i16 {
    fn to_i64(self) -> i64 {
        i64::from(self)
    }
    fn wrap(value: i64) -> Self {
        value as i16
    }
}

/// The reference `ResizeBilinearInteger`.
fn bilinear_integer<T: IntegerResize>(p: &ResizeParams, input: &[T], output: &mut [T]) {
    const ONE: i32 = 1 << 10;
    let (align, half) = (p.align_corners != 0, p.half_pixel_centers != 0);
    let height_scale = integer_scale(p.in_h, p.out_h, align);
    let width_scale = integer_scale(p.in_w, p.out_w, align);
    let depth = p.depth as usize;
    let at = |b: usize, y: i32, x: i32| {
        ((b * p.in_h as usize + y as usize) * p.in_w as usize + x as usize) * depth
    };
    let mut o = 0;
    for b in 0..p.batches as usize {
        for y in 0..p.out_h {
            let (input_y, y0, y1) = integer_interpolation(y, height_scale, half, p.in_h);
            let dy = input_y.wrapping_sub(ONE.wrapping_mul(y0));
            for x in 0..p.out_w {
                let (input_x, x0, x1) = integer_interpolation(x, width_scale, half, p.in_w);
                let dx = input_x.wrapping_sub(ONE.wrapping_mul(x0));
                let (i00, i10, i01, i11) =
                    (at(b, y0, x0), at(b, y1, x0), at(b, y0, x1), at(b, y1, x1));
                let (wy0, wy1) = (i64::from(ONE.wrapping_sub(dy)), i64::from(dy));
                let (wx0, wx1) = (i64::from(ONE.wrapping_sub(dx)), i64::from(dx));
                for c in 0..depth {
                    let output_20 = (input[i00 + c].to_i64() * wy0 * wx0)
                        .wrapping_add(input[i10 + c].to_i64() * wy1 * wx0)
                        .wrapping_add(input[i01 + c].to_i64() * wy0 * wx1)
                        .wrapping_add(input[i11 + c].to_i64() * wy1 * wx1);
                    let value = if cfg!(feature = "single-rounding") {
                        output_20.wrapping_add(1 << 19) >> 20
                    } else {
                        let round = if output_20 > 0 { 1 << 19 } else { -(1 << 19) };
                        output_20.wrapping_add(round) / (1 << 20)
                    };
                    output[o] = T::wrap(value);
                    o += 1;
                }
            }
        }
    }
}

fn bilinear_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: ResizeParams = *pod::read(ctx.persist)?;
    let (n_in, n_out) = (p.input_len(), p.output_len());
    // Otherwise, the input is not empty (checked by `resize_params`).
    if n_out == 0 {
        return Ok(());
    }
    match p.dtype {
        t if t == type_code(DataType::F32) => bilinear_float::<f32>(
            &p,
            in_slice(ctx.inputs, 0, n_in)?,
            out_slice(ctx.outputs, 0, n_out)?,
        ),
        t if t == type_code(DataType::U8) => bilinear_float::<u8>(
            &p,
            in_slice(ctx.inputs, 0, n_in)?,
            out_slice(ctx.outputs, 0, n_out)?,
        ),
        t if t == type_code(DataType::I8) => bilinear_integer::<i8>(
            &p,
            in_slice(ctx.inputs, 0, n_in)?,
            out_slice(ctx.outputs, 0, n_out)?,
        ),
        t if t == type_code(DataType::I16) => bilinear_integer::<i16>(
            &p,
            in_slice(ctx.inputs, 0, n_in)?,
            out_slice(ctx.outputs, 0, n_out)?,
        ),
        _ => return Err(crate::util::size_mismatch(0, 0)),
    }
    Ok(())
}

fixed_kernel!(
    /// `RESIZE_BILINEAR` of `f32` values.
    RESIZE_BILINEAR_F32,
    "rust/resize_bilinear_f32",
    bilinear_params_f32,
    bilinear_eval
);

fixed_kernel!(
    /// `RESIZE_BILINEAR` of `u8` values (in `float`, like TFLite), and of `i8` and `i16` values
    /// (in fixed point).
    RESIZE_BILINEAR_Q,
    "rust/resize_bilinear_q",
    bilinear_params_q,
    bilinear_eval
);
