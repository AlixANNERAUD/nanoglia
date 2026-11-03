//! Element-wise activations: `RELU`, `RELU6`, `RELU_N1_TO_1`, `LEAKY_RELU`, `LOGISTIC`, `TANH` and
//! `HARD_SWISH` (TFLite `activations.cc`, reference kernels).

use crate::util::{
    clamp_f32, cmax, cmin, expect_arity, expect_type, fixed_kernel, in_slice, input, int_range,
    out_slice, output, per_tensor, pod_struct, same_shape, type_code, unsupported,
    unsupported_type,
};
use nanonn_core::backend::EvalCtx;
use nanonn_core::pod;
use nanonn_core::quant::{
    Multiplier, multiply_by_quantized_multiplier, quantize_multiplier, round_f32,
};
use nanonn_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result, TensorDesc};

/// Validates a unary operator of one input and one output of the same type and shape; returns
/// them and the number of elements.
fn unary<'o, 'a>(op: &'o OpDesc<'a, '_>) -> Result<(&'o TensorDesc<'a>, &'o TensorDesc<'a>, u32)> {
    expect_arity(op, 1, 1)?;
    let (input, output) = (input(op, 0)?, output(op, 0)?);
    if input.dtype != output.dtype {
        return Err(unsupported_type(output));
    }
    let len = same_shape(input, output)? as u32;
    Ok((input, output, len))
}

/// Applies `f` element-wise from input 0 to output 0.
#[inline]
fn map<I: pod::Pod, O: pod::Pod>(
    ctx: &mut EvalCtx<'_, '_>,
    len: usize,
    f: impl Fn(I) -> O,
) -> Result<()> {
    let input = in_slice::<I>(ctx.inputs, 0, len)?;
    let output = out_slice::<O>(ctx.outputs, 0, len)?;
    output.iter_mut().zip(input).for_each(|(o, &x)| *o = f(x));
    Ok(())
}

/// The real clamping range of a ReLU-like operator.
fn relu_range(op: &OpDesc<'_, '_>) -> Result<(f32, f32)> {
    match op.builtin() {
        Some(BuiltinOp::RELU) => Ok((0.0, f32::INFINITY)),
        Some(BuiltinOp::RELU6) => Ok((0.0, 6.0)),
        Some(BuiltinOp::RELU_N1_TO_1) => Ok((-1.0, 1.0)),
        _ => Err(unsupported(op)),
    }
}

pod_struct! {
    /// Parameters of a float ReLU.
    pub(crate) struct ReluF32Params {
        len: u32,
        min: f32,
        max: f32,
    }
}

fn relu_f32_params(op: &OpDesc<'_, '_>) -> Result<ReluF32Params> {
    let (input, _, len) = unary(op)?;
    expect_type(input, DataType::F32)?;
    let (min, max) = relu_range(op)?;
    Ok(ReluF32Params { len, min, max })
}

fn relu_f32_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: ReluF32Params = *pod::read(ctx.persist)?;
    map(ctx, p.len as usize, |x: f32| clamp_f32(x, p.min, p.max))
}

fixed_kernel!(
    /// `RELU`, `RELU6` and `RELU_N1_TO_1` on `f32`.
    RELU_F32,
    "rust/relu_f32",
    relu_f32_params,
    relu_f32_eval
);

pod_struct! {
    /// Parameters of a quantized ReLU (TFLite `QuantizedReluX`).
    pub(crate) struct ReluQParams {
        len: u32,
        dtype: u32,
        input_zero_point: i32,
        output_zero_point: i32,
        multiplier: Multiplier,
        min: i32,
        max: i32,
    }
}

fn relu_q_params(op: &OpDesc<'_, '_>) -> Result<ReluQParams> {
    let (input, output, len) = unary(op)?;
    let (tmin, tmax) = int_range(input.dtype).ok_or_else(|| unsupported_type(input))?;
    let (input_scale, input_zero_point) = per_tensor(input)?;
    let (output_scale, output_zero_point) = per_tensor(output)?;
    let (lo, hi) = relu_range(op)?;
    if input.dtype == DataType::I16
        && (input_zero_point != 0
            || output_zero_point != 0
            || op.builtin() == Some(BuiltinOp::RELU_N1_TO_1))
    {
        return Err(unsupported(op));
    }
    let quantize = |v: f32| output_zero_point.saturating_add(round_f32(v / output_scale) as i32);
    let min = tmin.max(quantize(lo));
    let max = if hi == f32::INFINITY {
        tmax
    } else {
        tmax.min(quantize(hi))
    };
    Ok(ReluQParams {
        len,
        dtype: type_code(input.dtype),
        input_zero_point,
        output_zero_point,
        multiplier: quantize_multiplier(f64::from(input_scale) / f64::from(output_scale)),
        min,
        max,
    })
}

fn relu_q_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: ReluQParams = *pod::read(ctx.persist)?;
    let f = |x: i32| {
        let y = p.output_zero_point
            + multiply_by_quantized_multiplier(x - p.input_zero_point, p.multiplier);
        y.max(p.min).min(p.max)
    };
    let len = p.len as usize;
    match p.dtype {
        t if t == type_code(DataType::I8) => map(ctx, len, |x: i8| f(x.into()) as i8),
        t if t == type_code(DataType::U8) => map(ctx, len, |x: u8| f(x.into()) as u8),
        _ => map(ctx, len, |x: i16| f(x.into()) as i16),
    }
}

fixed_kernel!(
    /// `RELU`, `RELU6` and `RELU_N1_TO_1` on `i8`, `u8` and `i16`, with rescaling.
    RELU_Q,
    "rust/relu_q",
    relu_q_params,
    relu_q_eval
);

fn leaky_alpha(op: &OpDesc<'_, '_>) -> Result<f32> {
    match op.options {
        OpOptions::LeakyRelu { alpha } => Ok(alpha),
        _ => Err(unsupported(op)),
    }
}

pod_struct! {
    /// Parameters of a float `LEAKY_RELU`.
    pub(crate) struct LeakyReluF32Params {
        len: u32,
        alpha: f32,
    }
}

fn leaky_relu_f32_params(op: &OpDesc<'_, '_>) -> Result<LeakyReluF32Params> {
    let (input, _, len) = unary(op)?;
    expect_type(input, DataType::F32)?;
    Ok(LeakyReluF32Params {
        len,
        alpha: leaky_alpha(op)?,
    })
}

fn leaky_relu_f32_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: LeakyReluF32Params = *pod::read(ctx.persist)?;
    map(
        ctx,
        p.len as usize,
        |x: f32| if x > 0.0 { x } else { x * p.alpha },
    )
}

fixed_kernel!(
    /// `LEAKY_RELU` on `f32`.
    LEAKY_RELU_F32,
    "rust/leaky_relu_f32",
    leaky_relu_f32_params,
    leaky_relu_f32_eval
);

pod_struct! {
    /// Parameters of a quantized `LEAKY_RELU` (TFLite `QuantizeLeakyRelu`).
    pub(crate) struct LeakyReluQParams {
        len: u32,
        dtype: u32,
        input_zero_point: i32,
        output_zero_point: i32,
        identity: Multiplier,
        alpha: Multiplier,
    }
}

fn leaky_relu_q_params(op: &OpDesc<'_, '_>) -> Result<LeakyReluQParams> {
    let (input, output, len) = unary(op)?;
    int_range(input.dtype).ok_or_else(|| unsupported_type(input))?;
    let alpha = leaky_alpha(op)?;
    let (input_scale, input_zero_point) = per_tensor(input)?;
    let (output_scale, output_zero_point) = per_tensor(output)?;
    if input.dtype == DataType::I16 && (input_zero_point != 0 || output_zero_point != 0) {
        return Err(unsupported(op));
    }
    let (input_scale, output_scale) = (f64::from(input_scale), f64::from(output_scale));
    Ok(LeakyReluQParams {
        len,
        dtype: type_code(input.dtype),
        input_zero_point,
        output_zero_point,
        identity: quantize_multiplier(input_scale / output_scale),
        alpha: quantize_multiplier(input_scale * f64::from(alpha) / output_scale),
    })
}

fn leaky_relu_q_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: LeakyReluQParams = *pod::read(ctx.persist)?;
    let negative_alpha = p.alpha.multiplier < 0;
    let alpha = Multiplier {
        multiplier: p.alpha.multiplier.wrapping_abs(),
        shift: p.alpha.shift,
    };
    let (tmin, tmax) = match p.dtype {
        t if t == type_code(DataType::I8) => (i32::from(i8::MIN), i32::from(i8::MAX)),
        t if t == type_code(DataType::U8) => (0, i32::from(u8::MAX)),
        _ => (i32::from(i16::MIN), i32::from(i16::MAX)),
    };
    let f = |x: i32| {
        let x = x - p.input_zero_point;
        let y = if x >= 0 {
            p.output_zero_point + multiply_by_quantized_multiplier(x, p.identity)
        } else {
            let scaled = multiply_by_quantized_multiplier(x, alpha);
            p.output_zero_point + if negative_alpha { -scaled } else { scaled }
        };
        y.clamp(tmin, tmax)
    };
    let len = p.len as usize;
    match p.dtype {
        t if t == type_code(DataType::I8) => map(ctx, len, |x: i8| f(x.into()) as i8),
        t if t == type_code(DataType::U8) => map(ctx, len, |x: u8| f(x.into()) as u8),
        _ => map(ctx, len, |x: i16| f(x.into()) as i16),
    }
}

fixed_kernel!(
    /// `LEAKY_RELU` on `i8`, `u8` and `i16`.
    LEAKY_RELU_Q,
    "rust/leaky_relu_q",
    leaky_relu_q_params,
    leaky_relu_q_eval
);

/// TFLite reference float `Logistic`.
fn logistic(x: f32) -> f32 {
    const CUTOFF_UPPER: f32 = 16.619_047;
    const CUTOFF_LOWER: f32 = -9.0;
    if x > CUTOFF_UPPER {
        1.0
    } else if x < CUTOFF_LOWER {
        crate::math::expf(x)
    } else {
        1.0 / (1.0 + crate::math::expf(-x))
    }
}

pod_struct! {
    /// Parameters of a float unary function.
    pub(crate) struct UnaryF32Params {
        len: u32,
    }
}

fn unary_f32_params(op: &OpDesc<'_, '_>) -> Result<UnaryF32Params> {
    let (input, _, len) = unary(op)?;
    expect_type(input, DataType::F32)?;
    Ok(UnaryF32Params { len })
}

fn logistic_f32_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: UnaryF32Params = *pod::read(ctx.persist)?;
    map(ctx, p.len as usize, logistic)
}

fn tanh_f32_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: UnaryF32Params = *pod::read(ctx.persist)?;
    map(ctx, p.len as usize, crate::math::tanhf)
}

/// TFLite reference float `HardSwish`: `x * min(6, max(0, x + 3)) / 6`.
fn hard_swish(x: f32) -> f32 {
    x * cmin(6.0, cmax(0.0, x + 3.0)) / 6.0
}

fn hard_swish_f32_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: UnaryF32Params = *pod::read(ctx.persist)?;
    map(ctx, p.len as usize, hard_swish)
}

fixed_kernel!(
    /// `LOGISTIC` on `f32`.
    LOGISTIC_F32,
    "rust/logistic_f32",
    unary_f32_params,
    logistic_f32_eval
);

fixed_kernel!(
    /// `TANH` on `f32`.
    TANH_F32,
    "rust/tanh_f32",
    unary_f32_params,
    tanh_f32_eval
);

fixed_kernel!(
    /// `HARD_SWISH` on `f32`.
    HARD_SWISH_F32,
    "rust/hard_swish_f32",
    unary_f32_params,
    hard_swish_f32_eval
);

pod_struct! {
    /// Parameters of an 8-bit lookup-table activation.
    pub(crate) struct LutParams {
        len: u32,
        /// Output for each input byte (indexed by the raw byte, as TFLite does for `i8`).
        lut: [u8; 256],
    }
}

/// TFLite `LUTPopulateInt8`: tabulates `f` for every 8-bit input value.
fn lut_populate(
    dtype: DataType,
    (input_scale, input_zero_point): (f32, i32),
    (output_scale, output_zero_point): (f32, i32),
    f: fn(f32) -> f32,
) -> [u8; 256] {
    let mut lut = [0u8; 256];
    let inverse_scale = 1.0 / output_scale;
    let (min, max) = int_range(dtype).unwrap_or((-128, 127));
    for value in min..=max {
        let dequantized = input_scale * (value - input_zero_point) as f32;
        let transformed = f(dequantized);
        let rescaled = round_f32(transformed * inverse_scale);
        let quantized = (rescaled + output_zero_point as f32) as i32;
        lut[(value as u8) as usize] = quantized.clamp(min, max) as u8;
    }
    lut
}

fn lut_params(op: &OpDesc<'_, '_>) -> Result<LutParams> {
    let (input, output, len) = unary(op)?;
    if !matches!(input.dtype, DataType::I8 | DataType::U8) {
        return Err(unsupported_type(input));
    }
    let input_q = per_tensor(input)?;
    let output_q = per_tensor(output)?;
    let f: fn(f32) -> f32 = match op.builtin() {
        Some(BuiltinOp::LOGISTIC) => {
            // TFLite requires the output to cover [0, 1) exactly.
            if f64::from(output_q.0) != 1.0 / 256.0 {
                return Err(unsupported(op));
            }
            |x| 1.0 / (1.0 + crate::math::expf(-x))
        }
        Some(BuiltinOp::TANH) => crate::math::tanhf,
        _ => return Err(unsupported(op)),
    };
    Ok(LutParams {
        len,
        lut: lut_populate(input.dtype, input_q, output_q, f),
    })
}

fn lut_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p = pod::read::<LutParams>(ctx.persist)?;
    let lut = &p.lut;
    map(ctx, p.len as usize, |x: u8| lut[usize::from(x)])
}

fixed_kernel!(
    /// `LOGISTIC` and `TANH` on `i8` and `u8`, through a 256-entry table like TFLite.
    LUT_Q,
    "rust/lut_q",
    lut_params,
    lut_eval
);

pod_struct! {
    /// TFLite `HardSwishParams`.
    pub(crate) struct HardSwishQParams {
        len: u32,
        dtype: u32,
        input_zero_point: i32,
        output_zero_point: i32,
        reluish_multiplier_exponent: i32,
        output_multiplier_exponent: i32,
        reluish_multiplier: i32,
        output_multiplier: i32,
    }
}

/// TFLite `DownScaleInt32ToInt16Multiplier`.
fn downscale_multiplier(m: i32) -> i16 {
    const ROUNDING: i32 = 1 << 15;
    if m >= i32::MAX - ROUNDING {
        return i16::MAX;
    }
    ((m + ROUNDING) >> 16) as i16
}

fn hard_swish_q_params(op: &OpDesc<'_, '_>) -> Result<HardSwishQParams> {
    let (input, output, len) = unary(op)?;
    if !matches!(input.dtype, DataType::I8 | DataType::U8) {
        return Err(unsupported_type(input));
    }
    let (input_scale, input_zero_point) = per_tensor(input)?;
    let (output_scale, output_zero_point) = per_tensor(output)?;
    let hires_input_scale = (1.0f32 / 128.0) * input_scale;
    let reluish_scale = 3.0f32 / 32768.0;
    let output = quantize_multiplier(f64::from(hires_input_scale / output_scale));
    let reluish = quantize_multiplier(f64::from(hires_input_scale / reluish_scale));
    if output.shift > 0 || output.multiplier < 0 || reluish.multiplier < 0 {
        return Err(unsupported(op));
    }
    Ok(HardSwishQParams {
        len,
        dtype: type_code(input.dtype),
        input_zero_point,
        output_zero_point,
        reluish_multiplier_exponent: reluish.shift,
        output_multiplier_exponent: output.shift,
        reluish_multiplier: downscale_multiplier(reluish.multiplier).into(),
        output_multiplier: downscale_multiplier(output.multiplier).into(),
    })
}

/// gemmlowp `SaturatingRoundingDoublingHighMul` on `i16`.
fn srdhm16(a: i16, b: i16) -> i16 {
    if a == b && a == i16::MIN {
        return i16::MAX;
    }
    let ab = i32::from(a) * i32::from(b);
    let nudge = if ab >= 0 { 1 << 14 } else { 1 - (1 << 14) };
    ((ab + nudge) / (1 << 15)) as i16
}

/// TFLite `SaturatingDoublingHighMul` on `i16` (rounds toward zero).
fn sdhm16(a: i16, b: i16) -> i16 {
    if a == b && a == i16::MIN {
        return i16::MAX;
    }
    ((i32::from(a) * i32::from(b)) / (1 << 15)) as i16
}

/// gemmlowp `RoundingDivideByPOT` on `i16`.
fn rounding_divide_by_pot16(x: i16, exponent: i32) -> i16 {
    let mask = ((1i64 << exponent) - 1) as i16;
    let remainder = x & mask;
    let threshold = (mask >> 1) + i16::from(x < 0);
    ((i32::from(x) >> exponent) + i32::from(remainder > threshold)) as i16
}

/// TFLite `SaturatingLeftShift` on `i16`.
fn saturating_left_shift16(value: i16, amount: i32) -> i16 {
    let result = i64::from(value) * (1i64 << amount);
    result.clamp(i16::MIN.into(), i16::MAX.into()) as i16
}

/// TFLite reference quantized `HardSwish` for one value.
fn hard_swish_q(p: &HardSwishQParams, x: i32, (min, max): (i32, i32)) -> i32 {
    let input_value = (x - p.input_zero_point) as i16;
    let hires = (i32::from(input_value) * (1 << 7)) as i16;
    let preshift_output = srdhm16(hires, p.output_multiplier as i16);
    let mut reluish = hires;
    if p.reluish_multiplier_exponent > 0 {
        reluish = saturating_left_shift16(reluish, p.reluish_multiplier_exponent - 1);
    }
    reluish = srdhm16(reluish, p.reluish_multiplier as i16);
    if p.reluish_multiplier_exponent > 0 {
        reluish = saturating_left_shift16(reluish, 1);
    }
    if p.reluish_multiplier_exponent < 0 {
        reluish = rounding_divide_by_pot16(reluish, -p.reluish_multiplier_exponent);
    }
    reluish = ((i32::from(reluish) + (1 << 15)) >> 1) as i16;
    let preshift = sdhm16(reluish, preshift_output);
    let mut out = rounding_divide_by_pot16(preshift, -p.output_multiplier_exponent);
    out = (i32::from(out) + p.output_zero_point) as i16;
    i32::from(out).clamp(min, max)
}

fn hard_swish_q_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: HardSwishQParams = *pod::read(ctx.persist)?;
    let len = p.len as usize;
    if p.dtype == type_code(DataType::I8) {
        map(ctx, len, |x: i8| {
            hard_swish_q(&p, x.into(), (-128, 127)) as i8
        })
    } else {
        map(ctx, len, |x: u8| hard_swish_q(&p, x.into(), (0, 255)) as u8)
    }
}

fixed_kernel!(
    /// `HARD_SWISH` on `i8` and `u8`.
    HARD_SWISH_Q,
    "rust/hard_swish_q",
    hard_swish_q_params,
    hard_swish_q_eval
);
