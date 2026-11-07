//! Element-wise activations: `RELU`, `RELU6`, `RELU_N1_TO_1`, `LEAKY_RELU`, `LOGISTIC`, `TANH` and
//! `HARD_SWISH` (TFLite `activations.cc`, reference kernels).

use crate::util::{
    clamp_f32, cmax, cmin, expect_arity, expect_type, fixed_kernel, in_slice, input, int_range,
    out_slice, output, per_tensor, pod_struct, same_shape, type_code, unsupported,
    unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod;
use nanoglia_core::quant::{
    Multiplier, multiply_by_quantized_multiplier, quantize_multiplier, round_f32,
};
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result, TensorDesc};

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
            // TFLite requires the output to cover [0, 1) exactly: scale 1/256, zero point -128
            // for int8 and 0 for uint8.
            let zero_point = if input.dtype == DataType::I8 { -128 } else { 0 };
            if f64::from(output_q.0) != 1.0 / 256.0 || output_q.1 != zero_point {
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

/// TFLite `sigmoid_table_uint16` (`kernels/internal/common.h`): `65536 × sigmoid(x)` for 256
/// values of `x` from 0 by steps of 1/24, the table of the int16 `LOGISTIC` and `TANH`.
const SIGMOID_TABLE_U16: [u16; 256] = [
    32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498, 40149, 40794,
    41432, 42064, 42688, 43304, 43912, 44511, 45102, 45683, 46255, 46817, 47369, 47911, 48443,
    48964, 49475, 49975, 50464, 50942, 51409, 51865, 52311, 52745, 53169, 53581, 53983, 54374,
    54755, 55125, 55485, 55834, 56174, 56503, 56823, 57133, 57433, 57724, 58007, 58280, 58544,
    58800, 59048, 59288, 59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110, 61279,
    61441, 61599, 61750, 61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886, 62990,
    63090, 63186, 63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835, 63903, 63968, 64030,
    64090, 64148, 64204, 64257, 64308, 64357, 64405, 64450, 64494, 64536, 64576, 64614, 64652,
    64687, 64721, 64754, 64786, 64816, 64845, 64873, 64900, 64926, 64950, 64974, 64997, 65019,
    65039, 65060, 65079, 65097, 65115, 65132, 65149, 65164, 65179, 65194, 65208, 65221, 65234,
    65246, 65258, 65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360,
    65367, 65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425, 65429, 65433,
    65438, 65442, 65445, 65449, 65453, 65456, 65459, 65462, 65465, 65468, 65471, 65474, 65476,
    65479, 65481, 65483, 65485, 65488, 65489, 65491, 65493, 65495, 65497, 65498, 65500, 65501,
    65503, 65504, 65505, 65507, 65508, 65509, 65510, 65511, 65512, 65513, 65514, 65515, 65516,
    65517, 65517, 65518, 65519, 65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524,
    65525, 65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529, 65529,
    65529, 65530, 65530, 65530, 65530, 65531, 65531, 65531, 65531, 65531, 65532, 65532, 65532,
    65532, 65532, 65532, 65533, 65533, 65533, 65533, 65533, 65533, 65533, 65533, 65534, 65534,
    65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65535,
];

/// TFLite `CheckedLog2` (`kernels/internal/quantization_util.h`): `log2(x)` rounded half away
/// from zero, and whether `x` is within `1e-3` (in `log2`) of that power of two, computed in
/// `f32` with `logf` like TFLite.
fn checked_log2(x: f32) -> (i32, bool) {
    let x_log2 = libm::logf(x) * (1.0 / libm::logf(2.0));
    let rounded = round_f32(x_log2);
    (rounded as i32, (x_log2 - rounded).abs() < 1e-3)
}

pod_struct! {
    /// Parameters of the int16 `LOGISTIC` and `TANH` (TFLite `reference_integer_ops::Logistic`
    /// and `Tanh` for `int16_t`).
    pub(crate) struct SigmoidI16Params {
        len: u32,
        /// Multiplier bringing the input to the scale 1/(3 × 4096) of the table, after the
        /// power-of-two substitution `3 << input_left_shift` of the reference kernels.
        input_multiplier: i32,
        /// Right shift applied after the multiplier (`input_left_shift` in TFLite).
        input_shift: i32,
    }
}

/// TFLite `SigmoidPrepare` and `TanhPrepare` for int16 (`activations.cc`): zero points 0, an
/// output scale of 2^-15 (exactly 1/32768 for `LOGISTIC`, as LiteRT's reference kernel checks),
/// and the input multiplier and shift of the table lookup.
fn sigmoid_i16_params(op: &OpDesc<'_, '_>) -> Result<SigmoidI16Params> {
    let (input, output, len) = unary(op)?;
    expect_type(input, DataType::I16)?;
    let tanh = match op.builtin() {
        Some(BuiltinOp::LOGISTIC) => false,
        Some(BuiltinOp::TANH) => true,
        _ => return Err(unsupported(op)),
    };
    let (input_scale, input_zero_point) = per_tensor(input)?;
    let (output_scale, output_zero_point) = per_tensor(output)?;
    if input_zero_point != 0 || output_zero_point != 0 {
        return Err(unsupported(op));
    }
    // TFLite requires `CheckedLog2(output_scale) == -15`; LiteRT's reference `LOGISTIC` also
    // requires the scale to be exactly 1/32768 (TensorFlow Lite Micro does not).
    let (output_log2, output_pot) = checked_log2(output_scale);
    if !output_pot || output_log2 != -15 || (!tanh && f64::from(output_scale) != 1.0 / 32768.0) {
        return Err(unsupported(op));
    }

    // kInputIntegerBits = 3.
    let (input_log2, input_pot) = checked_log2(input_scale);
    let mut input_shift = 12 + input_log2;
    let pot = input_pot && (input_shift == 0 || (tanh && input_shift == 1));
    let mut input_multiplier = 0;
    if !pot {
        // Brings the input to the scale 1/(3 × 4096) of the table, in `double`.
        let mut multiplier = f64::from(input_scale) * 4096.0 * 3.0;
        input_shift = 0;
        while multiplier <= 32767.0 / 2.0 && input_shift <= 30 {
            input_shift += 1;
            multiplier *= 2.0;
        }
        // TFLite truncates to `int32_t`. A multiplier of 0 (scales below 4e-14) would be taken
        // for the power-of-two case and shift 3 by 31 bits; one of 65536 or more (scales above
        // 5.33) overflows the `int32_t` product `input × multiplier`: both are undefined in C.
        if !(1.0..65536.0).contains(&multiplier) {
            return Err(unsupported(op));
        }
        input_multiplier = multiplier as i32;
    }
    // The power-of-two case of the reference kernels.
    if input_multiplier == 0 {
        input_multiplier = 3 << input_shift;
        input_shift = 0;
    }
    Ok(SigmoidI16Params {
        len,
        input_multiplier,
        input_shift,
    })
}

/// The input of the int16 table lookup: `(x × multiplier + round) >> shift`, at the scale
/// 1/(3 × 4096). The parameters keep the product inside an `i32`.
#[inline]
fn sigmoid_i16_input(p: &SigmoidI16Params, x: i16) -> i32 {
    let shift = p.input_shift.clamp(0, 31);
    let round = if shift > 0 { 1 << (shift - 1) } else { 0 };
    i32::from(x)
        .wrapping_mul(p.input_multiplier)
        .wrapping_add(round)
        >> shift
}

/// TFLite `reference_integer_ops::Logistic` for one `int16_t` value.
fn logistic_i16(p: &SigmoidI16Params, x: i16) -> i16 {
    let input = sigmoid_i16_input(p, x);
    let abs = input.unsigned_abs();
    let uh = (abs >> 9) as usize;
    let mut result: u32 = if uh >= 255 {
        // Saturate to maximum.
        0x7fff << 10
    } else {
        let ua = u32::from(SIGMOID_TABLE_U16[uh]);
        let ub = u32::from(SIGMOID_TABLE_U16[uh + 1]);
        let ut = abs & 0x1ff;
        (ua << 9).wrapping_add(ut.wrapping_mul(ub.wrapping_sub(ua)))
    };
    result = if input >= 0 {
        result.wrapping_add(1 << 9)
    } else {
        (1u32 << (16 + 9))
            .wrapping_sub(result)
            .wrapping_add((1 << 9) - 1)
    };
    (result >> 10) as i16
}

/// TFLite `reference_integer_ops::Tanh` for one `int16_t` value: `2 × sigmoid(2x) - 1` from the
/// same table.
fn tanh_i16(p: &SigmoidI16Params, x: i16) -> i16 {
    let input = sigmoid_i16_input(p, x);
    let abs = input.unsigned_abs();
    let uh = (abs >> 8) as usize;
    let result: i32 = if uh >= 255 {
        // Saturate to maximum.
        0xffff << 8
    } else {
        let ua = u32::from(SIGMOID_TABLE_U16[uh]);
        let ub = u32::from(SIGMOID_TABLE_U16[uh + 1]);
        let ut = abs & 0xff;
        (ua << 8).wrapping_add(ut.wrapping_mul(ub.wrapping_sub(ua))) as i32
    };
    let result = if input >= 0 {
        result - (1 << (14 + 9)) + (1 << (9 - 2))
    } else {
        -result + (1 << (14 + 9)) + (1 << (9 - 2)) - 1
    };
    (result >> (9 - 1)) as i16
}

fn logistic_i16_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: SigmoidI16Params = *pod::read(ctx.persist)?;
    map(ctx, p.len as usize, |x: i16| logistic_i16(&p, x))
}

fn tanh_i16_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: SigmoidI16Params = *pod::read(ctx.persist)?;
    map(ctx, p.len as usize, |x: i16| tanh_i16(&p, x))
}

fixed_kernel!(
    /// `LOGISTIC` on `i16` (16x8 models), interpolating TFLite's 256-entry sigmoid table.
    LOGISTIC_I16,
    "rust/logistic_i16",
    sigmoid_i16_params,
    logistic_i16_eval
);

fixed_kernel!(
    /// `TANH` on `i16` (16x8 models), interpolating TFLite's 256-entry sigmoid table.
    TANH_I16,
    "rust/tanh_i16",
    sigmoid_i16_params,
    tanh_i16_eval
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
