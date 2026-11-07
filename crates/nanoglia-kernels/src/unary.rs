//! Element-wise unary operators: `ABS`, `NEG`, `SQRT`, `RSQRT`, `EXP` and `LOG` on `f32`
//! (TensorFlow Lite Micro `elementwise.cc`, `neg.cc`, `exp.cc`), `ABS` and `NEG` on plain
//! integers (LiteRT's reference kernels), and the quantized `ABS` (int8, int16) and `RSQRT`
//! (int8) of TensorFlow Lite Micro's `elementwise.cc`.
//!
//! Quantized `ABS` rescales `|x - input_zero_point|` with `QuantizeMultiplier(input_scale /
//! output_scale)` when the scales differ; quantized `RSQRT` computes `1 / sqrt(x -
//! input_zero_point)` with TFLite's fixed-point `GetInvSqrtQuantizedMultiplierExp`, then
//! requantizes with `QuantizeMultiplier(1 / (sqrt(input_scale) * output_scale))`. Both
//! multipliers are computed in `f64`, as in TensorFlow Lite Micro (LiteRT computes the ratios in
//! `f32`). RSQRT has 256 possible inputs: `prepare` tabulates them, and an input below the zero
//! point (a negative value) makes `compute` fail with `InvalidArgument`, as TFLite fails.

use crate::util::{
    expect_arity, fixed_kernel, in_slice, input, int_range, out_slice, output, per_tensor,
    pod_struct, same_shape, type_code, unsupported, unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod;
use nanoglia_core::quant::{
    Multiplier, multiply_by_quantized_multiplier, quantize_multiplier, rounding_divide_by_pot,
    saturating_rounding_doubling_high_mul,
};
use nanoglia_core::{BuiltinOp, DataType, Detail, Error, OpDesc, Result, TensorDesc};

const OP_ABS: u32 = 0;
const OP_NEG: u32 = 1;
const OP_SQRT: u32 = 2;
const OP_RSQRT: u32 = 3;
const OP_EXP: u32 = 4;
const OP_LOG: u32 = 5;

/// The operator code of `op`.
fn unary_code(op: &OpDesc<'_, '_>) -> Result<u32> {
    Ok(match op.builtin() {
        Some(BuiltinOp::ABS) => OP_ABS,
        Some(BuiltinOp::NEG) => OP_NEG,
        Some(BuiltinOp::SQRT) => OP_SQRT,
        Some(BuiltinOp::RSQRT) => OP_RSQRT,
        Some(BuiltinOp::EXP) => OP_EXP,
        Some(BuiltinOp::LOG) => OP_LOG,
        _ => return Err(unsupported(op)),
    })
}

/// The input and output of a unary operator, of the same type and shape, and their element
/// count.
fn unary_tensors<'o, 'a>(
    op: &'o OpDesc<'a, '_>,
) -> Result<(&'o TensorDesc<'a>, &'o TensorDesc<'a>, usize)> {
    expect_arity(op, 1, 1)?;
    let (x, y) = (input(op, 0)?, output(op, 0)?);
    if x.dtype != y.dtype {
        return Err(unsupported_type(y));
    }
    let len = same_shape(x, y)?;
    Ok((x, y, len))
}

pod_struct! {
    /// Parameters of a non-quantized unary operator.
    pub(crate) struct UnaryParams {
        op: u32,
        dtype: u32,
        len: u32,
    }
}

fn unary_f32_params(op: &OpDesc<'_, '_>) -> Result<UnaryParams> {
    let code = unary_code(op)?;
    let (x, _, len) = unary_tensors(op)?;
    if x.dtype != DataType::F32 {
        return Err(unsupported_type(x));
    }
    Ok(UnaryParams {
        op: code,
        dtype: type_code(DataType::F32),
        len: len as u32,
    })
}

/// Applies `f` to every element.
fn map<T: pod::Pod>(ctx: &mut EvalCtx<'_, '_>, len: u32, f: impl Fn(T) -> T) -> Result<()> {
    let x = in_slice::<T>(ctx.inputs, 0, len as usize)?;
    let y = out_slice::<T>(ctx.outputs, 0, len as usize)?;
    for (y, &x) in y.iter_mut().zip(x) {
        *y = f(x);
    }
    Ok(())
}

fn unary_f32_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: UnaryParams = *pod::read(ctx.persist)?;
    match p.op {
        OP_ABS => map(ctx, p.len, f32::abs),
        OP_NEG => map(ctx, p.len, |x: f32| -x),
        OP_SQRT => map(ctx, p.len, libm::sqrtf),
        OP_RSQRT => map(ctx, p.len, |x: f32| 1.0 / libm::sqrtf(x)),
        OP_EXP => map(ctx, p.len, crate::math::expf),
        _ => map(ctx, p.len, libm::logf),
    }
}

fixed_kernel!(
    /// `ABS`, `NEG`, `SQRT`, `RSQRT`, `EXP` and `LOG` on `f32`.
    UNARY_F32,
    "rust/unary_f32",
    unary_f32_params,
    unary_f32_eval
);

fn unary_int_params(op: &OpDesc<'_, '_>) -> Result<UnaryParams> {
    let code = unary_code(op)?;
    let (x, _, len) = unary_tensors(op)?;
    // LiteRT's `ABS` on int32 and on int16 without quantization, `NEG` on int32 and int64.
    let supported = match (code, x.dtype) {
        (OP_ABS, DataType::I32) => true,
        (OP_ABS, DataType::I16) => !x.quant.is_quantized(),
        (OP_NEG, DataType::I32 | DataType::I64) => true,
        _ => false,
    };
    if !supported {
        return Err(unsupported_type(x));
    }
    Ok(UnaryParams {
        op: code,
        dtype: type_code(x.dtype),
        len: len as u32,
    })
}

fn unary_int_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: UnaryParams = *pod::read(ctx.persist)?;
    // `std::abs` and `-x` of the most negative value wrap around in practice.
    match (p.op, p.dtype) {
        (OP_ABS, t) if t == type_code(DataType::I16) => map(ctx, p.len, i16::wrapping_abs),
        (OP_ABS, _) => map(ctx, p.len, i32::wrapping_abs),
        (_, t) if t == type_code(DataType::I32) => map(ctx, p.len, i32::wrapping_neg),
        _ => map(ctx, p.len, i64::wrapping_neg),
    }
}

fixed_kernel!(
    /// `ABS` on `i32` and non-quantized `i16`, `NEG` on `i32` and `i64`.
    UNARY_INT,
    "rust/unary_int",
    unary_int_params,
    unary_int_eval
);

pod_struct! {
    /// Parameters of a quantized `ABS` or `RSQRT`.
    pub(crate) struct UnaryQParams {
        op: u32,
        dtype: u32,
        len: u32,
        input_zero_point: i32,
        output_zero_point: i32,
        /// Whether `ABS` rescales (the input and output scales differ).
        rescale: u32,
        multiplier: Multiplier,
        min: i32,
        max: i32,
        /// `RSQRT`: the output of each int8 input, at index `input + 128`.
        table: [i8; 256],
    }
}

/// TFLite `GetInvSqrtQuantizedMultiplierExp`: `1 / sqrt(input)` as a multiplier and a shift,
/// computed with gemmlowp fixed-point Newton-Raphson iterations.
fn inv_sqrt_multiplier_exp(mut input: i32, reverse_shift: i32) -> Multiplier {
    if input <= 1 {
        return Multiplier {
            multiplier: i32::MAX,
            shift: 0,
        };
    }
    let mut shift = 11;
    while input >= 1 << 29 {
        input /= 4;
        shift += 1;
    }
    let max_left_shift_bits = (input as u32).leading_zeros() - 1;
    let left_shift_bit_pairs = max_left_shift_bits / 2 - 1;
    shift -= left_shift_bit_pairs as i32;
    input <<= 2 * left_shift_bit_pairs;
    // gemmlowp `FixedPoint<int32_t, 3>`: products of two F3 are F6, of three F9; `Rescale<3>`
    // multiplies by 2^(bits - 3) with saturation.
    let mul = saturating_rounding_doubling_high_mul;
    let rescale = |x: i32, exponent: u32| {
        let threshold = (1i32 << (31 - exponent)) - 1;
        if x > threshold {
            i32::MAX
        } else if x < -threshold {
            i32::MIN
        } else {
            x << exponent
        }
    };
    let half_input = rounding_divide_by_pot(input >> 1, 1);
    let half_three = (1 << 28) + (1 << 27);
    let mut x = 1 << 28;
    for _ in 0..5 {
        let x3 = rescale(mul(mul(x, x), x), 6);
        x = rescale(mul(half_three, x).wrapping_sub(mul(half_input, x3)), 3);
    }
    // sqrt(2) / 2 in `FixedPoint<int32_t, 0>`.
    let mut multiplier = mul(x, 1_518_500_250);
    if shift < 0 {
        multiplier = ((multiplier as u32) << -shift) as i32;
        shift = 0;
    }
    Multiplier {
        multiplier,
        shift: shift * reverse_shift,
    }
}

/// The scale and zero point of a quantized tensor, with a zero point in the range of its type
/// (0 for int16).
fn quantization(tensor: &TensorDesc<'_>) -> Result<(f32, i32)> {
    let (scale, zero_point) = per_tensor(tensor)?;
    let (min, max) = int_range(tensor.dtype).ok_or_else(|| unsupported_type(tensor))?;
    let symmetric = tensor.dtype == DataType::I16;
    if !(min..=max).contains(&zero_point) || (symmetric && zero_point != 0) {
        return Err(unsupported_type(tensor));
    }
    Ok((scale, zero_point))
}

/// The shifts `MultiplyByQuantizedMultiplier` supports without overflowing an `int32_t`.
fn valid_shift(shift: i32) -> bool {
    (-31..=30).contains(&shift)
}

fn unary_q_params(op: &OpDesc<'_, '_>) -> Result<UnaryQParams> {
    let code = unary_code(op)?;
    let (x, y, len) = unary_tensors(op)?;
    let supported = match code {
        OP_ABS => matches!(x.dtype, DataType::I8 | DataType::I16) && x.quant.is_quantized(),
        OP_RSQRT => x.dtype == DataType::I8,
        _ => false,
    };
    if !supported {
        return Err(unsupported_type(x));
    }
    let (input_scale, input_zero_point) = quantization(x)?;
    let (output_scale, output_zero_point) = quantization(y)?;
    let (min, max) = int_range(x.dtype).ok_or_else(|| unsupported_type(x))?;
    let mut params = UnaryQParams {
        op: code,
        dtype: type_code(x.dtype),
        len: len as u32,
        input_zero_point,
        output_zero_point,
        rescale: u32::from(input_scale != output_scale),
        multiplier: Multiplier::default(),
        min,
        max,
        table: [0; 256],
    };
    let (input_scale, output_scale) = (f64::from(input_scale), f64::from(output_scale));
    if code == OP_ABS {
        if params.rescale != 0 {
            params.multiplier = quantize_multiplier(input_scale / output_scale);
            if !valid_shift(params.multiplier.shift) {
                return Err(unsupported(op));
            }
        }
        return Ok(params);
    }
    // RSQRT: shifts by 20 bits keep the intermediate value an integer.
    const SHIFT: i32 = 20;
    let m = quantize_multiplier(1.0 / (libm::sqrt(input_scale) * output_scale));
    if !valid_shift(m.shift - SHIFT) {
        return Err(unsupported(op));
    }
    for (slot, q) in params.table.iter_mut().zip(-128i32..) {
        let value = q - input_zero_point;
        *slot = if value < 0 {
            // Refused at evaluation.
            0
        } else if value == 0 {
            i8::MAX
        } else {
            let inv = inv_sqrt_multiplier_exp(value, -1);
            let data = multiply_by_quantized_multiplier(
                1,
                Multiplier {
                    multiplier: inv.multiplier,
                    shift: inv.shift + SHIFT,
                },
            );
            let out = multiply_by_quantized_multiplier(
                data,
                Multiplier {
                    multiplier: m.multiplier,
                    shift: m.shift - SHIFT,
                },
            );
            out.wrapping_add(output_zero_point).clamp(min, max) as i8
        };
    }
    Ok(params)
}

fn unary_q_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: UnaryQParams = *pod::read(ctx.persist)?;
    let (zin, zout) = (p.input_zero_point, p.output_zero_point);
    if p.op == OP_RSQRT {
        let x = in_slice::<i8>(ctx.inputs, 0, p.len as usize)?;
        // TFLite fails on negative values: nothing is written then.
        if x.iter().any(|&v| i32::from(v) < zin) {
            return Err(Error::invalid_argument(Detail::None));
        }
        return map(ctx, p.len, |v: i8| p.table[(i32::from(v) + 128) as usize]);
    }
    let abs = |v: i32| -> i32 {
        let value = (v - zin).wrapping_abs();
        let value = if p.rescale != 0 {
            multiply_by_quantized_multiplier(value, p.multiplier)
        } else {
            value
        };
        value.wrapping_add(zout).clamp(p.min, p.max)
    };
    if p.dtype == type_code(DataType::I16) {
        map(ctx, p.len, |v: i16| abs(i32::from(v)) as i16)
    } else {
        map(ctx, p.len, |v: i8| abs(i32::from(v)) as i8)
    }
}

fixed_kernel!(
    /// Quantized `ABS` (int8, int16) and `RSQRT` (int8).
    UNARY_Q,
    "rust/unary_q",
    unary_q_params,
    unary_q_eval
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_sqrt_matches_one_over_sqrt() {
        // The multiplier and shift encode 1 / sqrt(input) as `multiplier * 2^(shift - 31)`
        // (with `reverse_shift` 1), within the precision of the fixed-point iterations, for every
        // magnitude, without overflowing.
        for input in [2, 3, 10, 255, 1 << 20, (1 << 29) - 1, 1 << 29, i32::MAX] {
            let m = inv_sqrt_multiplier_exp(input, 1);
            let value = f64::from(m.multiplier) * libm::exp2(f64::from(-m.shift - 31));
            let expected = 1.0 / libm::sqrt(f64::from(input));
            assert!(
                (value - expected).abs() <= expected * 1e-5,
                "{input}: {value} != {expected}"
            );
        }
        assert_eq!(inv_sqrt_multiplier_exp(1, -1).multiplier, i32::MAX);
        assert_eq!(inv_sqrt_multiplier_exp(0, -1).multiplier, i32::MAX);
    }
}
