//! Binary element-wise operators with broadcasting up to [`nanoglia_core::MAX_RANK`] dimensions:
//! `ADD`, `SUB`, `MUL` on `f32` (TFLite `BroadcastAdd6DSlow` & co.), on int8 and on int16
//! (16-bit activations, symmetric), and `MAXIMUM`, `MINIMUM` on any numeric type (TFLite
//! `maximum_minimum.cc`). `DIV` on `f32` and `i32` and `SQUARED_DIFFERENCE` on `f32`, `i32`,
//! int8 and int16 follow TensorFlow Lite Micro's `div.cc` and `squared_difference_common.cc`.
//!
//! The quantized operators port the `Prepare` of TFLite's `add.cc`, `sub.cc` and `mul.cc` and the
//! reference kernels `reference_integer_ops::BroadcastAdd6DSlow`,
//! `reference_ops::BroadcastQuantSubSlow` and `reference_integer_ops::BroadcastMul6DSlow`:
//! `ADD` and `SUB` rescale both inputs to a common scale with `left_shift = 20` (int8) or `15`
//! (int16) bits of headroom and `QuantizeMultiplierSmallerThanOneExp` multipliers; `MUL`
//! requantizes the product of the inputs. The `MUL` multiplier is computed in `f64` like TensorFlow
//! Lite Micro and the CMSIS-NN backend (LiteRT rounds the product of the scales to `f32` first).
//!
//! Quantized `SQUARED_DIFFERENCE` rescales both inputs like `ADD` (with `left_shift = 7` on int8
//! and `0` on int16), squares their difference in `i32` and requantizes it. Quantized `DIV` is
//! not supported: TensorFlow Lite Micro's `DivElementwise` divides by powers of two above 2^31
//! for small quotients, which C++ leaves undefined and CPUs compute differently.
//!
//! TFLite runs int16 `ADD` and `SUB` through a different, power-of-two kernel when all three
//! scales are powers of two and the operator's `pot_scale_int16` option (true by default) is
//! set; nanoglia does not decode that option, so these operators are left unsupported.

use crate::util::{
    Broadcast, clamp_f32, expect_arity, fixed_kernel, float_activation, in_slice, input, out_slice,
    output, per_tensor, pod_struct, quantized_activation, type_code, unsupported, unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod::{self, Pod};
use nanoglia_core::quant::{self, Multiplier, multiply_by_quantized_multiplier};
use nanoglia_core::{Activation, BuiltinOp, DataType, Detail, Error, OpDesc, OpOptions, Result};

const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_MUL: u32 = 2;
const OP_MAXIMUM: u32 = 3;
const OP_MINIMUM: u32 = 4;
const OP_DIV: u32 = 5;
const OP_SQUARED_DIFFERENCE: u32 = 6;

pod_struct! {
    /// Parameters of a binary element-wise operator.
    pub(crate) struct BinaryParams {
        broadcast: Broadcast,
        op: u32,
        dtype: u32,
        /// Clamping range of the fused activation (`f32`).
        min: f32,
        max: f32,
        /// Clamping range of the fused activation (`i32` `DIV`).
        imin: i32,
        imax: i32,
    }
}

fn binary_params(op: &OpDesc<'_, '_>) -> Result<BinaryParams> {
    expect_arity(op, 2, 1)?;
    let (a, b, out) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    if a.dtype != out.dtype || b.dtype != out.dtype {
        return Err(unsupported_type(out));
    }
    let code = match op.builtin() {
        Some(BuiltinOp::ADD) => OP_ADD,
        Some(BuiltinOp::SUB) => OP_SUB,
        Some(BuiltinOp::MUL) => OP_MUL,
        Some(BuiltinOp::MAXIMUM) => OP_MAXIMUM,
        Some(BuiltinOp::MINIMUM) => OP_MINIMUM,
        Some(BuiltinOp::DIV) => OP_DIV,
        Some(BuiltinOp::SQUARED_DIFFERENCE) => OP_SQUARED_DIFFERENCE,
        _ => return Err(unsupported(op)),
    };
    let (mut min, mut max) = (f32::MIN, f32::MAX);
    let (mut imin, mut imax) = (i32::MIN, i32::MAX);
    let activation = match op.options {
        OpOptions::Elementwise { activation } => activation,
        OpOptions::None => Activation::None,
        _ => return Err(unsupported(op)),
    };
    if matches!(code, OP_DIV | OP_SQUARED_DIFFERENCE) {
        // f32 and plain int32 (TFLite refuses quantized int32 `DIV`).
        let plain_i32 =
            out.dtype == DataType::I32 && !a.quant.is_quantized() && !b.quant.is_quantized();
        if out.dtype != DataType::F32 && !plain_i32 {
            return Err(unsupported_type(out));
        }
        if code == OP_DIV {
            (min, max) = float_activation(op, activation)?;
            // TFLite `CalculateActivationRange<int32_t>`.
            (imin, imax) = match activation {
                Activation::None => (i32::MIN, i32::MAX),
                Activation::Relu => (0, i32::MAX),
                Activation::Relu6 => (0, 6),
                Activation::ReluN1To1 => (-1, 1),
                _ => return Err(unsupported(op)),
            };
        } else if activation != Activation::None {
            return Err(unsupported(op));
        }
    } else if code <= OP_MUL {
        if out.dtype != DataType::F32 {
            return Err(unsupported_type(out));
        }
        (min, max) = float_activation(op, activation)?;
    } else if !matches!(
        out.dtype,
        DataType::F32 | DataType::I8 | DataType::U8 | DataType::I16 | DataType::I32 | DataType::I64
    ) {
        return Err(unsupported_type(out));
    }
    for t in [a, b] {
        crate::util::check_aligned(t, out.dtype.size().unwrap_or(1))?;
    }
    Ok(BinaryParams {
        broadcast: Broadcast::new(a, b, out)?,
        op: code,
        dtype: type_code(out.dtype),
        min,
        max,
        imin,
        imax,
    })
}

/// Applies `f` with broadcasting.
fn apply<T: Pod>(ctx: &mut EvalCtx<'_, '_>, b: &Broadcast, f: impl Fn(T, T) -> T) -> Result<()> {
    let x = in_slice::<T>(ctx.inputs, 0, b.len1 as usize)?;
    let y = in_slice::<T>(ctx.inputs, 1, b.len2 as usize)?;
    let out = out_slice::<T>(ctx.outputs, 0, b.len as usize)?;
    b.for_each(|o, i, j| out[o] = f(x[i], y[j]));
    Ok(())
}

/// `MAXIMUM` / `MINIMUM` on one type, with TFLite's comparisons.
fn min_max<T: Pod + PartialOrd>(ctx: &mut EvalCtx<'_, '_>, p: &BinaryParams) -> Result<()> {
    if p.op == OP_MAXIMUM {
        apply::<T>(ctx, &p.broadcast, |a, b| if a > b { a } else { b })
    } else {
        apply::<T>(ctx, &p.broadcast, |a, b| if a < b { a } else { b })
    }
}

fn binary_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: BinaryParams = *pod::read(ctx.persist)?;
    let (min, max) = (p.min, p.max);
    let float = p.dtype == type_code(DataType::F32);
    match p.op {
        OP_ADD => apply::<f32>(ctx, &p.broadcast, |a, b| clamp_f32(a + b, min, max)),
        OP_SUB => apply::<f32>(ctx, &p.broadcast, |a, b| clamp_f32(a - b, min, max)),
        OP_MUL => apply::<f32>(ctx, &p.broadcast, |a, b| clamp_f32(a * b, min, max)),
        OP_DIV if float => apply::<f32>(ctx, &p.broadcast, |a, b| clamp_f32(a / b, min, max)),
        OP_DIV => {
            // TFLite fails on a zero denominator, anywhere in the second input.
            let denominators = in_slice::<i32>(ctx.inputs, 1, p.broadcast.len2 as usize)?;
            if denominators.contains(&0) {
                return Err(Error::invalid_argument(Detail::None));
            }
            let (imin, imax) = (p.imin, p.imax);
            apply::<i32>(ctx, &p.broadcast, |a, b| {
                a.wrapping_div(b).clamp(imin, imax)
            })
        }
        OP_SQUARED_DIFFERENCE if float => apply::<f32>(ctx, &p.broadcast, |a, b| {
            let d = a - b;
            d * d
        }),
        OP_SQUARED_DIFFERENCE => apply::<i32>(ctx, &p.broadcast, |a, b| {
            let d = a.wrapping_sub(b);
            d.wrapping_mul(d)
        }),
        _ => match p.dtype {
            t if t == type_code(DataType::F32) => min_max::<f32>(ctx, &p),
            t if t == type_code(DataType::I8) => min_max::<i8>(ctx, &p),
            t if t == type_code(DataType::U8) => min_max::<u8>(ctx, &p),
            t if t == type_code(DataType::I16) => min_max::<i16>(ctx, &p),
            t if t == type_code(DataType::I32) => min_max::<i32>(ctx, &p),
            _ => min_max::<i64>(ctx, &p),
        },
    }
}

fixed_kernel!(
    /// `ADD`, `SUB` and `MUL` on `f32`; `DIV` and `SQUARED_DIFFERENCE` on `f32` and `i32`;
    /// `MAXIMUM` and `MINIMUM` on `f32` and integers.
    BINARY,
    "rust/binary",
    binary_params,
    binary_eval
);

pod_struct! {
    /// Parameters of a quantized `ADD`, `SUB` or `MUL` (TFLite `ArithmeticParams`).
    pub(crate) struct QuantizedBinaryParams {
        broadcast: Broadcast,
        op: u32,
        dtype: u32,
        /// Negated zero point of the first input.
        input1_offset: i32,
        /// Negated zero point of the second input.
        input2_offset: i32,
        /// Zero point of the output.
        output_offset: i32,
        /// Headroom of the inputs of `ADD` and `SUB`.
        left_shift: i32,
        /// Rescaling of the first input (`ADD`, `SUB`).
        input1: Multiplier,
        /// Rescaling of the second input (`ADD`, `SUB`).
        input2: Multiplier,
        /// Rescaling of the result.
        output: Multiplier,
        /// Clamping range of the fused activation.
        min: i32,
        max: i32,
    }
}

fn quantized_binary_params(op: &OpDesc<'_, '_>) -> Result<QuantizedBinaryParams> {
    expect_arity(op, 2, 1)?;
    let (a, b, out) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    if !matches!(out.dtype, DataType::I8 | DataType::I16) {
        return Err(unsupported_type(out));
    }
    if a.dtype != out.dtype || b.dtype != out.dtype {
        return Err(unsupported_type(out));
    }
    let code = match op.builtin() {
        Some(BuiltinOp::ADD) => OP_ADD,
        Some(BuiltinOp::SUB) => OP_SUB,
        Some(BuiltinOp::MUL) => OP_MUL,
        Some(BuiltinOp::SQUARED_DIFFERENCE) => OP_SQUARED_DIFFERENCE,
        _ => return Err(unsupported(op)),
    };
    let activation = match op.options {
        OpOptions::Elementwise { activation } => activation,
        OpOptions::None => Activation::None,
        _ => return Err(unsupported(op)),
    };
    let (s1, z1) = per_tensor(a)?;
    let (s2, z2) = per_tensor(b)?;
    let (so, zo) = per_tensor(out)?;
    let int16 = out.dtype == DataType::I16;
    for (t, z) in [(a, z1), (b, z2), (out, zo)] {
        // int16 activations are symmetric; int8 zero points must be int8 values.
        let valid = if int16 {
            z == 0
        } else {
            (-128..=127).contains(&z)
        };
        if !valid {
            return Err(unsupported_type(t));
        }
    }
    if int16
        && matches!(code, OP_ADD | OP_SUB)
        && [s1, s2, so].into_iter().all(quant::is_near_power_of_two)
    {
        // The power-of-two kernel of TFLite, selected by an option nanoglia does not decode.
        return Err(unsupported(op));
    }
    let (min, max) = quantized_activation(op, activation, out, (so, zo))?;
    let mut params = QuantizedBinaryParams {
        broadcast: Broadcast::new(a, b, out)?,
        op: code,
        dtype: type_code(out.dtype),
        input1_offset: -z1,
        input2_offset: -z2,
        output_offset: zo,
        left_shift: 0,
        input1: Multiplier::default(),
        input2: Multiplier::default(),
        output: Multiplier::default(),
        min,
        max,
    };
    if code == OP_SQUARED_DIFFERENCE {
        // TensorFlow Lite Micro's `PrepareQuantized` (`squared_difference_common.cc`).
        if activation != Activation::None {
            return Err(unsupported(op));
        }
        params.left_shift = if int16 { 0 } else { 7 };
        let twice_max_input_scale = 2.0 * f64::from(if s1 < s2 { s2 } else { s1 });
        let smaller =
            |real| quant::quantize_multiplier_smaller_than_one(real).ok_or(unsupported(op));
        params.input1 = smaller(f64::from(s1) / twice_max_input_scale)?;
        params.input2 = smaller(f64::from(s2) / twice_max_input_scale)?;
        let real_output = twice_max_input_scale * twice_max_input_scale
            / (f64::from(1u32 << (2 * params.left_shift)) * f64::from(so));
        params.output = quant::quantize_multiplier(real_output);
        if params.output.shift > 30 {
            return Err(unsupported(op));
        }
    } else if code == OP_MUL {
        let real = f64::from(s1) * f64::from(s2) / f64::from(so);
        params.output = quant::quantize_multiplier(real);
        // `MultiplyByQuantizedMultiplier` shifts an `int32_t` left by positive shifts.
        if params.output.shift > 30 {
            return Err(unsupported(op));
        }
    } else {
        params.left_shift = if int16 { 15 } else { 20 };
        let twice_max_input_scale = 2.0 * f64::from(if s1 < s2 { s2 } else { s1 });
        let real1 = f64::from(s1) / twice_max_input_scale;
        let real2 = f64::from(s2) / twice_max_input_scale;
        let real_output =
            twice_max_input_scale / (f64::from(1u32 << params.left_shift) * f64::from(so));
        let smaller =
            |real| quant::quantize_multiplier_smaller_than_one(real).ok_or(unsupported(op));
        params.input1 = smaller(real1)?;
        params.input2 = smaller(real2)?;
        // `SUB` accepts output multipliers above 1 (very narrow output ranges); the reference
        // kernel then shifts left, which only single-rounding builds of TFLite define.
        params.output = if code == OP_SUB && real_output > 1.0 && quant::SINGLE_ROUNDING {
            quant::quantize_multiplier_greater_than_one(real_output).ok_or(unsupported(op))?
        } else {
            smaller(real_output)?
        };
    }
    for t in [a, b] {
        crate::util::check_aligned(t, out.dtype.size().unwrap_or(1))?;
    }
    Ok(params)
}

/// An integer type of quantized activations.
trait Quantized: Pod {
    fn to_i32(self) -> i32;
    /// Truncates `value`, which the caller clamped to the range of the type.
    fn from_i32(value: i32) -> Self;
}

impl Quantized for i8 {
    fn to_i32(self) -> i32 {
        self.into()
    }
    fn from_i32(value: i32) -> Self {
        value as i8
    }
}

impl Quantized for i16 {
    fn to_i32(self) -> i32 {
        self.into()
    }
    fn from_i32(value: i32) -> Self {
        value as i16
    }
}

/// Applies a quantized binary operator on one type, with broadcasting.
fn quantized_apply<T: Quantized>(
    ctx: &mut EvalCtx<'_, '_>,
    p: &QuantizedBinaryParams,
) -> Result<()> {
    let clamp = |x: i32| T::from_i32(x.max(p.min).min(p.max));
    if p.op == OP_MUL {
        return apply::<T>(ctx, &p.broadcast, |a, b| {
            let product = (p.input1_offset + a.to_i32()).wrapping_mul(p.input2_offset + b.to_i32());
            clamp(multiply_by_quantized_multiplier(product, p.output).wrapping_add(p.output_offset))
        });
    }
    let sub = p.op == OP_SUB;
    let unit = 1i32 << p.left_shift;
    if p.op == OP_SQUARED_DIFFERENCE {
        return apply::<T>(ctx, &p.broadcast, |a, b| {
            let shifted1 = (p.input1_offset + a.to_i32()).wrapping_mul(unit);
            let shifted2 = (p.input2_offset + b.to_i32()).wrapping_mul(unit);
            let raw = multiply_by_quantized_multiplier(shifted1, p.input1)
                .wrapping_sub(multiply_by_quantized_multiplier(shifted2, p.input2));
            let squared = raw.wrapping_mul(raw);
            clamp(multiply_by_quantized_multiplier(squared, p.output).wrapping_add(p.output_offset))
        });
    }
    apply::<T>(ctx, &p.broadcast, |a, b| {
        let shifted1 = (p.input1_offset + a.to_i32()).wrapping_mul(unit);
        let shifted2 = (p.input2_offset + b.to_i32()).wrapping_mul(unit);
        let scaled1 = multiply_by_quantized_multiplier(shifted1, p.input1);
        let scaled2 = multiply_by_quantized_multiplier(shifted2, p.input2);
        let raw = if sub {
            scaled1.wrapping_sub(scaled2)
        } else {
            scaled1.wrapping_add(scaled2)
        };
        clamp(multiply_by_quantized_multiplier(raw, p.output).wrapping_add(p.output_offset))
    })
}

fn quantized_binary_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: QuantizedBinaryParams = *pod::read(ctx.persist)?;
    if p.dtype == type_code(DataType::I16) {
        quantized_apply::<i16>(ctx, &p)
    } else {
        quantized_apply::<i8>(ctx, &p)
    }
}

fixed_kernel!(
    /// `ADD`, `SUB`, `MUL` and `SQUARED_DIFFERENCE` on int8 and int16, with broadcasting.
    QUANTIZED_BINARY,
    "rust/binary_q",
    quantized_binary_params,
    quantized_binary_eval
);
