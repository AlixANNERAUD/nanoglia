//! Comparisons (`EQUAL`, `NOT_EQUAL`, `LESS`, `LESS_EQUAL`, `GREATER`, `GREATER_EQUAL`; TFLite
//! `comparisons.cc`, reference `ComparisonWithScaling` and `BroadcastComparison4DSlow*`) and
//! `SELECT_V2` (TFLite `select.cc`, reference `BroadcastSelect5DSlow`), with broadcasting up to
//! rank 6.
//!
//! Quantized int8 and uint8 inputs are compared after rescaling each to its own scale, as TFLite
//! does: `MultiplyByQuantizedMultiplierSmallerThanOneExp((x - zero_point) << 8,
//! QuantizeMultiplierSmallerThanOneExp(scale))`. `SELECT_V2` only moves bytes: like TFLite, it
//! does not look at quantization.

use crate::util::{
    Broadcast, check_aligned, expect_arity, fixed_kernel, in_slice, input, int_range, out_slice,
    output, per_tensor, pod_struct, shape_mismatch, type_code, unsupported, unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod::{self, Pod};
use nanoglia_core::quant::{
    Multiplier, multiply_by_quantized_multiplier, quantize_multiplier_smaller_than_one,
};
use nanoglia_core::{BuiltinOp, DataType, MAX_RANK, OpDesc, Result, TensorDesc};

const OP_EQUAL: u32 = 0;
const OP_NOT_EQUAL: u32 = 1;
const OP_LESS: u32 = 2;
const OP_LESS_EQUAL: u32 = 3;
const OP_GREATER: u32 = 4;
const OP_GREATER_EQUAL: u32 = 5;

/// TFLite's `left_shift` of quantized comparisons.
const LEFT_SHIFT: u32 = 8;

pod_struct! {
    /// Parameters of a comparison (TFLite `ComparisonParams`).
    pub(crate) struct ComparisonParams {
        broadcast: Broadcast,
        op: u32,
        dtype: u32,
        /// Whether the inputs are rescaled (int8, uint8).
        quantized: u32,
        /// Negated zero points of the inputs.
        input1_offset: i32,
        input2_offset: i32,
        /// Rescaling of each input to its own scale.
        input1: Multiplier,
        input2: Multiplier,
    }
}

/// The negated zero point and the multiplier of a quantized comparison input.
fn rescaling(op: &OpDesc<'_, '_>, t: &TensorDesc<'_>) -> Result<(i32, Multiplier)> {
    let (scale, zero_point) = per_tensor(t)?;
    let (min, max) = int_range(t.dtype).ok_or_else(|| unsupported_type(t))?;
    if !(min..=max).contains(&zero_point) {
        return Err(unsupported_type(t));
    }
    // TensorFlow Lite Micro's `QuantizeMultiplierSmallerThanOneExp` requires a scale below 1.
    let multiplier =
        quantize_multiplier_smaller_than_one(f64::from(scale)).ok_or_else(|| unsupported(op))?;
    Ok((-zero_point, multiplier))
}

fn comparison_params(op: &OpDesc<'_, '_>) -> Result<ComparisonParams> {
    expect_arity(op, 2, 1)?;
    let code = match op.builtin() {
        Some(BuiltinOp::EQUAL) => OP_EQUAL,
        Some(BuiltinOp::NOT_EQUAL) => OP_NOT_EQUAL,
        Some(BuiltinOp::LESS) => OP_LESS,
        Some(BuiltinOp::LESS_EQUAL) => OP_LESS_EQUAL,
        Some(BuiltinOp::GREATER) => OP_GREATER,
        Some(BuiltinOp::GREATER_EQUAL) => OP_GREATER_EQUAL,
        _ => return Err(unsupported(op)),
    };
    let (a, b, y) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    if b.dtype != a.dtype {
        return Err(unsupported_type(b));
    }
    if y.dtype != DataType::Bool {
        return Err(unsupported_type(y));
    }
    let equality = matches!(code, OP_EQUAL | OP_NOT_EQUAL);
    let quantized = match a.dtype {
        DataType::F32 | DataType::I32 | DataType::I64 => false,
        DataType::Bool if equality => false,
        DataType::I8 | DataType::U8 => true,
        _ => return Err(unsupported_type(a)),
    };
    let mut params = ComparisonParams {
        broadcast: Broadcast::new(a, b, y)?,
        op: code,
        dtype: type_code(a.dtype),
        quantized: u32::from(quantized),
        input1_offset: 0,
        input2_offset: 0,
        input1: Multiplier::default(),
        input2: Multiplier::default(),
    };
    if quantized {
        (params.input1_offset, params.input1) = rescaling(op, a)?;
        (params.input2_offset, params.input2) = rescaling(op, b)?;
    }
    for t in [a, b] {
        check_aligned(t, t.dtype.size().unwrap_or(1))?;
    }
    Ok(params)
}

/// Writes `cmp(a, b)` as a TFLite bool (one byte, 0 or 1) for every output element.
fn compare<T: Pod>(
    ctx: &mut EvalCtx<'_, '_>,
    b: &Broadcast,
    cmp: impl Fn(T, T) -> bool,
) -> Result<()> {
    let x = in_slice::<T>(ctx.inputs, 0, b.len1 as usize)?;
    let y = in_slice::<T>(ctx.inputs, 1, b.len2 as usize)?;
    let out = out_slice::<u8>(ctx.outputs, 0, b.len as usize)?;
    b.for_each(|o, i, j| out[o] = u8::from(cmp(x[i], y[j])));
    Ok(())
}

/// A comparison on one type, with TFLite's operators.
fn compare_as<T: Pod + PartialOrd>(ctx: &mut EvalCtx<'_, '_>, p: &ComparisonParams) -> Result<()> {
    let b = &p.broadcast;
    match p.op {
        OP_EQUAL => compare::<T>(ctx, b, |x, y| x == y),
        OP_NOT_EQUAL => compare::<T>(ctx, b, |x, y| x != y),
        OP_LESS => compare::<T>(ctx, b, |x, y| x < y),
        OP_LESS_EQUAL => compare::<T>(ctx, b, |x, y| x <= y),
        OP_GREATER => compare::<T>(ctx, b, |x, y| x > y),
        _ => compare::<T>(ctx, b, |x, y| x >= y),
    }
}

/// A quantized comparison: both inputs rescaled, then compared as `i32`.
fn compare_quantized<T: Pod + Into<i32>>(
    ctx: &mut EvalCtx<'_, '_>,
    p: &ComparisonParams,
) -> Result<()> {
    let scaled = |v: T, offset: i32, m: Multiplier| {
        multiply_by_quantized_multiplier((offset + v.into()).wrapping_mul(1 << LEFT_SHIFT), m)
    };
    let cmp: fn(i32, i32) -> bool = match p.op {
        OP_EQUAL => |x, y| x == y,
        OP_NOT_EQUAL => |x, y| x != y,
        OP_LESS => |x, y| x < y,
        OP_LESS_EQUAL => |x, y| x <= y,
        OP_GREATER => |x, y| x > y,
        _ => |x, y| x >= y,
    };
    compare::<T>(ctx, &p.broadcast, |x, y| {
        cmp(
            scaled(x, p.input1_offset, p.input1),
            scaled(y, p.input2_offset, p.input2),
        )
    })
}

fn comparison_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: ComparisonParams = *pod::read(ctx.persist)?;
    let is = |t: DataType| p.dtype == type_code(t);
    if p.quantized != 0 {
        return if is(DataType::I8) {
            compare_quantized::<i8>(ctx, &p)
        } else {
            compare_quantized::<u8>(ctx, &p)
        };
    }
    match () {
        // Bools are bytes holding 0 or 1, compared as such.
        _ if is(DataType::Bool) => compare_as::<u8>(ctx, &p),
        _ if is(DataType::F32) => compare_as::<f32>(ctx, &p),
        _ if is(DataType::I32) => compare_as::<i32>(ctx, &p),
        _ => compare_as::<i64>(ctx, &p),
    }
}

fixed_kernel!(
    /// `EQUAL`, `NOT_EQUAL`, `LESS`, `LESS_EQUAL`, `GREATER` and `GREATER_EQUAL` on `f32`, `i32`,
    /// `i64`, quantized `i8` and `u8`, and (equalities) `bool`, with broadcasting.
    COMPARISON,
    "rust/comparison",
    comparison_params,
    comparison_eval
);

pod_struct! {
    /// Parameters of `SELECT_V2`: how the condition broadcasts with each value input.
    pub(crate) struct SelectParams {
        /// The condition and `x`.
        with_x: Broadcast,
        /// The condition and `y`.
        with_y: Broadcast,
        /// Element size in bytes.
        size: u32,
    }
}

/// The dimensions of `t` left-padded with ones to [`MAX_RANK`].
fn padded(t: &TensorDesc<'_>) -> [i32; MAX_RANK] {
    let mut d = [1i32; MAX_RANK];
    let s = t.dims.as_slice();
    d[MAX_RANK - s.len()..].copy_from_slice(s);
    d
}

fn select_params(op: &OpDesc<'_, '_>) -> Result<SelectParams> {
    expect_arity(op, 3, 1)?;
    let (c, x, y, out) = (input(op, 0)?, input(op, 1)?, input(op, 2)?, output(op, 0)?);
    if c.dtype != DataType::Bool {
        return Err(unsupported_type(c));
    }
    if y.dtype != x.dtype || out.dtype != x.dtype {
        return Err(unsupported_type(out));
    }
    // The types of LiteRT's `select.cc` (TensorFlow Lite Micro: f32, int8, int16).
    if !matches!(
        x.dtype,
        DataType::F32
            | DataType::Bool
            | DataType::U8
            | DataType::I8
            | DataType::I16
            | DataType::I32
            | DataType::U32
            | DataType::I64
    ) {
        return Err(unsupported_type(x));
    }
    let size = x.dtype.size().ok_or_else(|| unsupported_type(x))?;
    let with_x = Broadcast::within(c, x, out)?;
    let with_y = Broadcast::within(c, y, out)?;
    // The output has exactly the broadcast shape of the three inputs.
    if [c, x, y].iter().any(|t| t.dims.rank() > out.dims.rank()) {
        return Err(shape_mismatch(out));
    }
    let (pc, px, py, po) = (padded(c), padded(x), padded(y), padded(out));
    for axis in 0..MAX_RANK {
        if po[axis] != 1 && ![pc[axis], px[axis], py[axis]].contains(&po[axis]) {
            return Err(shape_mismatch(out));
        }
    }
    for t in [x, y] {
        check_aligned(t, size)?;
    }
    Ok(SelectParams {
        with_x,
        with_y,
        size: size as u32,
    })
}

/// Writes `x` where the condition holds, then `y` where it does not.
fn select_as<T: Pod>(ctx: &mut EvalCtx<'_, '_>, p: &SelectParams) -> Result<()> {
    let c = in_slice::<u8>(ctx.inputs, 0, p.with_x.len1 as usize)?;
    let x = in_slice::<T>(ctx.inputs, 1, p.with_x.len2 as usize)?;
    let y = in_slice::<T>(ctx.inputs, 2, p.with_y.len2 as usize)?;
    let out = out_slice::<T>(ctx.outputs, 0, p.with_x.len as usize)?;
    p.with_x.for_each(|o, i, j| {
        if c[i] != 0 {
            out[o] = x[j];
        }
    });
    p.with_y.for_each(|o, i, j| {
        if c[i] == 0 {
            out[o] = y[j];
        }
    });
    Ok(())
}

fn select_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: SelectParams = *pod::read(ctx.persist)?;
    match p.size {
        1 => select_as::<u8>(ctx, &p),
        2 => select_as::<u16>(ctx, &p),
        4 => select_as::<u32>(ctx, &p),
        _ => select_as::<u64>(ctx, &p),
    }
}

fixed_kernel!(
    /// `SELECT_V2` of `f32`, `bool`, `u8`, `i8`, `i16`, `i32`, `u32` and `i64`, with
    /// broadcasting.
    SELECT_V2,
    "rust/select_v2",
    select_params,
    select_eval
);
