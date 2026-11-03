//! Binary element-wise operators with broadcasting up to [`nanonn_core::MAX_RANK`] dimensions:
//! `ADD`, `SUB`, `MUL` on `f32` (TFLite `BroadcastAdd6DSlow` & co.) and `MAXIMUM`, `MINIMUM` on
//! any numeric type (TFLite `maximum_minimum.cc`).

use crate::util::{
    Broadcast, clamp_f32, expect_arity, fixed_kernel, float_activation, in_slice, input, out_slice,
    output, pod_struct, type_code, unsupported, unsupported_type,
};
use nanonn_core::backend::EvalCtx;
use nanonn_core::pod::{self, Pod};
use nanonn_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result};

const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_MUL: u32 = 2;
const OP_MAXIMUM: u32 = 3;
const OP_MINIMUM: u32 = 4;

pod_struct! {
    /// Parameters of a binary element-wise operator.
    pub(crate) struct BinaryParams {
        broadcast: Broadcast,
        op: u32,
        dtype: u32,
        min: f32,
        max: f32,
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
        _ => return Err(unsupported(op)),
    };
    let (mut min, mut max) = (f32::MIN, f32::MAX);
    if code <= OP_MUL {
        if out.dtype != DataType::F32 {
            return Err(unsupported_type(out));
        }
        let activation = match op.options {
            OpOptions::Elementwise { activation } => activation,
            OpOptions::None => nanonn_core::Activation::None,
            _ => return Err(unsupported(op)),
        };
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
    match p.op {
        OP_ADD => apply::<f32>(ctx, &p.broadcast, |a, b| clamp_f32(a + b, min, max)),
        OP_SUB => apply::<f32>(ctx, &p.broadcast, |a, b| clamp_f32(a - b, min, max)),
        OP_MUL => apply::<f32>(ctx, &p.broadcast, |a, b| clamp_f32(a * b, min, max)),
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
    /// `ADD`, `SUB` and `MUL` on `f32`; `MAXIMUM` and `MINIMUM` on `f32` and integers.
    BINARY,
    "rust/binary",
    binary_params,
    binary_eval
);
