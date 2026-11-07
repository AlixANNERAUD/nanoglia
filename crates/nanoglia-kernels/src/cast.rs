//! `CAST` (TensorFlow Lite Micro `cast.cc`) between `bool`, `u8`, `i8`, `i16`, `i32`, `u32`,
//! `i64` and `f32`.
//!
//! Integers convert like C++'s `static_cast` (keeping the low bits for a narrower type), to
//! `bool` as `x != 0`, from `bool` as 0 or 1, and to `f32` with rounding to nearest. `f32`
//! converts to integers with TensorFlow Lite Micro's `SafeCast`: truncation toward zero,
//! saturation at the bounds of the type, NaN to 0, which is Rust's `as` (LiteRT 2.3 maps NaN to
//! the largest value instead). Quantization is not looked at, as in TFLite: the raw values are
//! converted.

use crate::util::{
    expect_arity, fixed_kernel, in_slice, input, out_slice, output, pod_struct, same_shape,
    type_code, unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::{DataType, Detail, Error, OpDesc, Result, pod};

pod_struct! {
    /// Parameters of `CAST`.
    pub(crate) struct CastParams {
        from: u32,
        to: u32,
        len: u32,
    }
}

/// The types `CAST` converts from and to.
const TYPES: [DataType; 8] = [
    DataType::Bool,
    DataType::U8,
    DataType::I8,
    DataType::I16,
    DataType::I32,
    DataType::U32,
    DataType::I64,
    DataType::F32,
];

fn cast_params(op: &OpDesc<'_, '_>) -> Result<CastParams> {
    expect_arity(op, 1, 1)?;
    let (x, y) = (input(op, 0)?, output(op, 0)?);
    for t in [x, y] {
        if !TYPES.contains(&t.dtype) {
            return Err(unsupported_type(t));
        }
    }
    let len = same_shape(x, y)?;
    Ok(CastParams {
        from: type_code(x.dtype),
        to: type_code(y.dtype),
        len: len as u32,
    })
}

/// A value of one of the types `CAST` supports, widened without loss.
#[derive(Clone, Copy)]
enum Value {
    Int(i64),
    Float(f32),
}

/// The input, typed.
enum Input<'a> {
    Bool(&'a [u8]),
    U8(&'a [u8]),
    I8(&'a [i8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    U32(&'a [u32]),
    I64(&'a [i64]),
    F32(&'a [f32]),
}

impl<'a> Input<'a> {
    fn new(inputs: &[&'a [u8]], code: u32, len: usize) -> Result<Self> {
        let is = |t: DataType| code == type_code(t);
        Ok(match () {
            _ if is(DataType::Bool) => Self::Bool(in_slice(inputs, 0, len)?),
            _ if is(DataType::U8) => Self::U8(in_slice(inputs, 0, len)?),
            _ if is(DataType::I8) => Self::I8(in_slice(inputs, 0, len)?),
            _ if is(DataType::I16) => Self::I16(in_slice(inputs, 0, len)?),
            _ if is(DataType::I32) => Self::I32(in_slice(inputs, 0, len)?),
            _ if is(DataType::U32) => Self::U32(in_slice(inputs, 0, len)?),
            _ if is(DataType::I64) => Self::I64(in_slice(inputs, 0, len)?),
            _ => Self::F32(in_slice(inputs, 0, len)?),
        })
    }

    fn get(&self, i: usize) -> Option<Value> {
        Some(match self {
            Self::Bool(x) => Value::Int(i64::from(*x.get(i)? != 0)),
            Self::U8(x) => Value::Int(i64::from(*x.get(i)?)),
            Self::I8(x) => Value::Int(i64::from(*x.get(i)?)),
            Self::I16(x) => Value::Int(i64::from(*x.get(i)?)),
            Self::I32(x) => Value::Int(i64::from(*x.get(i)?)),
            Self::U32(x) => Value::Int(i64::from(*x.get(i)?)),
            Self::I64(x) => Value::Int(*x.get(i)?),
            Self::F32(x) => Value::Float(*x.get(i)?),
        })
    }
}

/// The output, typed.
enum Output<'a> {
    Bool(&'a mut [u8]),
    U8(&'a mut [u8]),
    I8(&'a mut [i8]),
    I16(&'a mut [i16]),
    I32(&'a mut [i32]),
    U32(&'a mut [u32]),
    I64(&'a mut [i64]),
    F32(&'a mut [f32]),
}

/// `static_cast<T>` of an integer, or TFLite's `SafeCast<T>` of a float (Rust's `as` for both).
macro_rules! convert {
    ($v:expr, $t:ty) => {
        match $v {
            Value::Int(i) => i as $t,
            Value::Float(f) => f as $t,
        }
    };
}

impl Output<'_> {
    fn put(&mut self, i: usize, v: Value) -> Option<()> {
        match self {
            Self::Bool(y) => {
                *y.get_mut(i)? = u8::from(match v {
                    Value::Int(i) => i != 0,
                    Value::Float(f) => f != 0.0,
                });
            }
            Self::U8(y) => *y.get_mut(i)? = convert!(v, u8),
            Self::I8(y) => *y.get_mut(i)? = convert!(v, i8),
            Self::I16(y) => *y.get_mut(i)? = convert!(v, i16),
            Self::I32(y) => *y.get_mut(i)? = convert!(v, i32),
            Self::U32(y) => *y.get_mut(i)? = convert!(v, u32),
            Self::I64(y) => *y.get_mut(i)? = convert!(v, i64),
            Self::F32(y) => *y.get_mut(i)? = convert!(v, f32),
        }
        Some(())
    }
}

fn cast_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: CastParams = *pod::read(ctx.persist)?;
    let len = p.len as usize;
    let x = Input::new(ctx.inputs, p.from, len)?;
    let is = |t: DataType| p.to == type_code(t);
    let o = &mut *ctx.outputs;
    let mut y = match () {
        _ if is(DataType::Bool) => Output::Bool(out_slice(o, 0, len)?),
        _ if is(DataType::U8) => Output::U8(out_slice(o, 0, len)?),
        _ if is(DataType::I8) => Output::I8(out_slice(o, 0, len)?),
        _ if is(DataType::I16) => Output::I16(out_slice(o, 0, len)?),
        _ if is(DataType::I32) => Output::I32(out_slice(o, 0, len)?),
        _ if is(DataType::U32) => Output::U32(out_slice(o, 0, len)?),
        _ if is(DataType::I64) => Output::I64(out_slice(o, 0, len)?),
        _ => Output::F32(out_slice(o, 0, len)?),
    };
    for i in 0..len {
        x.get(i)
            .and_then(|v| y.put(i, v))
            .ok_or(Error::invalid_argument(Detail::None))?;
    }
    Ok(())
}

fixed_kernel!(
    /// `CAST` between `bool`, `u8`, `i8`, `i16`, `i32`, `u32`, `i64` and `f32`.
    CAST,
    "rust/cast",
    cast_params,
    cast_eval
);
