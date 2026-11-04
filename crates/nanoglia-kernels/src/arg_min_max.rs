//! `ARG_MAX` and `ARG_MIN` (TFLite `arg_min_max.cc`, reference `ArgMinMax`), with a constant axis
//! and `i32` or `i64` indices.
//!
//! The index of the first extreme value along the axis is kept: a later value replaces the current
//! one only if it is strictly greater (`ARG_MAX`) or smaller (`ARG_MIN`). For `f32`, NaN values
//! are skipped (an all-NaN slice gives index 0), like the reference comparison. Quantized inputs
//! are compared as raw integers, as in TFLite: the scale is positive, so the order is the same.
//!
//! TFLite accepts `f32`, `u8`, `i8`, `i32` and `bool` inputs; nanoglia also accepts `i16`
//! (16x8 models), compared the same way.

use crate::util::{
    constant, dims_product, elements, fixed_kernel, in_slice, input, out_slice, output, pod_struct,
    read_index, shape_mismatch, type_code, unsupported, unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod::Pod;
use nanoglia_core::{BuiltinOp, DataType, MAX_RANK, OpDesc, OpOptions, Result, pod};

pod_struct! {
    /// Parameters of `ARG_MAX` and `ARG_MIN`.
    pub(crate) struct ArgMinMaxParams {
        /// Product of the dimensions before the axis.
        outer: u32,
        /// Size of the reduced axis (at least 1).
        axis_size: u32,
        /// Product of the dimensions after the axis.
        inner: u32,
        /// The input type ([`type_code`]).
        dtype: u32,
        /// Whether the output holds `i64` indices (`i32` otherwise).
        wide: u32,
        /// Whether the operator is `ARG_MAX` (`ARG_MIN` otherwise).
        max: u32,
    }
}

impl ArgMinMaxParams {
    fn input_len(&self) -> usize {
        self.outer as usize * self.axis_size as usize * self.inner as usize
    }

    fn output_len(&self) -> usize {
        self.outer as usize * self.inner as usize
    }
}

fn arg_min_max_params(op: &OpDesc<'_, '_>) -> Result<ArgMinMaxParams> {
    let OpOptions::ArgMinMax { output_type } = op.options else {
        return Err(unsupported(op));
    };
    let max = match op.builtin() {
        Some(BuiltinOp::ARG_MAX) => true,
        Some(BuiltinOp::ARG_MIN) => false,
        _ => return Err(unsupported(op)),
    };
    if op.inputs.len() != 2 || op.outputs.len() != 1 {
        return Err(unsupported(op));
    }
    let (data, axis, out) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    let supported = matches!(
        data.dtype,
        DataType::F32
            | DataType::U8
            | DataType::I8
            | DataType::I16
            | DataType::I32
            | DataType::Bool
    );
    if !supported {
        return Err(unsupported_type(data));
    }
    // TFLite gives the output the type of the `output_type` option, `i32` or `i64`.
    let wide = match output_type {
        DataType::I32 => false,
        DataType::I64 => true,
        _ => return Err(unsupported(op)),
    };
    if out.dtype != output_type {
        return Err(unsupported_type(out));
    }
    elements(data)?;
    elements(out)?;

    // The axis: one constant int32 or int64 element, made non-negative (`ReadAndNormalizeAxis`).
    let dims = data.dims.as_slice();
    let rank = dims.len();
    if rank == 0
        || !matches!(axis.dtype, DataType::I32 | DataType::I64)
        || axis.dims.num_elements() != Some(1)
    {
        return Err(unsupported(op));
    }
    let value = read_index(axis, constant(op, axis)?, 0).ok_or_else(|| unsupported(op))?;
    let value = if value < 0 {
        value + rank as i64
    } else {
        value
    };
    if !(0..rank as i64).contains(&value) {
        return Err(unsupported(op));
    }
    let axis = value as usize;
    // TFLite refuses an empty axis.
    if dims[axis] <= 0 {
        return Err(shape_mismatch(data));
    }

    // The output has the input dimensions without the axis.
    let mut expected = [0i32; MAX_RANK];
    let mut n = 0;
    for (i, &d) in dims.iter().enumerate() {
        if i != axis {
            expected[n] = d;
            n += 1;
        }
    }
    if out.dims.as_slice() != &expected[..n] {
        return Err(shape_mismatch(out));
    }
    Ok(ArgMinMaxParams {
        outer: dims_product(&dims[..axis], data)?,
        axis_size: dims_product(&dims[axis..=axis], data)?,
        inner: dims_product(&dims[axis + 1..], data)?,
        dtype: type_code(data.dtype),
        wide: u32::from(wide),
        max: u32::from(max),
    })
}

/// The reference loop: for each output element, the index of the first value along the axis
/// that no later value `beats`.
fn arg<T: Pod, I: Pod>(
    p: &ArgMinMaxParams,
    input: &[T],
    output: &mut [I],
    beats: impl Fn(T, T) -> bool,
    index: impl Fn(usize) -> I,
) {
    let (axis_size, inner) = (p.axis_size as usize, p.inner as usize);
    for outer in 0..p.outer as usize {
        let base = outer * axis_size * inner;
        for i in 0..inner {
            let mut best = input[base + i];
            let mut best_index = 0;
            for k in 1..axis_size {
                let value = input[base + k * inner + i];
                if beats(value, best) {
                    best = value;
                    best_index = k;
                }
            }
            output[outer * inner + i] = index(best_index);
        }
    }
}

/// Runs [`arg`] with the comparison of the operator and the index type of the output.
fn run<T: Pod + PartialOrd>(
    ctx: &mut EvalCtx<'_, '_>,
    p: &ArgMinMaxParams,
    beats_max: fn(T, T) -> bool,
    beats_min: fn(T, T) -> bool,
) -> Result<()> {
    let input = in_slice::<T>(ctx.inputs, 0, p.input_len())?;
    let beats = if p.max != 0 { beats_max } else { beats_min };
    if p.wide != 0 {
        let output = out_slice::<i64>(ctx.outputs, 0, p.output_len())?;
        arg(p, input, output, beats, |k| k as i64);
    } else {
        let output = out_slice::<i32>(ctx.outputs, 0, p.output_len())?;
        arg(p, input, output, beats, |k| k as i32);
    }
    Ok(())
}

/// `std::greater` and `std::less`.
fn greater<T: PartialOrd>(a: T, b: T) -> bool {
    a > b
}

fn less<T: PartialOrd>(a: T, b: T) -> bool {
    a < b
}

/// The NaN-aware comparisons of the reference kernel for floating-point inputs.
fn greater_f32(candidate: f32, current: f32) -> bool {
    !candidate.is_nan() && (current.is_nan() || candidate > current)
}

fn less_f32(candidate: f32, current: f32) -> bool {
    !candidate.is_nan() && (current.is_nan() || candidate < current)
}

fn arg_min_max_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: ArgMinMaxParams = *pod::read(ctx.persist)?;
    match p.dtype {
        t if t == type_code(DataType::F32) => run::<f32>(ctx, &p, greater_f32, less_f32),
        // `bool` holds 0 or 1 in one byte: compared like `u8`.
        t if t == type_code(DataType::U8) || t == type_code(DataType::Bool) => {
            run::<u8>(ctx, &p, greater, less)
        }
        t if t == type_code(DataType::I8) => run::<i8>(ctx, &p, greater, less),
        t if t == type_code(DataType::I16) => run::<i16>(ctx, &p, greater, less),
        t if t == type_code(DataType::I32) => run::<i32>(ctx, &p, greater, less),
        _ => Err(crate::util::size_mismatch(0, 0)),
    }
}

fixed_kernel!(
    /// `ARG_MAX` and `ARG_MIN` of `f32`, `u8`, `i8`, `i16`, `i32` and `bool` values, with `i32`
    /// or `i64` indices.
    ARG_MIN_MAX,
    "rust/arg_min_max",
    arg_min_max_params,
    arg_min_max_eval
);
