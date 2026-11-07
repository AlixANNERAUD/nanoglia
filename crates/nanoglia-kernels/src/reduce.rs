//! Reductions: `MEAN`, `SUM`, `REDUCE_MAX` and `REDUCE_MIN` (TFLite `reduce.cc`, reference
//! kernels `reference_ops::Mean`, `reference_ops::ReduceGeneric` and
//! `reference_ops::QuantizedMeanOrSum`).
//!
//! Quantized `SUM` follows TensorFlow Lite Micro: the sum of the inputs minus their zero points is
//! requantized with `QuantizeMultiplier(input_scale / output_scale)`, in fixed point. LiteRT 2.3's
//! reference `SUM` reads int8 and int16 inputs as `uint8` bytes (only its `uint8` path is right),
//! and its optimized `SUM` requantizes in floating point.
//!
//! `REDUCE_MAX` and `REDUCE_MIN` start from the lowest (highest) finite value of the type and keep
//! the current value unless the next one is greater (smaller), as TensorFlow Lite Micro does: NaN
//! values are skipped, and a slice of `-inf` (`+inf`) reduces to the lowest (highest) finite
//! `f32`. LiteRT's reference kernel starts a reduction over all dimensions from the first element
//! instead, which differs only on these infinite inputs.

use crate::util::{
    constant, elements, expect_arity, in_slice, input, int_range, out_slice, output, per_tensor,
    pod_struct, read_index, shape_mismatch, size_mismatch, type_code, unsupported,
    unsupported_type,
};
use nanoglia_core::backend::{EvalCtx, FnKernel, KernelPlan};
use nanoglia_core::pod::{self, Pod, cast_slice_mut};
use nanoglia_core::quant::{Multiplier, multiply_by_quantized_multiplier, quantize_multiplier};
use nanoglia_core::{BuiltinOp, DataType, MAX_RANK, OpDesc, OpOptions, Result};

const OP_MEAN: u32 = 0;
const OP_SUM: u32 = 1;
const OP_MAX: u32 = 2;
const OP_MIN: u32 = 3;

pod_struct! {
    /// Parameters of a reduction.
    pub(crate) struct ReduceParams {
        /// Input dimensions.
        dims: [u32; MAX_RANK],
        /// Stride in the output of each input dimension (0 for reduced dimensions).
        out_strides: [u32; MAX_RANK],
        rank: u32,
        input_len: u32,
        output_len: u32,
        /// Number of input elements reduced into each output element.
        count: u32,
        dtype: u32,
        op: u32,
        /// Requantization of quantized `MEAN` (including the division by `count`) and `SUM`.
        multiplier: Multiplier,
        input_zero_point: i32,
        output_zero_point: i32,
        min: i32,
        max: i32,
    }
}

impl ReduceParams {
    /// Whether the reduction accumulates quantized values into an `i32` scratch buffer.
    fn quantized_sum(&self) -> bool {
        matches!(self.op, OP_MEAN | OP_SUM)
            && [DataType::I8, DataType::U8, DataType::I16]
                .iter()
                .any(|&t| type_code(t) == self.dtype)
    }

    /// Size of the scratch buffer: the `MEAN` and quantized `SUM` accumulators.
    fn scratch(&self) -> usize {
        if self.op == OP_MEAN || self.quantized_sum() {
            self.output_len as usize * 4
        } else {
            0
        }
    }
}

fn reduce_params(op: &OpDesc<'_, '_>) -> Result<ReduceParams> {
    expect_arity(op, 2, 1)?;
    let (input, axis, output) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    let code = match op.builtin() {
        Some(BuiltinOp::MEAN) => OP_MEAN,
        Some(BuiltinOp::SUM) => OP_SUM,
        Some(BuiltinOp::REDUCE_MAX) => OP_MAX,
        Some(BuiltinOp::REDUCE_MIN) => OP_MIN,
        _ => return Err(unsupported(op)),
    };
    if !matches!(op.options, OpOptions::Reducer { .. } | OpOptions::None)
        || input.dtype != output.dtype
    {
        return Err(unsupported(op));
    }
    let input_len = elements(input)?;
    let output_len = elements(output)?;
    if input_len == 0 {
        return Err(unsupported(op));
    }
    let rank = input.dims.rank();
    let dims = input.dims.as_slice();

    // TFLite `ResolveAxis`: wrap negative axes and drop duplicates.
    let axis_data = constant(op, axis)?;
    let axis_count = elements(axis)?;
    let mut reduced = [false; MAX_RANK];
    for i in 0..axis_count {
        let a = read_index(axis, axis_data, i).ok_or_else(|| unsupported_type(axis))?;
        let a = if a < 0 { a + rank as i64 } else { a };
        if !(0..rank as i64).contains(&a) {
            return Err(shape_mismatch(axis));
        }
        reduced[a as usize] = true;
    }

    let mut params = ReduceParams {
        dims: [1; MAX_RANK],
        out_strides: [0; MAX_RANK],
        rank: rank as u32,
        input_len: input_len as u32,
        output_len: output_len as u32,
        count: 1,
        dtype: type_code(input.dtype),
        op: code,
        multiplier: Multiplier::default(),
        input_zero_point: 0,
        output_zero_point: 0,
        min: 0,
        max: 0,
    };
    // `ReducedOutputOffset`: the output is the row-major array of the kept dimensions.
    let mut stride = 1u32;
    for d in (0..rank).rev() {
        params.dims[d] = dims[d] as u32;
        if reduced[d] {
            params.count *= dims[d] as u32;
        } else {
            params.out_strides[d] = stride;
            stride *= dims[d] as u32;
        }
    }
    if stride as usize != output_len {
        return Err(shape_mismatch(output));
    }

    let quantized = matches!(input.dtype, DataType::I8 | DataType::U8 | DataType::I16);
    match (code, input.dtype) {
        (_, DataType::F32) => {}
        (OP_SUM | OP_MAX | OP_MIN, DataType::I32 | DataType::I64) => {}
        (_, _) if quantized => {
            let (input_scale, input_zero_point) = per_tensor(input)?;
            let (output_scale, output_zero_point) = per_tensor(output)?;
            if input.dtype == DataType::I16 && (input_zero_point != 0 || output_zero_point != 0) {
                return Err(unsupported(op));
            }
            params.input_zero_point = input_zero_point;
            params.output_zero_point = output_zero_point;
            (params.min, params.max) =
                int_range(input.dtype).ok_or_else(|| unsupported_type(input))?;
            match code {
                OP_MAX | OP_MIN => {
                    // Values are compared without requantization: TFLite requires the same
                    // quantization on both sides.
                    if input_scale != output_scale || input_zero_point != output_zero_point {
                        return Err(unsupported(op));
                    }
                }
                _ => {
                    let mut m =
                        quantize_multiplier(f64::from(input_scale) / f64::from(output_scale));
                    if code == OP_MEAN {
                        // Fold the division by the element count into the multiplier, as
                        // TFLite does.
                        let count = i64::from(params.count);
                        let count_log2 = (63 - (count as u64).leading_zeros() as i32)
                            .min(32)
                            .min(31 + m.shift);
                        if count_log2 < 0 {
                            return Err(unsupported(op));
                        }
                        m.multiplier = ((i64::from(m.multiplier) << count_log2) / count) as i32;
                        m.shift -= count_log2;
                    }
                    // `MultiplyByQuantizedMultiplier` shifts an `int32_t` left by positive shifts.
                    if m.shift > 30 {
                        return Err(unsupported(op));
                    }
                    params.multiplier = m;
                }
            }
        }
        _ => return Err(unsupported_type(input)),
    }
    Ok(params)
}

fn reduce_plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    let p = reduce_params(op)?;
    Ok(KernelPlan {
        persistent: core::mem::size_of::<ReduceParams>(),
        scratch: p.scratch(),
    })
}

fn reduce_prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    pod::write(persist, reduce_params(op)?)
}

/// Calls `f(input_index, output_index)` for every input element in row-major order.
fn for_each_reduced(p: &ReduceParams, mut f: impl FnMut(usize, usize)) {
    let rank = p.rank as usize;
    if rank == 0 {
        f(0, 0);
        return;
    }
    let last = rank - 1;
    let inner = p.dims[last] as usize;
    let inner_stride = p.out_strides[last] as usize;
    let outer = p.input_len as usize / inner.max(1);
    let mut index = [0u32; MAX_RANK];
    let mut i = 0;
    for _ in 0..outer {
        let base: usize = (0..last)
            .map(|d| index[d] as usize * p.out_strides[d] as usize)
            .sum();
        for j in 0..inner {
            f(i, base + j * inner_stride);
            i += 1;
        }
        for d in (0..last).rev() {
            index[d] += 1;
            if index[d] < p.dims[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

/// TFLite `ReduceGeneric`: fills the output with `init`, then folds every input element into its
/// output element with `f(current, input)`.
fn reduce_generic<T: Pod>(
    ctx: &mut EvalCtx<'_, '_>,
    p: &ReduceParams,
    init: T,
    f: impl Fn(T, T) -> T,
) -> Result<()> {
    let input = in_slice::<T>(ctx.inputs, 0, p.input_len as usize)?;
    let output = out_slice::<T>(ctx.outputs, 0, p.output_len as usize)?;
    output.fill(init);
    for_each_reduced(p, |i, o| output[o] = f(output[o], input[i]));
    Ok(())
}

/// `REDUCE_MAX` or `REDUCE_MIN` on one type, with TFLite's comparisons and initial values.
fn min_max<T: Pod + PartialOrd>(
    ctx: &mut EvalCtx<'_, '_>,
    p: &ReduceParams,
    (lowest, highest): (T, T),
) -> Result<()> {
    if p.op == OP_MAX {
        reduce_generic(ctx, p, lowest, |c, x| if x > c { x } else { c })
    } else {
        reduce_generic(ctx, p, highest, |c, x| if x < c { x } else { c })
    }
}

fn reduce_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: ReduceParams = *pod::read(ctx.persist)?;
    let dtype = p.dtype;
    let is = |t: DataType| dtype == type_code(t);
    if p.op == OP_MAX || p.op == OP_MIN {
        return match () {
            _ if is(DataType::F32) => min_max(ctx, &p, (f32::MIN, f32::MAX)),
            _ if is(DataType::I8) => min_max(ctx, &p, (i8::MIN, i8::MAX)),
            _ if is(DataType::U8) => min_max(ctx, &p, (u8::MIN, u8::MAX)),
            _ if is(DataType::I16) => min_max(ctx, &p, (i16::MIN, i16::MAX)),
            _ if is(DataType::I32) => min_max(ctx, &p, (i32::MIN, i32::MAX)),
            _ => min_max(ctx, &p, (i64::MIN, i64::MAX)),
        };
    }
    if p.op == OP_SUM && !p.quantized_sum() {
        return match () {
            _ if is(DataType::F32) => reduce_generic(ctx, &p, 0.0f32, |c, x| x + c),
            _ if is(DataType::I32) => reduce_generic(ctx, &p, 0i32, |c, x| c.wrapping_add(x)),
            _ => reduce_generic(ctx, &p, 0i64, |c, x| c.wrapping_add(x)),
        };
    }

    let (input_len, output_len) = (p.input_len as usize, p.output_len as usize);
    let scratch = ctx
        .scratch
        .get_mut(..output_len * 4)
        .ok_or_else(|| size_mismatch(output_len * 4, 0))?;
    if is(DataType::F32) {
        // `MEAN`: sum, then divide by the element count.
        let sums = cast_slice_mut::<f32>(scratch)?;
        sums.fill(0.0);
        let input = in_slice::<f32>(ctx.inputs, 0, input_len)?;
        for_each_reduced(&p, |i, o| sums[o] += input[i]);
        let output = out_slice::<f32>(ctx.outputs, 0, output_len)?;
        let count = p.count as f32;
        output
            .iter_mut()
            .zip(sums.iter())
            .for_each(|(o, &s)| *o = s / count);
        return Ok(());
    }

    // Quantized `MEAN` and `SUM` (`QuantizedMeanOrSum`).
    let sums = cast_slice_mut::<i32>(scratch)?;
    sums.fill(0);
    let size = if is(DataType::I16) { 2 } else { 1 };
    let input = in_slice::<u8>(ctx.inputs, 0, input_len * size)?;
    let read = |i: usize| -> i32 {
        if is(DataType::I8) {
            i32::from(input[i] as i8)
        } else if is(DataType::U8) {
            i32::from(input[i])
        } else {
            i32::from(i16::from_ne_bytes([input[2 * i], input[2 * i + 1]]))
        }
    };
    for_each_reduced(&p, |i, o| sums[o] = sums[o].wrapping_add(read(i)));
    let output = out_slice::<u8>(ctx.outputs, 0, output_len * size)?;
    let zero_point_sum = i64::from(p.input_zero_point) * i64::from(p.count);
    for (o, &sum) in sums.iter().enumerate() {
        let shifted = (i64::from(sum) - zero_point_sum) as i32;
        let y = multiply_by_quantized_multiplier(shifted, p.multiplier)
            .wrapping_add(p.output_zero_point)
            .clamp(p.min, p.max);
        if size == 1 {
            output[o] = y as u8;
        } else {
            output[2 * o..2 * o + 2].copy_from_slice(&(y as i16).to_ne_bytes());
        }
    }
    Ok(())
}

/// `MEAN` on `f32`, `i8`, `u8` and `i16`.
pub(crate) static MEAN: FnKernel = FnKernel {
    name: "rust/mean",
    plan: reduce_plan,
    prepare: reduce_prepare,
    eval: reduce_eval,
};

/// `SUM` on `f32`, `i8`, `u8`, `i16`, `i32` and `i64`.
pub(crate) static SUM: FnKernel = FnKernel {
    name: "rust/sum",
    plan: reduce_plan,
    prepare: reduce_prepare,
    eval: reduce_eval,
};

/// `REDUCE_MAX` and `REDUCE_MIN` on `f32`, `i8`, `u8`, `i16`, `i32` and `i64`.
pub(crate) static REDUCE_MIN_MAX: FnKernel = FnKernel {
    name: "rust/reduce_min_max",
    plan: reduce_plan,
    prepare: reduce_prepare,
    eval: reduce_eval,
};
