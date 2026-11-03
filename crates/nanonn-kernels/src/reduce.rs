//! `MEAN` (TFLite `reduce.cc`, reference kernels `reference_ops::Mean` and
//! `reference_ops::QuantizedMeanOrSum`).

use crate::util::{
    constant, elements, expect_arity, in_slice, input, int_range, out_slice, output, per_tensor,
    pod_struct, read_index, shape_mismatch, type_code, unsupported, unsupported_type,
};
use nanonn_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanonn_core::pod::{self, cast_slice_mut};
use nanonn_core::quant::{Multiplier, multiply_by_quantized_multiplier, quantize_multiplier};
use nanonn_core::{DataType, MAX_RANK, OpDesc, OpOptions, Result};

pod_struct! {
    /// Parameters of `MEAN`.
    pub(crate) struct MeanParams {
        /// Input dimensions.
        dims: [u32; MAX_RANK],
        /// Stride in the output of each input dimension (0 for reduced dimensions).
        out_strides: [u32; MAX_RANK],
        rank: u32,
        input_len: u32,
        output_len: u32,
        /// Number of input elements averaged into each output element.
        count: u32,
        dtype: u32,
        /// Multiplier including the division by `count` (quantized types only).
        multiplier: Multiplier,
        input_zero_point: i32,
        output_zero_point: i32,
        min: i32,
        max: i32,
    }
}

fn mean_params(op: &OpDesc<'_, '_>) -> Result<MeanParams> {
    expect_arity(op, 2, 1)?;
    let (input, axis, output) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
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

    let mut params = MeanParams {
        dims: [1; MAX_RANK],
        out_strides: [0; MAX_RANK],
        rank: rank as u32,
        input_len: input_len as u32,
        output_len: output_len as u32,
        count: 1,
        dtype: type_code(input.dtype),
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

    match input.dtype {
        DataType::F32 => {}
        DataType::I8 | DataType::U8 | DataType::I16 => {
            let (input_scale, input_zero_point) = per_tensor(input)?;
            let (output_scale, output_zero_point) = per_tensor(output)?;
            if input.dtype == DataType::I16 && (input_zero_point != 0 || output_zero_point != 0) {
                return Err(unsupported(op));
            }
            let Multiplier {
                mut multiplier,
                mut shift,
            } = quantize_multiplier(f64::from(input_scale) / f64::from(output_scale));
            // Fold the division by the element count into the multiplier, as TFLite does.
            let count = i64::from(params.count);
            let count_log2 = (63 - (count as u64).leading_zeros() as i32)
                .min(32)
                .min(31 + shift);
            if count_log2 < 0 {
                return Err(unsupported(op));
            }
            multiplier = ((i64::from(multiplier) << count_log2) / count) as i32;
            shift -= count_log2;
            params.multiplier = Multiplier { multiplier, shift };
            params.input_zero_point = input_zero_point;
            params.output_zero_point = output_zero_point;
            (params.min, params.max) =
                int_range(input.dtype).ok_or_else(|| unsupported_type(input))?;
        }
        _ => return Err(unsupported_type(input)),
    }
    Ok(params)
}

fn mean_plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    let p = mean_params(op)?;
    Ok(KernelPlan {
        persistent: core::mem::size_of::<MeanParams>(),
        scratch: p.output_len as usize * 4,
    })
}

fn mean_prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    pod::write(persist, mean_params(op)?)
}

/// Calls `f(input_index, output_index)` for every input element in row-major order.
fn for_each_reduced(p: &MeanParams, mut f: impl FnMut(usize, usize)) {
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

fn mean_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: MeanParams = *pod::read(ctx.persist)?;
    let (input_len, output_len) = (p.input_len as usize, p.output_len as usize);
    let scratch = ctx
        .scratch
        .get_mut(..output_len * 4)
        .ok_or_else(|| crate::util::size_mismatch(output_len * 4, 0))?;
    if p.dtype == type_code(DataType::F32) {
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

    let sums = cast_slice_mut::<i32>(scratch)?;
    sums.fill(0);
    let size = if p.dtype == type_code(DataType::I16) {
        2
    } else {
        1
    };
    let input = in_slice::<u8>(ctx.inputs, 0, input_len * size)?;
    let read = |i: usize| -> i32 {
        match p.dtype {
            t if t == type_code(DataType::I8) => i32::from(input[i] as i8),
            t if t == type_code(DataType::U8) => i32::from(input[i]),
            _ => i32::from(i16::from_ne_bytes([input[2 * i], input[2 * i + 1]])),
        }
    };
    for_each_reduced(&p, |i, o| sums[o] = sums[o].wrapping_add(read(i)));
    let output = out_slice::<u8>(ctx.outputs, 0, output_len * size)?;
    let zero_point_sum = i64::from(p.input_zero_point) * i64::from(p.count);
    for (o, &sum) in sums.iter().enumerate() {
        let shifted = (i64::from(sum) - zero_point_sum) as i32;
        let y = (multiply_by_quantized_multiplier(shifted, p.multiplier) + p.output_zero_point)
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
pub(crate) static MEAN: Kernel = Kernel {
    name: "rust/mean",
    plan: mean_plan,
    prepare: mean_prepare,
    eval: mean_eval,
};
