//! `SPLIT` (TFLite `split.cc`) and `SPLIT_V` (TFLite `split_v.cc`), both with the reference
//! `Split`, with a constant axis and constant sizes.
//!
//! Like TFLite, these operators only move bytes: quantized tensors are copied unchanged,
//! whatever their quantization.

use crate::util::{
    constant, dims_product, elements, fixed_kernel, in_slice, input, out_slice, output, pod_struct,
    read_index, shape_mismatch, unsupported, unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::{
    BuiltinOp, DataType, MAX_OP_OUTPUTS, OpDesc, OpOptions, Result, TensorDesc, pod,
};

pod_struct! {
    /// Parameters of `SPLIT` and `SPLIT_V`.
    pub(crate) struct SplitParams {
        /// Product of the dimensions before the axis.
        outer: u32,
        /// Number of outputs.
        outputs: u32,
        /// The input holding the data: 1 for `SPLIT`, 0 for `SPLIT_V`.
        data: u32,
        /// Bytes copied to each output per outer step.
        chunk: [u32; MAX_OP_OUTPUTS],
    }
}

/// The value of a constant int32 axis tensor of one element, made non-negative for a tensor of
/// `rank` dimensions (TFLite reads it as `int`).
fn split_axis(op: &OpDesc<'_, '_>, axis: &TensorDesc<'_>, rank: usize) -> Result<usize> {
    if axis.dtype != DataType::I32 || axis.dims.num_elements() != Some(1) {
        return Err(unsupported(op));
    }
    let data = constant(op, axis)?;
    let value = read_index(axis, data, 0).ok_or_else(|| unsupported(op))?;
    let value = if value < 0 {
        value + rank as i64
    } else {
        value
    };
    if !(0..rank as i64).contains(&value) {
        return Err(unsupported(op));
    }
    Ok(value as usize)
}

fn split_params(op: &OpDesc<'_, '_>) -> Result<SplitParams> {
    let OpOptions::Split { num_splits } = op.options else {
        return Err(unsupported(op));
    };
    let v = match op.builtin() {
        Some(BuiltinOp::SPLIT) => false,
        Some(BuiltinOp::SPLIT_V) => true,
        _ => return Err(unsupported(op)),
    };
    let outputs = op.outputs.len();
    if op.inputs.len() != if v { 3 } else { 2 }
        || outputs == 0
        || outputs > MAX_OP_OUTPUTS
        || i32::try_from(outputs) != Ok(num_splits)
    {
        return Err(unsupported(op));
    }
    // SPLIT takes (axis, input), SPLIT_V (input, size_splits, axis).
    let input = input(op, if v { 0 } else { 1 })?;
    let supported = matches!(
        input.dtype,
        DataType::F32 | DataType::U8 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64
    );
    if !supported {
        return Err(unsupported_type(input));
    }
    let size = input.dtype.size().ok_or_else(|| unsupported_type(input))?;
    elements(input)?;
    let in_dims = input.dims.as_slice();
    let rank = in_dims.len();
    let axis = split_axis(op, crate::util::input(op, if v { 2 } else { 0 })?, rank)?;
    let axis_size = i64::from(in_dims[axis]);

    // The size of each output along the axis.
    let mut sizes = [0i64; MAX_OP_OUTPUTS];
    if v {
        let size_splits = crate::util::input(op, 1)?;
        let typed = matches!(size_splits.dtype, DataType::I32 | DataType::I64);
        if !typed || size_splits.dims.as_slice() != [outputs as i32] {
            return Err(unsupported(op));
        }
        let data = constant(op, size_splits)?;
        let mut minus_one = None;
        let mut sum = 0i64;
        for (i, slot) in sizes[..outputs].iter_mut().enumerate() {
            let s = read_index(size_splits, data, i).ok_or_else(|| unsupported(op))?;
            if s == -1 {
                // TFLite allows one -1: the rest of the axis.
                if minus_one.replace(i).is_some() {
                    return Err(unsupported(op));
                }
            } else if s < 0 {
                // TFLite would give the output a negative dimension.
                return Err(unsupported(op));
            } else {
                sum = sum.checked_add(s).ok_or_else(|| unsupported(op))?;
                *slot = s;
            }
        }
        match minus_one {
            Some(i) if sum <= axis_size => sizes[i] = axis_size - sum,
            None if sum == axis_size => {}
            _ => return Err(unsupported(op)),
        }
    } else {
        if axis_size % outputs as i64 != 0 {
            return Err(unsupported(op));
        }
        sizes[..outputs].fill(axis_size / outputs as i64);
    }

    let inner = dims_product(&in_dims[axis + 1..], input)? as usize;
    let mut params = SplitParams {
        outer: dims_product(&in_dims[..axis], input)?,
        outputs: outputs as u32,
        data: u32::from(!v),
        chunk: [0; MAX_OP_OUTPUTS],
    };
    for (i, &s) in sizes[..outputs].iter().enumerate() {
        let out = output(op, i)?;
        if out.dtype != input.dtype {
            return Err(unsupported_type(out));
        }
        elements(out)?;
        let dims = out.dims.as_slice();
        let matches = dims.len() == rank
            && dims
                .iter()
                .zip(in_dims)
                .enumerate()
                .all(|(d, (&o, &x))| if d == axis { i64::from(o) == s } else { o == x });
        if !matches {
            return Err(shape_mismatch(out));
        }
        params.chunk[i] =
            u32::try_from(s as usize * inner * size).map_err(|_| shape_mismatch(out))?;
    }
    Ok(params)
}

fn split_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: SplitParams = *pod::read(ctx.persist)?;
    let n = (p.outputs as usize).min(MAX_OP_OUTPUTS);
    let outer = p.outer as usize;
    let row: usize = p.chunk[..n].iter().map(|&c| c as usize).sum();
    let input = in_slice::<u8>(ctx.inputs, p.data as usize, outer * row)?;
    let mut offset = 0;
    for (i, &chunk) in p.chunk[..n].iter().enumerate() {
        let chunk = chunk as usize;
        let output = out_slice::<u8>(ctx.outputs, i, outer * chunk)?;
        if chunk != 0 {
            for (o, dst) in output.chunks_exact_mut(chunk).enumerate() {
                let src = o * row + offset;
                dst.copy_from_slice(&input[src..src + chunk]);
            }
        }
        offset += chunk;
    }
    Ok(())
}

fixed_kernel!(
    /// `SPLIT` and `SPLIT_V` of `f32`, `u8`, `i8`, `i16`, `i32` and `i64`, up to 8 outputs.
    SPLIT,
    "rust/split",
    split_params,
    split_eval
);
