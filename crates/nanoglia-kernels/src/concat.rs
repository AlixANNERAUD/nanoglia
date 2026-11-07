//! `CONCATENATION` (TFLite `concatenation.cc`, reference `Concatenation`), for any element type,
//! and `PACK` (TFLite `pack.cc`), which is a concatenation along a new axis.
//!
//! Like TFLite, quantized inputs must share the output's scale and zero point: the operators only
//! move bytes.

use crate::util::{
    dims_product, elements, fixed_kernel, in_slice, input, int_range, out_slice, output,
    pod_struct, shape_mismatch, unsupported, unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod;
use nanoglia_core::{Activation, DataType, MAX_OP_INPUTS, OpDesc, OpOptions, Result};

pod_struct! {
    /// Parameters of `CONCATENATION`.
    pub(crate) struct ConcatParams {
        /// Product of the dimensions before the axis.
        outer: u32,
        /// Number of inputs.
        inputs: u32,
        /// Bytes copied from each input per outer step.
        chunk: [u32; MAX_OP_INPUTS],
    }
}

fn concat_params(op: &OpDesc<'_, '_>) -> Result<ConcatParams> {
    let OpOptions::Concatenation { axis, activation } = op.options else {
        return Err(unsupported(op));
    };
    if activation != Activation::None
        || op.inputs.is_empty()
        || op.inputs.len() > MAX_OP_INPUTS
        || op.outputs.len() != 1
    {
        return Err(unsupported(op));
    }
    let output = output(op, 0)?;
    let size = output
        .dtype
        .size()
        .ok_or_else(|| unsupported_type(output))?;
    elements(output)?;
    let rank = output.dims.rank() as i32;
    let axis = if axis < 0 { axis + rank } else { axis };
    if !(0..rank).contains(&axis) {
        return Err(unsupported(op));
    }
    let axis = axis as usize;
    let out_dims = output.dims.as_slice();
    let quantized = int_range(output.dtype).is_some() && output.quant.is_quantized();

    let mut params = ConcatParams {
        outer: out_dims[..axis].iter().map(|&d| d as u32).product(),
        inputs: op.inputs.len() as u32,
        chunk: [0; MAX_OP_INPUTS],
    };
    let inner: usize = out_dims[axis + 1..].iter().map(|&d| d as usize).product();
    let mut axis_total = 0i64;
    for i in 0..op.inputs.len() {
        let t = input(op, i)?;
        if t.dtype != output.dtype {
            return Err(unsupported_type(t));
        }
        elements(t)?;
        let dims = t.dims.as_slice();
        if dims.len() != out_dims.len()
            || dims
                .iter()
                .zip(out_dims)
                .enumerate()
                .any(|(d, (a, b))| d != axis && a != b)
        {
            return Err(shape_mismatch(t));
        }
        if quantized
            && (t.quant.scale() != output.quant.scale()
                || t.quant.zero_point() != output.quant.zero_point())
        {
            return Err(unsupported(op));
        }
        axis_total += i64::from(dims[axis]);
        params.chunk[i] = (dims[axis] as usize * inner * size) as u32;
    }
    if axis_total != i64::from(out_dims[axis]) {
        return Err(shape_mismatch(output));
    }
    Ok(params)
}

fn concat_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: ConcatParams = *pod::read(ctx.persist)?;
    let n = p.inputs as usize;
    let outer = p.outer as usize;
    let row: usize = p.chunk[..n].iter().map(|&c| c as usize).sum();
    let output = out_slice::<u8>(ctx.outputs, 0, outer * row)?;
    let mut offset = 0;
    for (i, &chunk) in p.chunk[..n].iter().enumerate() {
        let chunk = chunk as usize;
        let input = in_slice::<u8>(ctx.inputs, i, outer * chunk)?;
        for o in 0..outer {
            let dst = o * row + offset;
            output[dst..dst + chunk].copy_from_slice(&input[o * chunk..(o + 1) * chunk]);
        }
        offset += chunk;
    }
    Ok(())
}

fixed_kernel!(
    /// `CONCATENATION` of any type with whole-byte elements.
    CONCATENATION,
    "rust/concatenation",
    concat_params,
    concat_eval
);

/// `PACK` (TFLite `pack.cc`): stacking `values_count` inputs of the same shape along a new
/// output axis is a concatenation in which every input contributes one slice per outer step.
fn pack_params(op: &OpDesc<'_, '_>) -> Result<ConcatParams> {
    let OpOptions::Pack { values_count, axis } = op.options else {
        return Err(unsupported(op));
    };
    let n = op.inputs.len();
    if n == 0 || n > MAX_OP_INPUTS || i32::try_from(n) != Ok(values_count) || op.outputs.len() != 1
    {
        return Err(unsupported(op));
    }
    let output = output(op, 0)?;
    let first = input(op, 0)?;
    // The types of LiteRT's `pack.cc` (TensorFlow Lite Micro: f32, int8, int16, int32, int64).
    if !matches!(
        first.dtype,
        DataType::F32
            | DataType::U8
            | DataType::I8
            | DataType::I16
            | DataType::I32
            | DataType::U32
            | DataType::I64
    ) {
        return Err(unsupported_type(first));
    }
    let size = first.dtype.size().ok_or_else(|| unsupported_type(first))?;
    elements(output)?;
    let in_dims = first.dims.as_slice();
    let rank = in_dims.len() as i32 + 1;
    let axis = if axis < 0 { axis + rank } else { axis };
    if !(0..rank).contains(&axis) {
        return Err(unsupported(op));
    }
    let axis = axis as usize;
    // The output has the input dimensions with `values_count` inserted at `axis`.
    let out_dims = output.dims.as_slice();
    let expected = in_dims[..axis]
        .iter()
        .chain([&values_count])
        .chain(&in_dims[axis..]);
    if out_dims.len() != rank as usize || !out_dims.iter().eq(expected) {
        return Err(shape_mismatch(output));
    }
    if output.dtype != first.dtype {
        return Err(unsupported_type(output));
    }
    for i in 0..n {
        let t = input(op, i)?;
        if t.dtype != first.dtype {
            return Err(unsupported_type(t));
        }
        if t.dims != first.dims {
            return Err(shape_mismatch(t));
        }
        // LiteRT requires every input to have the output's scale and zero point.
        if t.quant.scale() != output.quant.scale()
            || t.quant.zero_point() != output.quant.zero_point()
        {
            return Err(unsupported(op));
        }
    }
    let inner = dims_product(&in_dims[axis..], first)? as usize;
    let chunk = u32::try_from(inner * size).map_err(|_| shape_mismatch(first))?;
    let mut params = ConcatParams {
        outer: dims_product(&in_dims[..axis], first)?,
        inputs: n as u32,
        chunk: [0; MAX_OP_INPUTS],
    };
    params.chunk[..n].fill(chunk);
    Ok(params)
}

fixed_kernel!(
    /// `PACK` of `f32`, `u8`, `i8`, `i16`, `i32`, `u32` and `i64`, up to 16 inputs.
    PACK,
    "rust/pack",
    pack_params,
    concat_eval
);
