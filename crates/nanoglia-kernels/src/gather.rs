//! `GATHER` (TFLite `gather.cc`, reference `Gather`), for the element types TFLite gathers, with
//! int16, int32 or int64 indices that may be computed at run time.
//!
//! Like TFLite, the operator only moves bytes: quantized tensors are copied unchanged, whatever
//! their quantization. An index outside the gathered axis is an error at evaluation, as in
//! TFLite.

use crate::util::{
    dims_product, elements, expect_arity, fixed_kernel, in_slice, input, out_slice, output,
    pod_struct, shape_mismatch, size_mismatch, unsupported, unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::{DataType, Detail, Error, MAX_RANK, OpDesc, OpOptions, Result, pod};

pod_struct! {
    /// Parameters of `GATHER`.
    pub(crate) struct GatherParams {
        /// Product of the batch dimensions.
        batch: u32,
        /// Product of the input dimensions between the batch dimensions and the axis.
        outer: u32,
        /// Size of the gathered axis.
        axis_size: u32,
        /// Number of indices per batch.
        coords: u32,
        /// Bytes copied per index: the input dimensions after the axis, times the element size.
        inner: u32,
        /// Size of an index in bytes: 2, 4 or 8.
        index_size: u32,
        /// Number of indices.
        indices: u32,
        /// Input size in bytes.
        in_bytes: u32,
        /// Output size in bytes.
        out_bytes: u32,
    }
}

fn gather_params(op: &OpDesc<'_, '_>) -> Result<GatherParams> {
    let OpOptions::Gather { axis, batch_dims } = op.options else {
        return Err(unsupported(op));
    };
    expect_arity(op, 2, 1)?;
    let (input, positions, output) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    // TFLite gives the output the input's type, and gathers these types (int4 and strings
    // excepted).
    let gathered = matches!(
        input.dtype,
        DataType::F32
            | DataType::F16
            | DataType::BF16
            | DataType::U8
            | DataType::I8
            | DataType::I16
            | DataType::I32
            | DataType::I64
            | DataType::Bool
    );
    if !gathered || output.dtype != input.dtype {
        return Err(unsupported_type(output));
    }
    let index_size = match positions.dtype {
        DataType::I16 => 2,
        DataType::I32 => 4,
        DataType::I64 => 8,
        _ => return Err(unsupported_type(positions)),
    };
    let size = input.dtype.size().ok_or_else(|| unsupported_type(input))?;
    elements(input)?;
    let indices = elements(positions)?;
    elements(output)?;
    let bytes = |t: &nanoglia_core::TensorDesc<'_>| {
        t.byte_len()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| shape_mismatch(t))
    };

    // TFLite `Prepare`.
    let in_dims = input.dims.as_slice();
    let pos_dims = positions.dims.as_slice();
    let (rank, pos_rank) = (in_dims.len() as i32, pos_dims.len() as i32);
    let i16_range = i32::from(i16::MIN)..=i32::from(i16::MAX);
    if !i16_range.contains(&axis) || !i16_range.contains(&batch_dims) {
        return Err(unsupported(op));
    }
    let axis = if axis < 0 { axis + rank } else { axis };
    let batch_dims = if batch_dims < 0 {
        batch_dims + pos_rank
    } else {
        batch_dims
    };
    if !(0..rank).contains(&axis)
        || batch_dims > axis
        || !(0..rank).contains(&batch_dims)
        || batch_dims > pos_rank
    {
        return Err(unsupported(op));
    }
    let (axis, batch_dims) = (axis as usize, batch_dims as usize);
    if in_dims[..batch_dims] != pos_dims[..batch_dims] {
        return Err(shape_mismatch(positions));
    }

    // Output shape: input[..axis], positions[batch_dims..], input[axis + 1..].
    let mut expected = [0i32; 2 * MAX_RANK];
    let parts = [
        &in_dims[..axis],
        &pos_dims[batch_dims..],
        &in_dims[axis + 1..],
    ];
    let mut n = 0;
    for &d in parts.iter().copied().flatten() {
        expected[n] = d;
        n += 1;
    }
    if output.dims.as_slice() != &expected[..n] {
        return Err(shape_mismatch(output));
    }

    let product = |dims: &[i32]| dims_product(dims, input);
    let inner = product(&in_dims[axis + 1..])?
        .checked_mul(size as u32)
        .ok_or_else(|| shape_mismatch(input))?;
    Ok(GatherParams {
        batch: product(&in_dims[..batch_dims])?,
        outer: product(&in_dims[batch_dims..axis])?,
        axis_size: product(&in_dims[axis..axis + 1])?,
        coords: dims_product(&pos_dims[batch_dims..], positions)?,
        inner,
        index_size,
        indices: indices as u32,
        in_bytes: bytes(input)?,
        out_bytes: bytes(output)?,
    })
}

/// An index the reference kernel refuses: `position` is its place in the indices tensor.
fn out_of_range(position: usize) -> Error {
    Error::invalid_argument(Detail::IndexOutOfRange {
        index: u32::try_from(position).unwrap_or(u32::MAX),
    })
}

/// Checks the indices like TFLite (all non-negative, then below the axis size) and returns them
/// in their type.
fn checked_indices<'d>(
    inputs: &[&'d [u8]],
    p: &GatherParams,
    count: usize,
) -> Result<IndexView<'d>> {
    let view = match p.index_size {
        2 => IndexView::I16(in_slice(inputs, 1, count)?),
        4 => IndexView::I32(in_slice(inputs, 1, count)?),
        8 => IndexView::I64(in_slice(inputs, 1, count)?),
        _ => return Err(size_mismatch(count, 0)),
    };
    for i in 0..count {
        if view.get(i) < 0 {
            return Err(out_of_range(i));
        }
    }
    for i in 0..count {
        if view.get(i) >= i64::from(p.axis_size) {
            return Err(out_of_range(i));
        }
    }
    Ok(view)
}

/// The indices of a `GATHER`, in their type.
enum IndexView<'a> {
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

impl IndexView<'_> {
    fn get(&self, i: usize) -> i64 {
        match self {
            Self::I16(v) => v.get(i).map_or(-1, |&x| i64::from(x)),
            Self::I32(v) => v.get(i).map_or(-1, |&x| i64::from(x)),
            Self::I64(v) => v.get(i).copied().unwrap_or(-1),
        }
    }
}

fn gather_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: GatherParams = *pod::read(ctx.persist)?;
    let (batch, outer, axis_size) = (p.batch as usize, p.outer as usize, p.axis_size as usize);
    let (coords, inner) = (p.coords as usize, p.inner as usize);
    let indices = checked_indices(ctx.inputs, &p, p.indices as usize)?;
    let input = in_slice::<u8>(ctx.inputs, 0, p.in_bytes as usize)?;
    let output = out_slice::<u8>(ctx.outputs, 0, p.out_bytes as usize)?;
    // Otherwise, `batch * outer * coords * inner` is the output size.
    if p.out_bytes == 0 {
        return Ok(());
    }
    let mut chunks = output.chunks_exact_mut(inner);
    for b in 0..batch {
        for o in 0..outer {
            let base = (b * outer + o) * axis_size;
            for i in 0..coords {
                // Checked above: `0 <= coord < axis_size`.
                let coord = indices.get(b * coords + i) as usize;
                let from = (base + coord) * inner;
                let src = input
                    .get(from..from + inner)
                    .ok_or_else(|| out_of_range(b * coords + i))?;
                let dst = chunks.next().ok_or_else(|| size_mismatch(inner, 0))?;
                dst.copy_from_slice(src);
            }
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `GATHER` of `f32`, `f16`, `bf16`, `u8`, `i8`, `i16`, `i32`, `i64` and `bool` values with
    /// `i16`, `i32` or `i64` indices.
    GATHER,
    "rust/gather",
    gather_params,
    gather_eval
);
