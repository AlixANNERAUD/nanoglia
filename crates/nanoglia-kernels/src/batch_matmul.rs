//! `BATCH_MATMUL` on `f32` (TFLite `batch_matmul.cc`, `reference_ops::BatchMatMul`): the matrix
//! products of the last two dimensions of the inputs, the other dimensions broadcasting, with
//! either input transposed first (`adj_x`, `adj_y`).
//!
//! TFLite copies the right-hand side transposed (and the left-hand side when `adj_x` is set) into
//! temporary tensors, so that both operands of each sum are contiguous, then sums each output
//! over the shared dimension in increasing order, rounding as its AArch64 build does (see
//! [`dot_by_eight`]). This kernel reads both inputs in place with the strides of their layout
//! instead: each output is the same sum of the same products, rounded the same way in the same
//! order, so the results are the same bits without the copies or any scratch memory. The
//! right-hand side may be a constant or an activation alike.
//!
//! When the rows of the right-hand side are contiguous (no `adj_y`, or a depth of one): the kernel then adds each
//! term to a whole output row at once (row, depth, column) instead of summing one output at a
//! time, which reads both operands in memory order and lets the compiler vectorize the loop over
//! the columns. Each output still receives the same terms in the same order.
//!
//! TFLite accepts ranks 2 to 5; this kernel accepts ranks 2 to [`MAX_RANK`]. Quantized and hybrid
//! (float left-hand side, int8 right-hand side) products are refused; the
//! `asymmetric_quantize_inputs` option only concerns hybrid products, and TFLite ignores it on
//! float ones.

use crate::math::{dot_by_eight, mul_add};
use crate::util::{
    check_aligned, elements, expect_arity, expect_type, fixed_kernel, in_slice, input, out_slice,
    output, pod_struct, shape_mismatch, unsupported,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod;
use nanoglia_core::{DataType, MAX_RANK, OpDesc, OpOptions, Result, TensorDesc};

/// The number of batch dimensions: all but the two of the matrices.
const BATCH_RANK: usize = MAX_RANK - 2;

pod_struct! {
    /// Parameters of a float batch matrix product `[..., rows, depth] x [..., depth, cols]`.
    pub(crate) struct BatchMatMulParams {
        /// Output batch dimensions, left-padded with ones.
        batch: [u32; BATCH_RANK],
        /// Element strides of the left-hand side along the batch dimensions (0 when broadcast).
        lhs_batch_stride: [u32; BATCH_RANK],
        /// Element strides of the right-hand side along the batch dimensions (0 when broadcast).
        rhs_batch_stride: [u32; BATCH_RANK],
        rows: u32,
        depth: u32,
        cols: u32,
        /// Element strides of the left-hand side along its rows and its depth.
        lhs_row_stride: u32,
        lhs_depth_stride: u32,
        /// Element strides of the right-hand side along its depth and its columns.
        rhs_depth_stride: u32,
        rhs_col_stride: u32,
        lhs_len: u32,
        rhs_len: u32,
        out_len: u32,
    }
}

/// The dimensions of `tensor`, of rank 2 to [`MAX_RANK`], left-padded with ones to
/// [`MAX_RANK`].
fn padded_dims(tensor: &TensorDesc<'_>) -> Result<[i32; MAX_RANK]> {
    let dims = tensor.dims.as_slice();
    if dims.len() < 2 || dims.iter().any(|&d| d < 0) {
        return Err(shape_mismatch(tensor));
    }
    elements(tensor)?;
    let mut padded = [1; MAX_RANK];
    padded[MAX_RANK - dims.len()..].copy_from_slice(dims);
    Ok(padded)
}

/// The element strides of a contiguous tensor of dimensions `dims`.
fn strides(dims: &[i32; MAX_RANK]) -> [u32; MAX_RANK] {
    let mut strides = [0u32; MAX_RANK];
    let mut stride = 1u32;
    for axis in (0..MAX_RANK).rev() {
        strides[axis] = stride;
        // The product of all dimensions fits an `i32` (checked by `elements`).
        stride = stride.saturating_mul(dims[axis] as u32);
    }
    strides
}

fn batch_matmul_params(op: &OpDesc<'_, '_>) -> Result<BatchMatMulParams> {
    let OpOptions::BatchMatMul { adj_x, adj_y, .. } = op.options else {
        return Err(unsupported(op));
    };
    expect_arity(op, 2, 1)?;
    let (lhs, rhs, output) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    for t in [lhs, rhs, output] {
        expect_type(t, DataType::F32)?;
    }
    check_aligned(lhs, 4)?;
    check_aligned(rhs, 4)?;
    let (l, r) = (padded_dims(lhs)?, padded_dims(rhs)?);
    let (lhs_strides, rhs_strides) = (strides(&l), strides(&r));
    let rank = lhs.dims.rank().max(rhs.dims.rank());

    // The output: the broadcast batch dimensions, then the rows of the left-hand side and the
    // columns of the right-hand side (`ResizeOutputTensor`).
    let mut out = [1i32; MAX_RANK];
    let mut params = BatchMatMulParams {
        batch: [1; BATCH_RANK],
        lhs_batch_stride: [0; BATCH_RANK],
        rhs_batch_stride: [0; BATCH_RANK],
        rows: 0,
        depth: 0,
        cols: 0,
        lhs_row_stride: 0,
        lhs_depth_stride: 0,
        rhs_depth_stride: 0,
        rhs_col_stride: 0,
        lhs_len: elements(lhs)? as u32,
        rhs_len: elements(rhs)? as u32,
        out_len: elements(output)? as u32,
    };
    for axis in 0..BATCH_RANK {
        let (a, b) = (l[axis], r[axis]);
        if a != b && a != 1 && b != 1 {
            return Err(shape_mismatch(rhs));
        }
        let dim = if a == 1 { b } else { a };
        out[axis] = dim;
        params.batch[axis] = dim as u32;
        params.lhs_batch_stride[axis] = if a == 1 { 0 } else { lhs_strides[axis] };
        params.rhs_batch_stride[axis] = if b == 1 { 0 } else { rhs_strides[axis] };
    }
    // The last two dimensions, as stored: [rows, depth] or, adjoint, [depth, rows].
    let (l0, l1) = (l[MAX_RANK - 2], l[MAX_RANK - 1]);
    let (r0, r1) = (r[MAX_RANK - 2], r[MAX_RANK - 1]);
    let (rows, depth) = if adj_x { (l1, l0) } else { (l0, l1) };
    let (rhs_depth, cols) = if adj_y { (r1, r0) } else { (r0, r1) };
    if depth != rhs_depth {
        return Err(shape_mismatch(rhs));
    }
    out[MAX_RANK - 2] = rows;
    out[MAX_RANK - 1] = cols;
    if output.dims.as_slice() != &out[MAX_RANK - rank..] {
        return Err(shape_mismatch(output));
    }
    let (rows, depth, cols) = (rows as u32, depth as u32, cols as u32);
    params.rows = rows;
    params.depth = depth;
    params.cols = cols;
    (params.lhs_row_stride, params.lhs_depth_stride) = if adj_x { (1, rows) } else { (depth, 1) };
    (params.rhs_depth_stride, params.rhs_col_stride) = if adj_y { (1, depth) } else { (cols, 1) };
    Ok(params)
}

fn batch_matmul_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: BatchMatMulParams = *pod::read(ctx.persist)?;
    let lhs = in_slice::<f32>(ctx.inputs, 0, p.lhs_len as usize)?;
    let rhs = in_slice::<f32>(ctx.inputs, 1, p.rhs_len as usize)?;
    let output = out_slice::<f32>(ctx.outputs, 0, p.out_len as usize)?;
    if output.is_empty() {
        // Nothing to compute; the batch dimensions alone may then multiply past `usize`.
        return Ok(());
    }
    let (rows, depth, cols) = (p.rows as usize, p.depth as usize, p.cols as usize);
    let (lhs_row, lhs_depth) = (p.lhs_row_stride as usize, p.lhs_depth_stride as usize);
    let (rhs_depth, rhs_col) = (p.rhs_depth_stride as usize, p.rhs_col_stride as usize);
    let batches: usize = p.batch.iter().map(|&d| d as usize).product();
    let mut index = [0u32; BATCH_RANK];
    // The output is not empty, so `cols` is not zero.
    let mut out_rows = output.chunks_exact_mut(cols);
    // The terms whose products are rounded before being added (see `dot_by_eight`).
    let rounded = depth - depth % 8;
    for _ in 0..batches {
        let (mut lhs_base, mut rhs_base) = (0usize, 0usize);
        for ((&i, &ls), &rs) in index
            .iter()
            .zip(&p.lhs_batch_stride)
            .zip(&p.rhs_batch_stride)
        {
            lhs_base += i as usize * ls as usize;
            rhs_base += i as usize * rs as usize;
        }
        for row in 0..rows {
            let lhs_row_base = lhs_base + row * lhs_row;
            let Some(out_row) = out_rows.next() else {
                break;
            };
            if rhs_col == 1 {
                // The rows of the right-hand side are contiguous: add each term to the whole
                // output row at once. Every output still receives its terms in increasing
                // order, rounded as in `dot_by_eight`, and both rows are read in memory order,
                // which the compiler vectorizes for the rounded products.
                out_row.fill(0.0);
                for k in 0..depth {
                    let l = lhs[lhs_row_base + k * lhs_depth];
                    let start = rhs_base + k * rhs_depth;
                    let rhs_row = &rhs[start..start + cols];
                    if k < rounded {
                        for (y, &r) in out_row.iter_mut().zip(rhs_row) {
                            *y += r * l;
                        }
                    } else {
                        for (y, &r) in out_row.iter_mut().zip(rhs_row) {
                            *y = mul_add(r, l, *y);
                        }
                    }
                }
            } else {
                // The right-hand side is stored transposed (`adj_y`, depth above 1): each of
                // its columns is contiguous along the depth, so sum one output at a time.
                for (col, y) in out_row.iter_mut().enumerate() {
                    let rhs_col_base = rhs_base + col * rhs_col;
                    *y = dot_by_eight(depth, |k| {
                        (
                            rhs[rhs_col_base + k * rhs_depth],
                            lhs[lhs_row_base + k * lhs_depth],
                        )
                    });
                }
            }
        }
        for axis in (0..BATCH_RANK).rev() {
            index[axis] += 1;
            if index[axis] < p.batch[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `BATCH_MATMUL` on `f32`.
    BATCH_MATMUL_F32,
    "rust/batch_matmul_f32",
    batch_matmul_params,
    batch_matmul_eval
);
