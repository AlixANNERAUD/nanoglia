//! `TRANSPOSE` (TFLite `transpose.cc`, reference `Transpose`), for any element type of 1, 2, 4 or
//! 8 bytes.
//!
//! Like TFLite, the operator only moves bytes: quantized tensors are copied unchanged, whatever
//! their quantization.

use crate::strided::{StridedCopy, row_major_strides, strided_eval};
use crate::util::{
    constant, elements, expect_arity, fixed_kernel, input, output, read_index, shape_mismatch,
    unsupported, unsupported_type,
};
use nanoglia_core::{DataType, MAX_RANK, OpDesc, Result};

fn transpose_params(op: &OpDesc<'_, '_>) -> Result<StridedCopy> {
    expect_arity(op, 2, 1)?;
    let (input, perm, output) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    if input.dtype != output.dtype {
        return Err(unsupported_type(output));
    }
    let size = input.dtype.size().ok_or_else(|| unsupported_type(input))?;
    let len = elements(input)?;
    elements(output)?;
    let rank = input.dims.rank();
    // TFLite reads the permutation as int32 whatever its type, and a scalar has no strides.
    if rank == 0 || perm.dtype != DataType::I32 || perm.dims.as_slice() != [rank as i32] {
        return Err(unsupported(op));
    }
    if output.dims.rank() != rank {
        return Err(shape_mismatch(output));
    }
    let perm_data = constant(op, perm)?;
    let strides = row_major_strides(input.dims.as_slice());
    let mut used = [false; MAX_RANK];
    let mut axes = [(0usize, 0i64); MAX_RANK];
    for (d, axis) in axes[..rank].iter_mut().enumerate() {
        let p = read_index(perm, perm_data, d).ok_or_else(|| unsupported_type(perm))?;
        // TFLite accepts `-rank..rank`; it does not check that `perm` is a permutation, and
        // reads outside the input when it is not: such models are refused.
        if !(-(rank as i64)..rank as i64).contains(&p) {
            return Err(unsupported(op));
        }
        let p = if p < 0 { p + rank as i64 } else { p } as usize;
        if core::mem::replace(&mut used[p], true) {
            return Err(unsupported(op));
        }
        if output.dims.dim(d) != input.dims.dim(p) {
            return Err(shape_mismatch(output));
        }
        *axis = (input.dims.dim(p), strides[p]);
    }
    StridedCopy::new(&axes[..rank], 0, len, size, shape_mismatch(input))
}

fixed_kernel!(
    /// `TRANSPOSE` of any type with elements of 1, 2, 4 or 8 bytes, up to rank 6.
    TRANSPOSE,
    "rust/transpose",
    transpose_params,
    strided_eval
);
