//! `SLICE` (TFLite `slice.cc`, reference `Slice`) and `STRIDED_SLICE` (TFLite `strided_slice.cc`,
//! reference `StridedSlice` with `DynamicStridedSliceParams`), for any element type of 1, 2, 4 or
//! 8 bytes, with constant indices.
//!
//! Like TFLite, these operators only move bytes: quantized tensors are copied unchanged,
//! whatever their quantization. TFLite computes the output shape from the indices; nanoglia
//! checks that the output tensor of the model has that shape.

use crate::strided::{StridedCopy, row_major_strides, strided_eval};
use crate::util::{
    constant, elements, expect_arity, fixed_kernel, input, output, read_index, shape_mismatch,
    unsupported, unsupported_type,
};
use nanoglia_core::{DataType, MAX_RANK, OpDesc, OpOptions, Result, TensorDesc};

/// Fails unless `tensor` is a constant vector of `len` int32 (or int64 if `int64`) values; returns
/// its bytes.
fn index_vector<'a>(
    op: &OpDesc<'a, '_>,
    tensor: &TensorDesc<'a>,
    len: usize,
    int64: bool,
) -> Result<&'a [u8]> {
    let typed = tensor.dtype == DataType::I32 || (int64 && tensor.dtype == DataType::I64);
    if !typed || tensor.dims.as_slice() != [len as i32] {
        return Err(unsupported(op));
    }
    constant(op, tensor)
}

fn slice_params(op: &OpDesc<'_, '_>) -> Result<StridedCopy> {
    expect_arity(op, 3, 1)?;
    let (input, begin, size_t, output) =
        (input(op, 0)?, input(op, 1)?, input(op, 2)?, output(op, 0)?);
    if input.dtype != output.dtype {
        return Err(unsupported_type(output));
    }
    let size = input.dtype.size().ok_or_else(|| unsupported_type(input))?;
    let len = elements(input)?;
    elements(output)?;
    let rank = input.dims.rank();
    if begin.dtype != size_t.dtype || output.dims.rank() != rank {
        return Err(unsupported(op));
    }
    let begin_data = index_vector(op, begin, rank, true)?;
    let size_data = index_vector(op, size_t, rank, true)?;

    let strides = row_major_strides(input.dims.as_slice());
    let mut axes = [(0usize, 0i64); MAX_RANK];
    let mut start = 0i64;
    for (d, axis) in axes[..rank].iter_mut().enumerate() {
        let dim = input.dims.dim(d) as i64;
        let b = read_index(begin, begin_data, d).ok_or_else(|| unsupported(op))?;
        let mut s = read_index(size_t, size_data, d).ok_or_else(|| unsupported(op))?;
        // TFLite `CalculateOutputShapeVector`.
        if !(0..=dim).contains(&b) {
            return Err(unsupported(op));
        }
        if s < 0 {
            if s != -1 {
                return Err(unsupported(op));
            }
            s = dim - b;
        } else if s > dim - b {
            return Err(unsupported(op));
        }
        if output.dims.dim(d) as i64 != s {
            return Err(shape_mismatch(output));
        }
        *axis = (s as usize, strides[d]);
        start += b * strides[d];
    }
    StridedCopy::new(&axes[..rank], start, len, size, shape_mismatch(input))
}

fixed_kernel!(
    /// `SLICE` of any type with elements of 1, 2, 4 or 8 bytes, up to rank 6.
    SLICE,
    "rust/slice",
    slice_params,
    strided_eval
);

/// The most effective dimensions (input dimensions plus new axes) a strided slice may have: new
/// axes are output dimensions, so a valid model has at most `MAX_RANK` of them.
const MAX_EFFECTIVE: usize = 2 * MAX_RANK;

/// One dimension of a strided slice once the masks are applied (TFLite
/// `BuildStridedSliceParams`).
#[derive(Clone, Copy, Default)]
struct Axis {
    start: i64,
    stop: i64,
    stride: i64,
    begin_mask: bool,
    end_mask: bool,
    shrink: bool,
    /// The input dimension this axis walks, or `None` for a new axis of size 1.
    input_dim: Option<usize>,
    /// Size of the axis in the effective input shape.
    size: i64,
}

/// TFLite `StartForAxis` (`DynamicStridedSliceParams` version).
fn start_for_axis(a: &Axis) -> i64 {
    let mut start = a.start;
    if start < 0 {
        start += a.size;
    }
    start = if a.stride > 0 {
        start.clamp(0, a.size)
    } else {
        start.clamp(-1, a.size - 1)
    };
    if a.begin_mask {
        start = if a.stride > 0 { 0 } else { a.size - 1 };
    }
    start
}

/// TFLite `EndForAxis` (`DynamicStridedSliceParams` version).
fn end_for_axis(a: &Axis, start: i64, offset: bool) -> i64 {
    if a.shrink {
        return if start >= a.size { start } else { start + 1 };
    }
    let mut end = a.stop;
    if offset {
        end += start;
    }
    if end < 0 {
        end += a.size;
    }
    end = if a.stride > 0 {
        end.clamp(0, a.size)
    } else {
        end.clamp(-1, a.size - 1)
    };
    if a.end_mask {
        end = if a.stride > 0 { a.size } else { -1 };
    }
    end
}

/// The output size of an axis as TFLite's `ResizeOutputTensor` computes it from `extent`
/// (`end - start`, or the `end` value with `offset`), or `None` where TFLite fails.
fn resized_dim(extent: i64, stride: i64) -> Option<i64> {
    let dim = if (extent < 0) != (stride < 0) {
        0
    } else if stride < 0 {
        (extent + 1) / stride + 1
    } else if extent == 0 {
        0
    } else {
        (extent - 1) / stride + 1
    };
    (0..=i64::from(i32::MAX)).contains(&dim).then_some(dim)
}

/// The number of elements the reference loop visits from `start` to `stop` (excluded).
fn visited(start: i64, stop: i64, stride: i64) -> i64 {
    if stride > 0 && stop > start {
        (stop - start + stride - 1) / stride
    } else if stride < 0 && start > stop {
        (start - stop - stride - 1) / -stride
    } else {
        0
    }
}

fn strided_slice_params(op: &OpDesc<'_, '_>) -> Result<StridedCopy> {
    let OpOptions::StridedSlice(options) = op.options else {
        return Err(unsupported(op));
    };
    expect_arity(op, 4, 1)?;
    let (input, begin, end, strides_t, output) = (
        input(op, 0)?,
        input(op, 1)?,
        input(op, 2)?,
        input(op, 3)?,
        output(op, 0)?,
    );
    if input.dtype != output.dtype {
        return Err(unsupported_type(output));
    }
    let size = input.dtype.size().ok_or_else(|| unsupported_type(input))?;
    let len = elements(input)?;
    elements(output)?;
    let rank = input.dims.rank();
    let count = begin.dims.as_slice().first().copied().unwrap_or(-1);
    // TFLite: int32 indices only, at most 31 of them. It reads `strides` as long as `begin`.
    if rank == 0 || !(0..=31).contains(&count) {
        return Err(unsupported(op));
    }
    let count = count as usize;
    let begin_data = index_vector(op, begin, count, false)?;
    let end_data = index_vector(op, end, count, false)?;
    let strides_data = index_vector(op, strides_t, count, false)?;
    let read = |t: &TensorDesc<'_>, data: &[u8], i: usize| {
        read_index(t, data, i).ok_or_else(|| unsupported(op))
    };

    let bit = |mask: i32, i: usize| i < 32 && (mask as u32 >> i) & 1 != 0;
    let ellipsis_mask = options.ellipsis_mask;
    // TensorFlow allows a single ellipsis.
    if (ellipsis_mask as u32).count_ones() > 1 {
        return Err(unsupported(op));
    }

    // TFLite `BuildStridedSliceParams`: new axes not overridden by the ellipsis.
    let add_axes = (0..count)
        .filter(|&i| !bit(ellipsis_mask, i) && bit(options.new_axis_mask, i))
        .count();
    let effective = rank + add_axes;
    if effective > MAX_EFFECTIVE {
        return Err(unsupported(op));
    }

    // Expand the ellipsis.
    let mut ellipsis = [false; MAX_EFFECTIVE];
    let mut new_axis = [false; MAX_EFFECTIVE];
    let (mut ellipsis_start, mut expanded) = (effective, 0usize);
    let mut i = 0;
    while i < effective {
        if bit(ellipsis_mask, i) {
            ellipsis_start = i;
            let wanted = (i + 1 + add_axes + rank) as i64 - count as i64;
            let ellipsis_end = ((i + 1) as i64).max(wanted.min(effective as i64)) as usize;
            expanded = ellipsis_end - ellipsis_start - 1;
            while i < ellipsis_end {
                ellipsis[i] = true;
                i += 1;
            }
            continue;
        }
        new_axis[i] = i
            .checked_sub(expanded)
            .is_some_and(|j| bit(options.new_axis_mask, j));
        i += 1;
    }

    // The effective axes.
    let mut axes = [Axis::default(); MAX_EFFECTIVE];
    let (mut added_ellipsis, mut added_axes) = (0usize, 0usize);
    for (i, axis) in axes[..effective].iter_mut().enumerate() {
        if new_axis[i] && !ellipsis[i] {
            *axis = Axis {
                stop: 1,
                stride: 1,
                size: 1,
                ..Axis::default()
            };
            added_axes += 1;
            continue;
        }
        let dim = i - added_axes;
        if dim >= rank {
            return Err(unsupported(op));
        }
        *axis = Axis {
            stride: 1,
            input_dim: Some(dim),
            size: input.dims.dim(dim) as i64,
            ..Axis::default()
        };
        if ellipsis[i] {
            added_ellipsis = i - ellipsis_start;
            axis.begin_mask = true;
            axis.end_mask = true;
        } else if i >= count + expanded {
            axis.begin_mask = true;
            axis.end_mask = true;
        } else {
            let orig = i - added_ellipsis;
            if orig >= count {
                return Err(unsupported(op));
            }
            axis.start = read(begin, begin_data, orig)?;
            axis.stop = read(end, end_data, orig)?;
            axis.stride = read(strides_t, strides_data, orig)?;
            axis.begin_mask = bit(options.begin_mask, orig);
            axis.end_mask = bit(options.end_mask, orig);
            axis.shrink = bit(options.shrink_axis_mask, orig);
        }
    }
    // Every input dimension must be walked exactly once.
    if effective - added_axes != rank {
        return Err(unsupported(op));
    }

    // Output shape (TFLite `ResizeOutputTensor`) and the elements the reference loop visits.
    let in_strides = row_major_strides(input.dims.as_slice());
    let mut copy_axes = [(0usize, 0i64); MAX_RANK];
    let (mut copy_rank, mut out_rank) = (0usize, 0usize);
    let mut start_offset = 0i64;
    let out_dims = output.dims.as_slice();
    for (idx, axis) in axes[..effective].iter().enumerate() {
        if axis.stride == 0 || axis.stride == i64::from(i32::MIN) {
            return Err(unsupported(op));
        }
        let start = start_for_axis(axis);
        let stop = end_for_axis(axis, start, options.offset);
        if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&stop) {
            return Err(unsupported(op));
        }
        let n = visited(start, stop, axis.stride);
        if axis.shrink {
            // TFLite leaves the output unwritten when a shrunk axis selects nothing.
            if n != 1 {
                return Err(unsupported(op));
            }
        } else {
            let extent = if options.offset {
                // TFLite reads `end` at the effective index here.
                if idx >= count {
                    return Err(unsupported(op));
                }
                read(end, end_data, idx)?
            } else {
                stop - start
            };
            let dim = resized_dim(extent, axis.stride).ok_or_else(|| unsupported(op))?;
            // The output TFLite allocates must be exactly what its loop writes.
            if dim != n {
                return Err(unsupported(op));
            }
            if out_dims.get(out_rank).map(|&d| i64::from(d)) != Some(dim) {
                return Err(shape_mismatch(output));
            }
            out_rank += 1;
        }
        if let Some(d) = axis.input_dim {
            copy_axes[copy_rank] = (n as usize, axis.stride * in_strides[d]);
            copy_rank += 1;
            start_offset += start * in_strides[d];
        }
    }
    if out_rank != out_dims.len() {
        return Err(shape_mismatch(output));
    }
    StridedCopy::new(
        &copy_axes[..copy_rank],
        start_offset,
        len,
        size,
        shape_mismatch(input),
    )
}

fixed_kernel!(
    /// `STRIDED_SLICE` of any type with elements of 1, 2, 4 or 8 bytes, up to rank 6, with
    /// constant int32 indices.
    STRIDED_SLICE,
    "rust/strided_slice",
    strided_slice_params,
    strided_eval
);
