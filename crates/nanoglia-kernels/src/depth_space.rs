//! `SPACE_TO_DEPTH` and `DEPTH_TO_SPACE` (TFLite `space_to_depth.cc`, `depth_to_space.cc`,
//! reference `SpaceToDepth` and `DepthToSpace`) as strided copies.
//!
//! Seen as `[batch, height / block, block, width / block, block, depth]`, `SPACE_TO_DEPTH` swaps
//! the second and third axes; `DEPTH_TO_SPACE` is the inverse. Like TFLite, the operators only
//! move bytes: quantized tensors are copied unchanged, whatever their quantization.

use crate::strided::{StridedCopy, strided_eval};
use crate::util::{
    dims4, elements, expect_arity, fixed_kernel, input, output, shape_mismatch, unsupported,
    unsupported_type,
};
use nanoglia_core::{BuiltinOp, DataType, OpDesc, OpOptions, Result};

fn depth_space_params(op: &OpDesc<'_, '_>) -> Result<StridedCopy> {
    expect_arity(op, 1, 1)?;
    let OpOptions::BlockSize { block_size } = op.options else {
        return Err(unsupported(op));
    };
    let to_depth = match op.builtin() {
        Some(BuiltinOp::SPACE_TO_DEPTH) => true,
        Some(BuiltinOp::DEPTH_TO_SPACE) => false,
        _ => return Err(unsupported(op)),
    };
    let (input, output) = (input(op, 0)?, output(op, 0)?);
    // The types of LiteRT (TensorFlow Lite Micro: f32 and int8).
    if !matches!(
        input.dtype,
        DataType::F32 | DataType::U8 | DataType::I8 | DataType::I32 | DataType::I64
    ) {
        return Err(unsupported_type(input));
    }
    if output.dtype != input.dtype {
        return Err(unsupported_type(output));
    }
    let size = input.dtype.size().ok_or_else(|| unsupported_type(input))?;
    if block_size <= 0 {
        return Err(unsupported(op));
    }
    let len = elements(input)?;
    let [b, h, w, d] = dims4(input)?.map(i64::from);
    let out = dims4(output)?.map(i64::from);
    let bs = i64::from(block_size);
    let area = bs * bs;
    // Steps saturate: with a zero dimension, nothing is copied and `StridedCopy` ignores them;
    // otherwise they are below the input's element count.
    let m = |a: i64, b: i64| a.saturating_mul(b);
    // Output axes, outermost first: `(count, input step)`.
    let axes = if to_depth {
        if h % bs != 0
            || w % bs != 0
            || Some(out) != d.checked_mul(area).map(|od| [b, h / bs, w / bs, od])
        {
            return Err(shape_mismatch(output));
        }
        // [b, h / bs, w / bs, bs (rows), bs (columns), d]
        [
            (b, m(m(h, w), d)),
            (h / bs, m(m(bs, w), d)),
            (w / bs, m(bs, d)),
            (bs, m(w, d)),
            (bs, d),
            (d, 1),
        ]
    } else {
        if d % area != 0 || out != [b, h * bs, w * bs, d / area] {
            return Err(shape_mismatch(output));
        }
        let od = d / area;
        // [b, h, bs (rows), w, bs (columns), d / bs²]
        [
            (b, m(m(h, w), d)),
            (h, m(w, d)),
            (bs, m(bs, od)),
            (w, d),
            (bs, od),
            (od, 1),
        ]
    };
    let mut copy_axes = [(0usize, 0i64); 6];
    for (slot, (count, step)) in copy_axes.iter_mut().zip(axes) {
        *slot = (
            usize::try_from(count).map_err(|_| shape_mismatch(output))?,
            step,
        );
    }
    StridedCopy::new(&copy_axes, 0, len, size, shape_mismatch(input))
}

fixed_kernel!(
    /// `SPACE_TO_DEPTH` of `f32`, `u8`, `i8`, `i32` and `i64`.
    SPACE_TO_DEPTH,
    "rust/space_to_depth",
    depth_space_params,
    strided_eval
);

fixed_kernel!(
    /// `DEPTH_TO_SPACE` of `f32`, `u8`, `i8`, `i32` and `i64`.
    DEPTH_TO_SPACE,
    "rust/depth_to_space",
    depth_space_params,
    strided_eval
);
