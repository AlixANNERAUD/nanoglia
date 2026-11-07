//! `SPACE_TO_BATCH_ND` and `BATCH_TO_SPACE_ND` (TFLite `space_to_batch_nd.cc`,
//! `batch_to_space_nd.cc`, reference `SpaceToBatchND` and `BatchToSpaceND`), with a constant
//! block shape and constant paddings or crops, on rank 3 and 4 tensors as in TFLite.
//!
//! The operators move whole pixels (`depth` elements). `SPACE_TO_BATCH_ND` fills the padding
//! with the byte of the output zero point (8-bit types) or with zeros, as the reference kernel's
//! `memset` does. Quantized inputs and outputs must share their quantization, as LiteRT requires.

use crate::util::{
    constant, elements, expect_arity, fixed_kernel, input, output, per_tensor, pod_struct,
    read_index, shape_mismatch, size_mismatch, unsupported, unsupported_type,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::{BuiltinOp, DataType, Detail, Error, OpDesc, Result, TensorDesc, pod};

pod_struct! {
    /// Parameters of `SPACE_TO_BATCH_ND` and `BATCH_TO_SPACE_ND`, in TFLite's 4D view (a rank 3
    /// tensor `[b, h, d]` is `[b, h, 1, d]`).
    pub(crate) struct SpaceBatchParams {
        /// Input `[batch, height, width, depth]`.
        input: [u32; 4],
        /// Output `[batch, height, width, depth]`.
        output: [u32; 4],
        /// Block height and width.
        block: [u32; 2],
        /// Padding (`SPACE_TO_BATCH_ND`) or crop (`BATCH_TO_SPACE_ND`) before the height and
        /// the width.
        before: [u32; 2],
        /// Element size in bytes.
        size: u32,
        /// Byte written into the padding.
        pad: u32,
        /// 1 for `SPACE_TO_BATCH_ND`, 0 for `BATCH_TO_SPACE_ND`.
        to_batch: u32,
    }
}

/// The `n` values of a constant int32 tensor of shape `shape`.
fn constant_i32<const N: usize>(
    op: &OpDesc<'_, '_>,
    tensor: &TensorDesc<'_>,
    shape: &[i32],
    n: usize,
) -> Result<[i64; N]> {
    if tensor.dtype != DataType::I32 || tensor.dims.as_slice() != shape || n > N {
        return Err(unsupported(op));
    }
    let data = constant(op, tensor)?;
    let mut values = [0i64; N];
    for (i, v) in values[..n].iter_mut().enumerate() {
        *v = read_index(tensor, data, i).ok_or_else(|| unsupported_type(tensor))?;
    }
    Ok(values)
}

/// TFLite's 4D view of a rank 3 or 4 shape.
fn extend(t: &TensorDesc<'_>) -> Result<[i64; 4]> {
    elements(t)?;
    let d = t.dims.as_slice();
    if d.iter().any(|&x| x < 0) {
        return Err(shape_mismatch(t));
    }
    Ok(match *d {
        [b, h, w, c] => [b, h, w, c].map(i64::from),
        [b, h, c] => [i64::from(b), i64::from(h), 1, i64::from(c)],
        _ => return Err(unsupported_type(t)),
    })
}

fn space_batch_params(op: &OpDesc<'_, '_>) -> Result<SpaceBatchParams> {
    expect_arity(op, 3, 1)?;
    let to_batch = match op.builtin() {
        Some(BuiltinOp::SPACE_TO_BATCH_ND) => true,
        Some(BuiltinOp::BATCH_TO_SPACE_ND) => false,
        _ => return Err(unsupported(op)),
    };
    let (input, block, pads, output) =
        (input(op, 0)?, input(op, 1)?, input(op, 2)?, output(op, 0)?);
    // The types of LiteRT's reference registration (TensorFlow Lite Micro: f32 and int8).
    if !matches!(
        input.dtype,
        DataType::F32 | DataType::U8 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64
    ) {
        return Err(unsupported_type(input));
    }
    if output.dtype != input.dtype {
        return Err(unsupported_type(output));
    }
    let rank = input.dims.rank();
    if !(3..=4).contains(&rank) || output.dims.rank() != rank {
        return Err(unsupported(op));
    }
    let size = input.dtype.size().ok_or_else(|| unsupported_type(input))?;
    let mut pad = 0u8;
    if matches!(input.dtype, DataType::U8 | DataType::I8 | DataType::I16) {
        let (in_scale, in_zero) = per_tensor(input)?;
        let (out_scale, out_zero) = per_tensor(output)?;
        if in_scale != out_scale
            || in_zero != out_zero
            || (input.dtype == DataType::I16 && in_zero != 0)
        {
            return Err(unsupported(op));
        }
        // `memset(out, zero_point, ...)` keeps the low byte of the zero point.
        pad = out_zero as u8;
    }

    let spatial = rank - 2;
    let block = constant_i32::<2>(op, block, &[spatial as i32], spatial)?;
    let pads = constant_i32::<4>(op, pads, &[spatial as i32, 2], spatial * 2)?;
    if block[..spatial].iter().any(|&b| b < 1) || pads[..spatial * 2].iter().any(|&p| p < 0) {
        return Err(unsupported(op));
    }
    // A rank 3 tensor has no width: one block column, no padding.
    let (block_h, block_w) = (block[0], if spatial == 2 { block[1] } else { 1 });
    let (before_h, after_h) = (pads[0], pads[1]);
    let (before_w, after_w) = if spatial == 2 {
        (pads[2], pads[3])
    } else {
        (0, 0)
    };

    let x = extend(input)?;
    let y = extend(output)?;
    let blocks = block_h * block_w;
    // The output shape TFLite computes (`ResizeOutputTensor`).
    // TFLite computes the padded or uncropped sizes in `int`; larger ones are refused.
    let fits = |v: i64| v <= i64::from(i32::MAX);
    let expected = if to_batch {
        let h = x[1] + before_h + after_h;
        let w = x[2] + before_w + after_w;
        if !fits(h) || !fits(w) || h % block_h != 0 || w % block_w != 0 {
            return Err(shape_mismatch(output));
        }
        [
            x[0].checked_mul(blocks),
            Some(h / block_h),
            Some(w / block_w),
            Some(x[3]),
        ]
    } else {
        if x[0] % blocks != 0 {
            return Err(shape_mismatch(output));
        }
        let h = Some(x[1] * block_h)
            .filter(|&h| fits(h))
            .map(|h| h - before_h - after_h);
        let w = Some(x[2] * block_w)
            .filter(|&w| fits(w))
            .map(|w| w - before_w - after_w);
        // A negative size would be refused by TFLite's `ResizeTensor`.
        if h.is_none_or(|h| h < 0) || w.is_none_or(|w| w < 0) {
            return Err(shape_mismatch(output));
        }
        [Some(x[0] / blocks), h, w, Some(x[3])]
    };
    if expected != y.map(Some) {
        return Err(shape_mismatch(output));
    }
    let u = |v: i64| u32::try_from(v).map_err(|_| unsupported(op));
    Ok(SpaceBatchParams {
        input: [u(x[0])?, u(x[1])?, u(x[2])?, u(x[3])?],
        output: [u(y[0])?, u(y[1])?, u(y[2])?, u(y[3])?],
        block: [u(block_h)?, u(block_w)?],
        before: [u(before_h)?, u(before_w)?],
        size: size as u32,
        pad: u32::from(pad),
        to_batch: u32::from(to_batch),
    })
}

/// The `len` bytes of `buffer` at `at`.
fn pixel_mut(buffer: &mut [u8], len: usize, at: usize) -> Result<&mut [u8]> {
    buffer
        .get_mut(at..at + len)
        .ok_or(Error::invalid_argument(Detail::None))
}

fn space_batch_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: SpaceBatchParams = *pod::read(ctx.persist)?;
    let [ib, ih, iw, depth] = p.input.map(|v| v as usize);
    let [ob, oh, ow, _] = p.output.map(|v| v as usize);
    let [block_h, block_w] = p.block.map(|v| v as usize);
    let [top, left] = p.before.map(|v| v as usize);
    let pixel = depth * p.size as usize;
    let in_len = ib * ih * iw * pixel;
    let out_len = ob * oh * ow * pixel;
    let input = ctx.inputs.first().copied().unwrap_or(&[]);
    let input = input
        .get(..in_len)
        .ok_or_else(|| size_mismatch(in_len, input.len()))?;
    let output: &mut [u8] = match ctx.outputs.first_mut() {
        Some(output) => output,
        None => return Err(size_mismatch(out_len, 0)),
    };
    let actual = output.len();
    let output = output
        .get_mut(..out_len)
        .ok_or_else(|| size_mismatch(out_len, actual))?;
    // Parameters written by `prepare` keep every pixel inside the buffers; `get` turns a
    // corrupted parameter block into an error rather than a panic.
    let corrupted = || Error::invalid_argument(Detail::None);
    let pixel_at = |b: usize, h: usize, w: usize, height: usize, width: usize| {
        ((b * height + h) * width + w) * pixel
    };
    let source = |src: usize| input.get(src..src + pixel).ok_or_else(corrupted);
    if p.to_batch != 0 {
        if ib == 0 {
            return Ok(());
        }
        // Reference `SpaceToBatchND`, output pixel by output pixel.
        for out_b in 0..ob {
            let in_b = out_b % ib;
            let shift_w = (out_b / ib) % block_w;
            let shift_h = (out_b / ib) / block_w;
            for y in 0..oh {
                let h = y * block_h + shift_h;
                for x in 0..ow {
                    let w = x * block_w + shift_w;
                    let dst = pixel_mut(output, pixel, pixel_at(out_b, y, x, oh, ow))?;
                    if h < top || h >= top + ih || w < left || w >= left + iw {
                        dst.fill(p.pad as u8);
                    } else {
                        dst.copy_from_slice(source(pixel_at(in_b, h - top, w - left, ih, iw))?);
                    }
                }
            }
        }
    } else {
        if ob == 0 {
            return Ok(());
        }
        // Reference `BatchToSpaceND`, input pixel by input pixel; cropped pixels are skipped.
        for in_b in 0..ib {
            let out_b = in_b % ob;
            let offset = in_b / ob;
            for y in 0..ih {
                let h = match (y * block_h + offset / block_w).checked_sub(top) {
                    Some(h) if h < oh => h,
                    _ => continue,
                };
                for x in 0..iw {
                    let w = match (x * block_w + offset % block_w).checked_sub(left) {
                        Some(w) if w < ow => w,
                        _ => continue,
                    };
                    let dst = pixel_mut(output, pixel, pixel_at(out_b, h, w, oh, ow))?;
                    dst.copy_from_slice(source(pixel_at(in_b, y, x, ih, iw))?);
                }
            }
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `SPACE_TO_BATCH_ND` of `f32`, `u8`, `i8`, `i16`, `i32` and `i64`, rank 3 or 4.
    SPACE_TO_BATCH_ND,
    "rust/space_to_batch_nd",
    space_batch_params,
    space_batch_eval
);

fixed_kernel!(
    /// `BATCH_TO_SPACE_ND` of `f32`, `u8`, `i8`, `i16`, `i32` and `i64`, rank 3 or 4.
    BATCH_TO_SPACE_ND,
    "rust/batch_to_space_nd",
    space_batch_params,
    space_batch_eval
);
