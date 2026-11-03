//! `PAD` and `PADV2` (TFLite `pad.cc`, reference `PadImpl`), for any element type.

use crate::util::{
    constant, elements, fixed_kernel, in_slice, input, int_range, out_slice, output, pod_struct,
    read_index, shape_mismatch, unsupported, unsupported_type,
};
use nanonn_core::backend::EvalCtx;
use nanonn_core::pod;
use nanonn_core::{BuiltinOp, DataType, MAX_RANK, OpDesc, Result};

pod_struct! {
    /// Parameters of `PAD`.
    pub(crate) struct PadParams {
        /// Input dimensions, left-padded with ones to [`MAX_RANK`].
        in_dims: [u32; MAX_RANK],
        /// Output dimensions, left-padded with ones to [`MAX_RANK`].
        out_dims: [u32; MAX_RANK],
        /// Padding before each dimension.
        before: [u32; MAX_RANK],
        /// Element size in bytes.
        size: u32,
        /// Whether the pad value is read from input 2 at evaluation time.
        value_from_input: u32,
        /// The pad value as raw bytes (first `size` bytes).
        value: [u8; 8],
    }
}

fn pad_params(op: &OpDesc<'_, '_>) -> Result<PadParams> {
    let v2 = match op.builtin() {
        Some(BuiltinOp::PAD) => false,
        Some(BuiltinOp::PADV2) => true,
        _ => return Err(unsupported(op)),
    };
    let max_inputs = if v2 { 3 } else { 2 };
    if op.inputs.len() < 2 || op.inputs.len() > max_inputs || op.outputs.len() != 1 {
        return Err(unsupported(op));
    }
    let (input, paddings, output) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    if input.dtype != output.dtype {
        return Err(unsupported_type(output));
    }
    let size = input.dtype.size().ok_or_else(|| unsupported_type(input))?;
    elements(input)?;
    elements(output)?;
    let rank = input.dims.rank();
    if output.dims.rank() != rank || paddings.dims.as_slice() != [rank as i32, 2] {
        return Err(shape_mismatch(paddings));
    }
    let padding_data = constant(op, paddings)?;

    let mut params = PadParams {
        in_dims: [1; MAX_RANK],
        out_dims: [1; MAX_RANK],
        before: [0; MAX_RANK],
        size: size as u32,
        value_from_input: 0,
        value: [0; 8],
    };
    let offset = MAX_RANK - rank;
    for d in 0..rank {
        let before =
            read_index(paddings, padding_data, 2 * d).ok_or_else(|| unsupported_type(paddings))?;
        let after = read_index(paddings, padding_data, 2 * d + 1)
            .ok_or_else(|| unsupported_type(paddings))?;
        if before < 0 || after < 0 {
            return Err(shape_mismatch(paddings));
        }
        let (in_dim, out_dim) = (input.dims.dim(d) as i64, output.dims.dim(d) as i64);
        if in_dim + before + after != out_dim {
            return Err(shape_mismatch(output));
        }
        params.in_dims[offset + d] = in_dim as u32;
        params.out_dims[offset + d] = out_dim as u32;
        params.before[offset + d] = before as u32;
    }

    match op.input(2) {
        Some(value) => {
            if value.dtype != input.dtype || value.dims.num_elements() != Some(1) {
                return Err(unsupported_type(value));
            }
            if int_range(input.dtype).is_some()
                && (value.quant.scale() != output.quant.scale()
                    || value.quant.zero_point() != output.quant.zero_point())
            {
                return Err(unsupported(op));
            }
            match value.data {
                Some(data) => {
                    let bytes = data.get(..size).ok_or_else(|| shape_mismatch(value))?;
                    params.value[..size].copy_from_slice(bytes);
                }
                None => params.value_from_input = 1,
            }
        }
        None => {
            // Quantized integers pad with the zero point, everything else with zero bits.
            let zero_point = i64::from(output.quant.zero_point());
            let quantized = matches!(
                input.dtype,
                DataType::I8 | DataType::U8 | DataType::I16 | DataType::I32 | DataType::I64
            );
            if quantized {
                if let Some((min, max)) = int_range(input.dtype)
                    && !(i64::from(min)..=i64::from(max)).contains(&zero_point)
                {
                    return Err(unsupported(op));
                }
                params.value = zero_point.to_le_bytes();
            }
        }
    }
    Ok(params)
}

fn pad_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: PadParams = *pod::read(ctx.persist)?;
    let size = p.size as usize;
    let mut value = p.value;
    if p.value_from_input != 0 {
        value[..size].copy_from_slice(in_slice::<u8>(ctx.inputs, 2, size)?);
    }
    let value = &value[..size];
    let count = |dims: &[u32; MAX_RANK]| dims.iter().map(|&d| d as usize).product::<usize>();
    let input = in_slice::<u8>(ctx.inputs, 0, count(&p.in_dims) * size)?;
    let output = out_slice::<u8>(ctx.outputs, 0, count(&p.out_dims) * size)?;

    const LAST: usize = MAX_RANK - 1;
    let row_out = p.out_dims[LAST] as usize * size;
    let row_in = p.in_dims[LAST] as usize * size;
    let before = p.before[LAST] as usize * size;
    let fill = |dst: &mut [u8]| {
        dst.chunks_exact_mut(size)
            .for_each(|e| e.copy_from_slice(value))
    };

    if row_out == 0 {
        return Ok(());
    }
    let mut index = [0u32; LAST];
    let mut src = 0;
    for row in output.chunks_exact_mut(row_out) {
        let inside =
            (0..LAST).all(|d| index[d] >= p.before[d] && index[d] < p.before[d] + p.in_dims[d]);
        if inside {
            fill(&mut row[..before]);
            row[before..before + row_in].copy_from_slice(&input[src..src + row_in]);
            fill(&mut row[before + row_in..]);
            src += row_in;
        } else {
            fill(row);
        }
        for d in (0..LAST).rev() {
            index[d] += 1;
            if index[d] < p.out_dims[d] {
                break;
            }
            index[d] = 0;
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `PAD` and `PADV2` of any type with whole-byte elements.
    PAD,
    "rust/pad",
    pad_params,
    pad_eval
);
