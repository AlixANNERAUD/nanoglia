//! `RESHAPE`, `SQUEEZE` and `EXPAND_DIMS`: the output holds the input's bytes unchanged.

use crate::util::{
    fixed_kernel, in_slice, input, out_slice, output, pod_struct, shape_mismatch, unsupported,
    unsupported_type,
};
use nanonn_core::backend::EvalCtx;
use nanonn_core::pod;
use nanonn_core::{BuiltinOp, OpDesc, Result};

pod_struct! {
    /// Parameters of a copy.
    pub(crate) struct CopyParams {
        /// Number of bytes to copy.
        bytes: u32,
    }
}

fn copy_params(op: &OpDesc<'_, '_>) -> Result<CopyParams> {
    // RESHAPE has an optional shape input, EXPAND_DIMS an axis input; the output shape is static.
    let max_inputs = match op.builtin() {
        Some(BuiltinOp::RESHAPE | BuiltinOp::EXPAND_DIMS) => 2,
        Some(BuiltinOp::SQUEEZE) => 1,
        _ => return Err(unsupported(op)),
    };
    if op.inputs.is_empty() || op.inputs.len() > max_inputs || op.outputs.len() != 1 {
        return Err(unsupported(op));
    }
    let (input, output) = (input(op, 0)?, output(op, 0)?);
    if input.dtype != output.dtype {
        return Err(unsupported_type(output));
    }
    let bytes = input.byte_len().ok_or_else(|| shape_mismatch(input))?;
    if output.byte_len() != Some(bytes) || input.dims.num_elements() != output.dims.num_elements() {
        return Err(shape_mismatch(output));
    }
    Ok(CopyParams {
        bytes: u32::try_from(bytes).map_err(|_| shape_mismatch(input))?,
    })
}

fn copy_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: CopyParams = *pod::read(ctx.persist)?;
    let len = p.bytes as usize;
    let input = in_slice::<u8>(ctx.inputs, 0, len)?;
    let output = out_slice::<u8>(ctx.outputs, 0, len)?;
    // The interpreter may give the output the input's memory: nothing to do then.
    if input.as_ptr() != output.as_ptr() {
        output.copy_from_slice(input);
    }
    Ok(())
}

fixed_kernel!(
    /// `RESHAPE`, `SQUEEZE` and `EXPAND_DIMS` of any type.
    RESHAPE,
    "rust/reshape",
    copy_params,
    copy_eval
);
