//! `PAD` and `PADV2` of int8 tensors of rank 1 to 4 with constant paddings, through `arm_pad_s8`.
//!
//! Like LiteRT, the values are copied unchanged and the padding value is the output zero point
//! (`PAD`) or the constant value input (`PADV2`, whose zero point must be the output's).

use nanonn_cmsis_nn_sys as sys;
use nanonn_core::backend::{EvalCtx, Kernel, KernelPlan};
use nanonn_core::le::LeSlice;
use nanonn_core::pod::Pod;
use nanonn_core::{BuiltinOp, DataType, OpDesc, Result, TensorDesc};

use crate::common::{self, nhwc};
use crate::layout::{read_params, write_params};

/// The `PAD`/`PADV2` kernel.
pub(crate) static KERNEL: Kernel = Kernel {
    name: "cmsis-nn/pad_s8",
    plan,
    prepare,
    eval,
};

/// Persistent parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    /// The input shape, extended to rank 4 with leading ones.
    input: sys::cmsis_nn_dims,
    pre: sys::cmsis_nn_dims,
    post: sys::cmsis_nn_dims,
    output_len: i32,
    pad_value: i32,
}

// SAFETY: `#[repr(C)]` struct of `i32` fields only: no padding, any bit pattern is valid.
unsafe impl Pod for Params {}

/// The `(before, after)` padding of dimension `axis` from a constant `[rank, 2]` tensor.
fn padding(paddings: &TensorDesc<'_>, axis: usize) -> Option<(i32, i32)> {
    let data = paddings.data?;
    let (before, after) = match paddings.dtype {
        DataType::I32 => {
            let values = LeSlice::<i32>::new(data);
            (values.get(2 * axis)?, values.get(2 * axis + 1)?)
        }
        DataType::I64 => {
            let values = LeSlice::<i64>::new(data);
            (
                i32::try_from(values.get(2 * axis)?).ok()?,
                i32::try_from(values.get(2 * axis + 1)?).ok()?,
            )
        }
        _ => return None,
    };
    (before >= 0 && after >= 0).then_some((before, after))
}

fn analyze(op: &OpDesc<'_, '_>) -> Option<Params> {
    let v2 = match op.builtin()? {
        BuiltinOp::PAD => false,
        BuiltinOp::PADV2 => true,
        _ => return None,
    };
    if !(2..=3).contains(&op.inputs.len()) || op.outputs.len() != 1 {
        return None;
    }
    let input = op.input(0)?;
    let paddings = op.input(1)?;
    let output = op.output(0)?;
    if input.dtype != DataType::I8 || output.dtype != DataType::I8 {
        return None;
    }
    let rank = input.dims.rank();
    if !(1..=4).contains(&rank) || output.dims.rank() != rank {
        return None;
    }
    if paddings.dims.as_slice() != [rank as i32, 2] {
        return None;
    }
    let output_zero_point = output.quant.zero_point();
    if !(i8::MIN as i32..=i8::MAX as i32).contains(&output_zero_point) {
        return None;
    }
    let pad_value = match op.input(2) {
        Some(constant) if v2 => {
            if constant.dtype != DataType::I8
                || common::element_count(&constant.dims)? != 1
                || constant.quant.zero_point() != output_zero_point
            {
                return None;
            }
            *constant.data?.first()? as i8 as i32
        }
        Some(_) => return None,
        None => output_zero_point,
    };

    // Extend to rank 4 with leading dimensions of size 1 and no padding.
    let mut shape = [1i32; 4];
    let mut pre = [0i32; 4];
    let mut post = [0i32; 4];
    for axis in 0..rank {
        let slot = 4 - rank + axis;
        let size = input.dims.get(axis)?;
        let (before, after) = padding(paddings, axis)?;
        let total = size.checked_add(before)?.checked_add(after)?;
        if size <= 0 || output.dims.get(axis)? != total {
            return None;
        }
        (shape[slot], pre[slot], post[slot]) = (size, before, after);
    }
    common::element_count(&input.dims)?;
    let output_len = common::element_count(&output.dims)?;
    Some(Params {
        input: nhwc(shape[0], shape[1], shape[2], shape[3]),
        pre: nhwc(pre[0], pre[1], pre[2], pre[3]),
        post: nhwc(post[0], post[1], post[2], post[3]),
        output_len,
        pad_value,
    })
}

/// Whether this kernel runs `op`.
pub(crate) fn supports(op: &OpDesc<'_, '_>) -> bool {
    analyze(op).is_some()
}

fn plan(op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
    analyze(op).ok_or_else(|| common::unsupported(op))?;
    Ok(KernelPlan {
        persistent: core::mem::size_of::<Params>(),
        scratch: 0,
    })
}

fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
    let p = analyze(op).ok_or_else(|| common::unsupported(op))?;
    write_params(persist, p)
}

fn eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: Params = read_params(ctx.persist)?;
    let input = common::input(ctx.inputs, 0);
    common::check_len(input, common::dims_len(&p.input))?;
    let output = ctx.output(0);
    common::check_len(output, p.output_len)?;
    // SAFETY: the input holds the elements of `input`, the output those of the padded shape.
    let status = unsafe {
        sys::arm_pad_s8(
            input.as_ptr().cast(),
            output.as_mut_ptr().cast(),
            p.pad_value as i8,
            &p.input,
            &p.pre,
            &p.post,
        )
    };
    common::check_status(status)
}
