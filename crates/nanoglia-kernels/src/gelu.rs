//! `GELU` on `f32` (TFLite `activations.cc` `GeluEval`, `reference_ops::Gelu`): the exact
//! function `0.5 * x * erfc(-x / sqrt(2))`, or its tanh approximation
//! `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`, evaluated as Eigen does (see
//! [`crate::eigen`]).

use crate::eigen;
use crate::util::{
    expect_arity, expect_type, fixed_kernel, in_slice, input, out_slice, output, pod_struct,
    same_shape, unsupported,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod;
use nanoglia_core::{DataType, OpDesc, OpOptions, Result};

/// `sqrt(2 / pi)` as `float(M_2_SQRTPI * M_SQRT1_2)`.
const SQRT_2_OVER_PI: f32 = f32::from_bits(0x3f4c_422a);
/// `0.044715f`.
const CUBE_COEFFICIENT: f32 = f32::from_bits(0x3d37_2713);
/// `float(-M_SQRT1_2)`.
const NEG_SQRT_HALF: f32 = f32::from_bits(0xbf35_04f3);

pod_struct! {
    /// Parameters of a float GELU.
    pub(crate) struct GeluParams {
        len: u32,
        /// Whether the tanh approximation is computed.
        approximate: u32,
    }
}

fn gelu_params(op: &OpDesc<'_, '_>) -> Result<GeluParams> {
    let OpOptions::Gelu { approximate } = op.options else {
        return Err(unsupported(op));
    };
    expect_arity(op, 1, 1)?;
    let (input, output) = (input(op, 0)?, output(op, 0)?);
    expect_type(input, DataType::F32)?;
    expect_type(output, DataType::F32)?;
    let len = same_shape(input, output)? as u32;
    Ok(GeluParams {
        len,
        approximate: u32::from(approximate),
    })
}

/// `GeluTransformApproximate`, as Eigen evaluates it: `(0.5 * x) * (1 + tanh(...))`, products and
/// sums rounded one by one.
fn gelu_tanh(x: f32) -> f32 {
    let cube = x * x * x;
    (0.5 * x) * (1.0 + eigen::tanhf(SQRT_2_OVER_PI * (x + CUBE_COEFFICIENT * cube)))
}

/// `GeluTransform`, as Eigen evaluates it in a packet (`packet`) or in scalar code.
fn gelu_erf(x: f32, packet: bool) -> f32 {
    (0.5 * x) * eigen::erfcf(x * NEG_SQRT_HALF, packet)
}

fn gelu_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: GeluParams = *pod::read(ctx.persist)?;
    let len = p.len as usize;
    let input = in_slice::<f32>(ctx.inputs, 0, len)?;
    let output = out_slice::<f32>(ctx.outputs, 0, len)?;
    if p.approximate != 0 {
        for (y, &x) in output.iter_mut().zip(input) {
            *y = gelu_tanh(x);
        }
    } else {
        // Eigen's packets of four, then the scalar tail.
        let packed = len - len % 4;
        for (i, (y, &x)) in output.iter_mut().zip(input).enumerate() {
            *y = gelu_erf(x, i < packed);
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `GELU` on `f32`.
    GELU_F32,
    "rust/gelu_f32",
    gelu_params,
    gelu_eval
);
