//! `ADD` and `MUL` of two int8 tensors of the same shape, through
//! `esp_nn_{add,mul}_elementwise_s8`. Broadcasting is left to other backends.
//!
//! Port of esp-tflite-micro's `kernels/esp_nn/{add,mul}.cc` with TFLite Micro's
//! `CalculateOpDataAdd` (general path, `left_shift = 20`) and `CalculateOpDataMul`.

use nanoglia_core::backend::{EvalCtx, FnKernel, KernelPlan};
use nanoglia_core::pod::Pod;
use nanoglia_core::quant;
use nanoglia_core::{BuiltinOp, OpDesc, OpOptions, Result};
use nanoglia_esp_nn_sys as sys;

use crate::common;
use crate::fallback::{self, EspOp};
use crate::layout::{read_params, write_params};

/// The `ADD` kernel.
pub(crate) static ADD: FnKernel = fallback::kernel::<Binary<false>>();

/// The `MUL` kernel.
pub(crate) static MUL: FnKernel = fallback::kernel::<Binary<true>>();

/// ESP-NN parameters (the input multipliers are only used by `ADD`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Params {
    len: i32,
    input1_offset: i32,
    input1_multiplier: i32,
    input1_shift: i32,
    input2_offset: i32,
    input2_multiplier: i32,
    input2_shift: i32,
    left_shift: i32,
    output_offset: i32,
    output_multiplier: i32,
    output_shift: i32,
    activation_min: i32,
    activation_max: i32,
}

// SAFETY: `#[repr(C)]` struct of `i32` fields only: no padding, any bit pattern is valid.
unsafe impl Pod for Params {}

/// The left shift applied to 8-bit inputs before rescaling them (TFLite `ADD`).
const ADD_LEFT_SHIFT: i32 = 20;

fn analyze(op: &OpDesc<'_, '_>, mul: bool) -> Option<Params> {
    let code = if mul { BuiltinOp::MUL } else { BuiltinOp::ADD };
    if op.builtin()? != code {
        return None;
    }
    let OpOptions::Elementwise { activation } = op.options else {
        return None;
    };
    if op.inputs.len() != 2 || op.outputs.len() != 1 {
        return None;
    }
    let input1 = op.input(0)?;
    let input2 = op.input(1)?;
    let output = op.output(0)?;
    let (scale1, zero_point1) = common::activation_quant(input1)?;
    let (scale2, zero_point2) = common::activation_quant(input2)?;
    let (output_scale, output_zero_point) = common::activation_quant(output)?;
    if input1.dims != input2.dims || input1.dims != output.dims {
        return None;
    }
    let len = common::element_count(&output.dims)?;
    let (activation_min, activation_max) =
        common::activation_range(activation, output_scale, output_zero_point)?;

    let mut p = Params {
        len,
        input1_offset: -zero_point1,
        input2_offset: -zero_point2,
        output_offset: output_zero_point,
        activation_min,
        activation_max,
        ..Params::default()
    };
    if mul {
        let real = f64::from(scale1) * f64::from(scale2) / f64::from(output_scale);
        let m = quant::quantize_multiplier(real);
        (p.output_multiplier, p.output_shift) = (m.multiplier, m.shift);
    } else {
        let twice_max_input_scale = 2.0 * f64::from(scale1.max(scale2));
        let real_input1 = f64::from(scale1) / twice_max_input_scale;
        let real_input2 = f64::from(scale2) / twice_max_input_scale;
        let real_output =
            twice_max_input_scale / (f64::from(1i32 << ADD_LEFT_SHIFT) * f64::from(output_scale));
        // "Smaller than one": the shifts are at most 0, as ESP-NN expects.
        let m1 = quant::quantize_multiplier_smaller_than_one(real_input1)?;
        let m2 = quant::quantize_multiplier_smaller_than_one(real_input2)?;
        let mo = quant::quantize_multiplier_smaller_than_one(real_output)?;
        p.left_shift = ADD_LEFT_SHIFT;
        (p.input1_multiplier, p.input1_shift) = (m1.multiplier, m1.shift);
        (p.input2_multiplier, p.input2_shift) = (m2.multiplier, m2.shift);
        (p.output_multiplier, p.output_shift) = (mo.multiplier, mo.shift);
    }
    Some(p)
}

/// `ADD` (`MUL = false`) or `MUL` (`MUL = true`) on ESP-NN.
pub(crate) struct Binary<const MUL: bool>;

impl<const MUL: bool> EspOp for Binary<MUL> {
    const NAME: &'static str = if MUL {
        "esp-nn/mul_s8"
    } else {
        "esp-nn/add_s8"
    };

    fn fallback() -> &'static FnKernel {
        if MUL {
            &nanoglia_cmsis_nn::kernels::MUL_S8
        } else {
            &nanoglia_cmsis_nn::kernels::ADD_S8
        }
    }

    fn plan(op: &OpDesc<'_, '_>) -> Option<KernelPlan> {
        analyze(op, MUL)?;
        Some(KernelPlan {
            persistent: core::mem::size_of::<Params>(),
            scratch: 0,
        })
    }

    fn prepare(op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
        let p = analyze(op, MUL).ok_or_else(|| common::unsupported(op))?;
        write_params(persist, p)
    }

    fn run(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
        let p: Params = read_params(ctx.persist)?;
        let input1 = common::input(ctx.inputs, 0);
        let input2 = common::input(ctx.inputs, 1);
        common::check_len(input1, p.len)?;
        common::check_len(input2, p.len)?;
        let output = ctx.output(0);
        common::check_len(output, p.len)?;
        let (input1, input2, output) = (
            input1.as_ptr().cast(),
            input2.as_ptr().cast(),
            output.as_mut_ptr().cast(),
        );
        // SAFETY: the three buffers hold `len` elements.
        unsafe {
            if MUL {
                sys::nanoglia_esp_nn_mul_elementwise_s8(
                    input1,
                    input2,
                    p.input1_offset,
                    p.input2_offset,
                    output,
                    p.output_offset,
                    p.output_multiplier,
                    p.output_shift,
                    p.activation_min,
                    p.activation_max,
                    p.len,
                );
            } else {
                sys::nanoglia_esp_nn_add_elementwise_s8(
                    input1,
                    input2,
                    p.input1_offset,
                    p.input2_offset,
                    p.input1_multiplier,
                    p.input2_multiplier,
                    p.input1_shift,
                    p.input2_shift,
                    p.left_shift,
                    output,
                    p.output_offset,
                    p.output_multiplier,
                    p.output_shift,
                    p.activation_min,
                    p.activation_max,
                    p.len,
                );
            }
        }
        Ok(())
    }
}
