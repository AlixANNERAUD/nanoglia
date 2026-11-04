//! MicroFlow: the model is compiled into code at build time (path relative to the workspace
//! root). Its tensors are values passed between the generated layers, on the stack.

#![no_std]
#![no_main]

use panic_halt as _;

#[microflow::model("../../testdata/mlperf-tiny/kws_ref_model.tflite")]
struct Model;

/// Output quantization of the model (scale 1/256, zero point -128): `predict_quantized` returns
/// dequantized values.
const OUTPUT_SCALE: f32 = 0.003_906_25;
const OUTPUT_ZERO_POINT: i32 = -128;

#[cortex_m_rt::entry]
fn main() -> ! {
    let input = fw_common::input();
    let buffer = [nalgebra::SMatrix::<[i8; 1], 49, 10>::from_fn(|r, c| {
        [input[r * 10 + c]]
    })];
    let y = Model::predict_quantized(buffer);
    let mut output = [0i8; fw_common::OUTPUT_LEN];
    for (o, &f) in output.iter_mut().zip(y.as_slice()) {
        *o = (round_half_away(f / OUTPUT_SCALE) as i32 + OUTPUT_ZERO_POINT) as i8;
    }
    fw_common::finish(&output)
}

/// Rounds half away from zero (`f32::round` needs `std`).
fn round_half_away(x: f32) -> f32 {
    if x >= 0.0 {
        (x + 0.5) as i32 as f32
    } else {
        (x - 0.5) as i32 as f32
    }
}
