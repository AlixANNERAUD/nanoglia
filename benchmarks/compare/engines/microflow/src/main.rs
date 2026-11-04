//! MicroFlow: the `model` macro turns each `.tflite` file into Rust code at build time (paths are
//! relative to the workspace root, where Cargo runs rustc). `predict_quantized` takes the int8
//! input and returns the dequantized `f32` output, which is quantized back to int8 with the
//! output's scale and zero point (exact: `scale * (q - zero_point)` is inverted by rounding).

/// A model with an NHWC int8 input of `$h` x `$w` x `$c`.
#[allow(unused_macros)]
macro_rules! image_model {
    ($feature:tt, $module:ident, $name:tt, $path:tt, $h:tt, $w:tt, $c:tt) => {
        #[cfg(feature = $feature)]
        mod $module {
            #[microflow::model($path)]
            pub struct Model;

            pub fn run() {
                let (scale, zero_point) = harness::output_quantization($name);
                harness::bench("MicroFlow", $name, |input, output| {
                    let buffer = [nalgebra::SMatrix::<[i8; $c], $h, $w>::from_fn(|r, c| {
                        core::array::from_fn(|ch| input[(r * $w + c) * $c + ch])
                    })];
                    let y = Model::predict_quantized(buffer);
                    super::requantize(y.as_slice(), scale, zero_point, output);
                });
            }
        }
    };
}

/// A model with a flat int8 input of `$n` values.
#[allow(unused_macros)]
macro_rules! vector_model {
    ($feature:tt, $module:ident, $name:tt, $path:tt, $n:tt) => {
        #[cfg(feature = $feature)]
        mod $module {
            #[microflow::model($path)]
            pub struct Model;

            pub fn run() {
                let (scale, zero_point) = harness::output_quantization($name);
                harness::bench("MicroFlow", $name, |input, output| {
                    let buffer = nalgebra::SMatrix::<i8, 1, $n>::from_fn(|_, c| input[c]);
                    let y = Model::predict_quantized(buffer);
                    super::requantize(y.as_slice(), scale, zero_point, output);
                });
            }
        }
    };
}

#[allow(dead_code)]
fn requantize(y: &[f32], scale: f32, zero_point: i32, output: &mut [i8]) {
    for (o, &f) in output.iter_mut().zip(y) {
        *o = ((f / scale).round() as i32 + zero_point) as i8;
    }
}

image_model!(
    "kws",
    kws,
    "kws_ref_model",
    "../../testdata/mlperf-tiny/kws_ref_model.tflite",
    49,
    10,
    1
);
image_model!(
    "vww",
    vww,
    "vww_96_int8",
    "../../testdata/mlperf-tiny/vww_96_int8.tflite",
    96,
    96,
    3
);
image_model!(
    "resnet",
    resnet,
    "pretrainedResnet_quant",
    "../../testdata/mlperf-tiny/pretrainedResnet_quant.tflite",
    32,
    32,
    3
);
vector_model!(
    "ad",
    ad,
    "ad01_int8",
    "../../testdata/mlperf-tiny/ad01_int8.tflite",
    640
);

fn main() {
    harness::run(|| {
        #[cfg(feature = "kws")]
        kws::run();
        #[cfg(not(feature = "kws"))]
        harness::skip("MicroFlow", "kws_ref_model", "not built (feature kws)");
        #[cfg(feature = "vww")]
        vww::run();
        #[cfg(not(feature = "vww"))]
        harness::skip("MicroFlow", "vww_96_int8", "not built (feature vww)");
        #[cfg(feature = "resnet")]
        resnet::run();
        #[cfg(not(feature = "resnet"))]
        harness::skip(
            "MicroFlow",
            "pretrainedResnet_quant",
            "build fails: the model macro rejects ADD (feature resnet)",
        );
        #[cfg(feature = "ad")]
        ad::run();
        #[cfg(not(feature = "ad"))]
        harness::skip("MicroFlow", "ad01_int8", "not built (feature ad)");
    });
}
