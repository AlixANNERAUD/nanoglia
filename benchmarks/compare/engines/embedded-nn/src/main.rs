//! embedded-nn: `embedded_nn_model` imports each `.tflite` file at build time (path relative to
//! this crate's manifest), plans a static arena and generates the inference code. `predict`
//! copies the input into the caller's arena and returns the output slice.

#[allow(unused_macros)]
macro_rules! model {
    ($feature:tt, $module:ident, $name:tt, $path:tt) => {
        #[cfg(feature = $feature)]
        mod $module {
            #[embedded_nn_macros::embedded_nn_model($path)]
            pub struct Model;

            pub fn run() {
                let mut arena = vec![0u8; Model::ARENA_SIZE];
                let arena: &mut [u8; Model::ARENA_SIZE] = arena.as_mut_slice().try_into().unwrap();
                harness::bench("embedded-nn", $name, |input, output| {
                    output.copy_from_slice(Model::predict(input, arena).unwrap());
                });
            }
        }
    };
}

model!(
    "kws",
    kws,
    "kws_ref_model",
    "../../../../testdata/mlperf-tiny/kws_ref_model.tflite"
);
model!(
    "vww",
    vww,
    "vww_96_int8",
    "../../../../testdata/mlperf-tiny/vww_96_int8.tflite"
);
model!(
    "resnet",
    resnet,
    "pretrainedResnet_quant",
    "../../../../testdata/mlperf-tiny/pretrainedResnet_quant.tflite"
);
model!(
    "ad",
    ad,
    "ad01_int8",
    "../../../../testdata/mlperf-tiny/ad01_int8.tflite"
);

fn main() {
    harness::run(|| {
        #[cfg(feature = "kws")]
        kws::run();
        #[cfg(not(feature = "kws"))]
        harness::skip("embedded-nn", "kws_ref_model", "not built (feature kws)");
        #[cfg(feature = "vww")]
        vww::run();
        #[cfg(not(feature = "vww"))]
        harness::skip("embedded-nn", "vww_96_int8", "not built (feature vww)");
        #[cfg(feature = "resnet")]
        resnet::run();
        #[cfg(not(feature = "resnet"))]
        harness::skip(
            "embedded-nn",
            "pretrainedResnet_quant",
            "not built (feature resnet)",
        );
        #[cfg(feature = "ad")]
        ad::run();
        #[cfg(not(feature = "ad"))]
        harness::skip("embedded-nn", "ad01_int8", "not built (feature ad)");
    });
}
