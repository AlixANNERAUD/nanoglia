//! ember-rs: the `model` macro turns each `.tflite` file into Rust code at build time (paths are
//! relative to the workspace root, where Cargo runs rustc), dispatched to the pure Rust reference
//! backend `RefBackend`. Activations are local arrays of the generated function.

#[allow(unused_macros)]
macro_rules! model {
    ($feature:tt, $module:ident, $name:tt, $path:tt) => {
        #[cfg(feature = $feature)]
        mod $module {
            #[ember_infer_macros::model($path)]
            pub struct Model;

            pub fn run() {
                let mut backend = ember_infer_ref::RefBackend;
                let mut scratch = vec![0u8; Model::scratch_len::<ember_infer_ref::RefBackend>()];
                harness::bench("ember-rs", $name, |input, output| {
                    Model::predict_quantized_with_scratch(
                        &mut backend,
                        input,
                        output,
                        &mut scratch,
                    )
                    .unwrap();
                });
            }
        }
    };
}

model!(
    "kws",
    kws,
    "kws_ref_model",
    "../../testdata/mlperf-tiny/kws_ref_model.tflite"
);
model!(
    "vww",
    vww,
    "vww_96_int8",
    "../../testdata/mlperf-tiny/vww_96_int8.tflite"
);
model!(
    "resnet",
    resnet,
    "pretrainedResnet_quant",
    "../../testdata/mlperf-tiny/pretrainedResnet_quant.tflite"
);
model!(
    "ad",
    ad,
    "ad01_int8",
    "../../testdata/mlperf-tiny/ad01_int8.tflite"
);

fn main() {
    harness::run(|| {
        #[cfg(feature = "kws")]
        kws::run();
        #[cfg(not(feature = "kws"))]
        harness::skip("ember-rs", "kws_ref_model", "not built (feature kws)");
        #[cfg(feature = "vww")]
        vww::run();
        #[cfg(not(feature = "vww"))]
        harness::skip("ember-rs", "vww_96_int8", "not built (feature vww)");
        #[cfg(feature = "resnet")]
        resnet::run();
        #[cfg(not(feature = "resnet"))]
        harness::skip(
            "ember-rs",
            "pretrainedResnet_quant",
            "not built (feature resnet)",
        );
        #[cfg(feature = "ad")]
        ad::run();
        #[cfg(not(feature = "ad"))]
        harness::skip("ember-rs", "ad01_int8", "not built (feature ad)");
    });
}
