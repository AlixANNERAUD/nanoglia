//! Parses arbitrary bytes as a TFLite model and reads everything the reader exposes.
//!
//! Besides not panicking, a successfully parsed model must never report a malformed structure
//! from its accessors: `Model::parse` verifies everything up front.

#![no_main]

use libfuzzer_sys::fuzz_target;
use nanonn_core::{Detail, Error};
use nanonn_tflite::Model;

fn not_malformed(err: Error) {
    assert!(
        !matches!(err.detail, Detail::MalformedModel { .. }),
        "malformed error after a successful parse: {err}"
    );
}

fuzz_target!(|bytes: &[u8]| {
    let Ok(model) = Model::parse(bytes) else {
        return;
    };
    let _ = model.description().len();
    for s in 0..model.subgraph_count() {
        let graph = match model.subgraph(s) {
            Ok(graph) => graph,
            Err(err) => return not_malformed(err),
        };
        let _ = graph.name().len();
        for t in graph.inputs().iter().chain(graph.outputs().iter()) {
            assert!(t >= 0 && (t as usize) < graph.tensor_count());
        }
        for i in 0..graph.tensor_count() {
            match graph.tensor(i) {
                Ok(t) => {
                    assert_eq!(t.index as usize, i);
                    let _ = t
                        .quant
                        .scale
                        .iter()
                        .map(f32::to_bits)
                        .fold(0, u32::wrapping_add);
                    let _ = t.quant.zero_point.iter().fold(0, i64::wrapping_add);
                    let _ = t.data.map(|d| (d.first(), d.last()));
                    let _ = t.byte_len();
                }
                Err(err) => not_malformed(err),
            }
            if let Err(err) = graph.tensor_is_variable(i) {
                not_malformed(err);
            }
        }
        for i in 0..graph.operator_count() {
            match graph.operator(i) {
                Ok(op) => {
                    for t in op.inputs.iter() {
                        assert!(t == -1 || (t >= 0 && (t as usize) < graph.tensor_count()));
                    }
                    for t in op.outputs.iter() {
                        assert!(t >= 0 && (t as usize) < graph.tensor_count());
                    }
                    let _ = format!("{op:?}");
                }
                Err(err) => not_malformed(err),
            }
        }
    }
});
