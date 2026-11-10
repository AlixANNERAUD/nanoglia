//! Checks the `op-*` features: a model loads and gives LiteRT's bytes when the features of all its
//! operators are on, and is refused with `UnsupportedOp`, at its first operator without a
//! feature, otherwise; `ModelReport` then reports that operator without a kernel.
//!
//! The test adapts to the features it is built with. `mise run op-features` runs it with the
//! operators of the keyword-spotting model only (that model runs, the ResNet one is refused at
//! its first `ADD`) and with no operator at all (both are refused at their first operator); the
//! default build runs both models.

use nanoglia::core::allocator_api2::alloc::Global;
use nanoglia::{BuiltinOp, Detail, Encoding, ErrorCode, Graph, ModelReport, OpCode, Target};

mod common;

use common::{Aligned, repo_root};

/// A model of `testdata/mlperf-tiny`, with its operators in order of first use and whether the
/// feature of each is on.
struct Model {
    name: &'static str,
    operators: &'static [(BuiltinOp, bool)],
}

const KWS: Model = Model {
    name: "kws_ref_model",
    operators: &[
        (BuiltinOp::CONV_2D, cfg!(feature = "op-conv-2d")),
        (
            BuiltinOp::DEPTHWISE_CONV_2D,
            cfg!(feature = "op-depthwise-conv-2d"),
        ),
        (
            BuiltinOp::AVERAGE_POOL_2D,
            cfg!(feature = "op-average-pool-2d"),
        ),
        (BuiltinOp::RESHAPE, cfg!(feature = "op-reshape")),
        (
            BuiltinOp::FULLY_CONNECTED,
            cfg!(feature = "op-fully-connected"),
        ),
        (BuiltinOp::SOFTMAX, cfg!(feature = "op-softmax")),
    ],
};

const RESNET: Model = Model {
    name: "pretrainedResnet_quant",
    operators: &[
        (BuiltinOp::CONV_2D, cfg!(feature = "op-conv-2d")),
        (BuiltinOp::ADD, cfg!(feature = "op-add")),
        (
            BuiltinOp::AVERAGE_POOL_2D,
            cfg!(feature = "op-average-pool-2d"),
        ),
        (BuiltinOp::RESHAPE, cfg!(feature = "op-reshape")),
        (
            BuiltinOp::FULLY_CONNECTED,
            cfg!(feature = "op-fully-connected"),
        ),
        (BuiltinOp::SOFTMAX, cfg!(feature = "op-softmax")),
    ],
};

/// What every operator of the model at `bytes` is, and whether it has a kernel.
fn report(bytes: &[u8]) -> (ModelReport, Vec<(BuiltinOp, bool)>) {
    let mut operators = Vec::new();
    let report =
        ModelReport::check_in(&[bytes], Encoding::Autodetect, Target::Auto, Global, |op| {
            let OpCode::Builtin(builtin) = op.opcode() else {
                panic!("custom operator");
            };
            operators.push((builtin, op.kernel().is_ok()));
        })
        .unwrap();
    (report, operators)
}

fn check(model: &Model) {
    let directory = repo_root().join("testdata/mlperf-tiny");
    let bytes = Aligned::read(&directory.join(format!("{}.tflite", model.name)));
    let enabled = |op: BuiltinOp| {
        let (_, enabled) = model.operators.iter().find(|(o, _)| *o == op).unwrap();
        *enabled
    };

    // The report: an operator has a kernel exactly when its feature is on.
    let (summary, operators) = report(bytes.bytes());
    for (index, (op, has_kernel)) in operators.iter().enumerate() {
        assert_eq!(
            *has_kernel,
            enabled(*op),
            "{}: operator {index} ({})",
            model.name,
            op.name().unwrap()
        );
    }
    let missing = operators.iter().position(|(_, has_kernel)| !has_kernel);
    assert_eq!(summary.runs(), missing.is_none());

    // Loading: refused at the first operator without a kernel, as any unsupported operator.
    let graph = Graph::load_in(&[bytes.bytes()], Encoding::Autodetect, Target::Auto, Global);
    if let Some(index) = missing {
        let Err(error) = graph else {
            panic!("{}: the model must be refused", model.name);
        };
        assert_eq!(error.code, ErrorCode::UnsupportedOperation);
        assert_eq!(
            error.detail,
            Detail::UnsupportedOp {
                op_index: index as u32,
                opcode: operators[index].0.0,
            }
        );
        return;
    }

    // Every feature is on: the model gives LiteRT's bytes.
    let graph = graph.unwrap();
    let mut context = graph.init_execution_context_in(Global).unwrap();
    for sample in 0..3 {
        let case = directory.join(format!("golden/{}/sample_{sample}", model.name));
        let input = std::fs::read(case.join("input_0.bin")).unwrap();
        let expected = std::fs::read(case.join("output_0.bin")).unwrap();
        context.set_input(0, &input).unwrap();
        context.compute().unwrap();
        assert_eq!(
            context.output(0).unwrap().bytes(),
            expected,
            "{}: sample {sample}",
            model.name
        );
    }
}

#[test]
fn keyword_spotting_model() {
    check(&KWS);
}

#[test]
fn resnet_model() {
    check(&RESNET);
}
