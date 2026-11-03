//! Reads the MLPerf Tiny int8 reference models.

use std::collections::BTreeMap;

use nanonn_core::{Activation, BuiltinOp, DataType, OpCode, OpOptions, Padding, TensorDesc};
use nanonn_tflite::{Model, Operator, SubGraph};

fn load(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/../../testdata/mlperf-tiny/{name}.tflite",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn operators<'m>(graph: &SubGraph<'m>) -> Vec<Operator<'m>> {
    (0..graph.operator_count())
        .map(|i| graph.operator(i).unwrap())
        .collect()
}

fn histogram(ops: &[Operator<'_>]) -> BTreeMap<i32, usize> {
    let mut counts = BTreeMap::new();
    for op in ops {
        let OpCode::Builtin(code) = op.code else {
            panic!("unexpected custom operator {:?}", op.code)
        };
        *counts.entry(code.0).or_default() += 1;
    }
    counts
}

fn io<'m>(graph: &SubGraph<'m>) -> (TensorDesc<'m>, TensorDesc<'m>) {
    assert_eq!(graph.inputs().len(), 1);
    assert_eq!(graph.outputs().len(), 1);
    let input = graph
        .tensor(graph.inputs().get(0).unwrap() as usize)
        .unwrap();
    let output = graph
        .tensor(graph.outputs().get(0).unwrap() as usize)
        .unwrap();
    (input, output)
}

/// Walks every tensor and operator and checks the invariants the reader guarantees.
fn check_graph(graph: &SubGraph<'_>) {
    for i in 0..graph.tensor_count() {
        let t = graph.tensor(i).unwrap();
        assert_eq!(t.index as usize, i);
        assert!(!graph.tensor_is_variable(i).unwrap());
        if let Some(data) = t.data {
            assert_eq!(Some(data.len()), t.byte_len(), "tensor {i} ({})", t.name);
        }
        if t.quant.is_quantized() {
            assert_eq!(t.quant.scale.len(), t.quant.zero_point.len());
        }
    }
    for op in operators(graph) {
        // The MLPerf Tiny models have no omitted optional input.
        for t in op.inputs.iter().chain(op.outputs.iter()) {
            assert!(t >= 0 && (t as usize) < graph.tensor_count());
        }
    }
    assert!(graph.tensor(graph.tensor_count()).is_err());
    assert!(graph.operator(graph.operator_count()).is_err());
}

/// Checks that each convolution filter is quantized per output channel.
fn check_per_channel_filters(graph: &SubGraph<'_>) {
    let mut checked = 0;
    for op in operators(graph) {
        let channel_axis = match op.code {
            OpCode::Builtin(BuiltinOp::CONV_2D) => 0,
            OpCode::Builtin(BuiltinOp::DEPTHWISE_CONV_2D) => 3,
            _ => continue,
        };
        let filter = graph.tensor(op.inputs.get(1).unwrap() as usize).unwrap();
        assert_eq!(filter.dtype, DataType::I8);
        assert!(filter.data.is_some());
        let channels = filter.dims.dim(channel_axis);
        assert_eq!(filter.quant.scale.len(), channels, "{}", filter.name);
        assert_eq!(filter.quant.zero_point.len(), channels);
        assert_eq!(filter.quant.axis, channel_axis as i32);
        assert!(filter.quant.zero_point.iter().all(|z| z == 0));
        let bias = graph.tensor(op.inputs.get(2).unwrap() as usize).unwrap();
        assert_eq!(bias.dtype, DataType::I32);
        assert_eq!(bias.quant.scale.len(), channels);
        checked += 1;
    }
    assert!(checked > 0);
}

#[test]
fn rejects_non_tflite() {
    assert!(!Model::is_tflite(b"TFL3"));
    assert!(Model::parse(b"").is_err());
    assert!(Model::parse(b"\0\0\0\0TFL2\0\0\0\0").is_err());
}

#[test]
fn kws_ref_model() {
    let bytes = load("kws_ref_model");
    assert!(Model::is_tflite(&bytes));
    let model = Model::parse(&bytes).unwrap();
    assert_eq!(model.version(), 3);
    assert_eq!(model.subgraph_count(), 1);
    assert!(model.subgraph(1).is_err());
    let graph = model.subgraph(0).unwrap();
    check_graph(&graph);

    let ops = operators(&graph);
    let expected = BTreeMap::from([
        (BuiltinOp::CONV_2D.0, 5),
        (BuiltinOp::DEPTHWISE_CONV_2D.0, 4),
        (BuiltinOp::AVERAGE_POOL_2D.0, 1),
        (BuiltinOp::RESHAPE.0, 1),
        (BuiltinOp::FULLY_CONNECTED.0, 1),
        (BuiltinOp::SOFTMAX.0, 1),
    ]);
    assert_eq!(histogram(&ops), expected);

    let (input, output) = io(&graph);
    assert_eq!(input.name, "input_1");
    assert_eq!(input.dims.as_slice(), [1, 49, 10, 1]);
    assert_eq!(input.dtype, DataType::I8);
    assert!(input.data.is_none());
    assert!(input.quant.is_per_tensor());
    assert_eq!(output.name, "Identity");
    assert_eq!(output.dims.as_slice(), [1, 12]);
    assert_eq!(output.dtype, DataType::I8);
    // TFLite softmax int8 outputs are quantized with scale 1/256 and zero point -128.
    assert_eq!(output.quant.scale(), 1.0 / 256.0);
    assert_eq!(output.quant.zero_point(), -128);

    // The first layer is a 10x4 convolution with stride 2 and a fused ReLU.
    let OpOptions::Conv2D(conv) = ops[0].options else {
        panic!("{:?}", ops[0])
    };
    assert_eq!(conv.padding, Padding::Same);
    assert_eq!((conv.stride_w, conv.stride_h), (2, 2));
    assert_eq!((conv.dilation_w, conv.dilation_h), (1, 1));
    assert_eq!(conv.activation, Activation::Relu);
    let filter = graph
        .tensor(ops[0].inputs.get(1).unwrap() as usize)
        .unwrap();
    assert_eq!(filter.dims.as_slice(), [64, 10, 4, 1]);
    assert_eq!(ops[0].inputs.len(), 3);
    assert_eq!(ops[0].outputs.len(), 1);

    let dw = ops
        .iter()
        .find_map(|op| match op.options {
            OpOptions::DepthwiseConv2D(o) => Some(o),
            _ => None,
        })
        .unwrap();
    assert_eq!(dw.padding, Padding::Same);
    assert_eq!((dw.stride_w, dw.stride_h), (1, 1));
    assert_eq!(dw.depth_multiplier, 1);
    assert_eq!(dw.activation, Activation::Relu);

    let pool = ops
        .iter()
        .find_map(|op| match op.options {
            OpOptions::Pool2D(o) => Some(o),
            _ => None,
        })
        .unwrap();
    assert_eq!(pool.padding, Padding::Valid);
    assert_eq!((pool.filter_w, pool.filter_h), (5, 25));
    assert_eq!(pool.activation, Activation::None);

    let last = ops.last().unwrap();
    assert_eq!(last.code, OpCode::Builtin(BuiltinOp::SOFTMAX));
    assert_eq!(last.options, OpOptions::Softmax { beta: 1.0 });
    let fc = &ops[ops.len() - 2];
    let OpOptions::FullyConnected(fc_options) = fc.options else {
        panic!("{fc:?}")
    };
    assert!(fc_options.default_weights_format);
    assert_eq!(fc_options.activation, Activation::None);
    assert_eq!(
        ops.iter()
            .find(|op| op.code == OpCode::Builtin(BuiltinOp::RESHAPE))
            .unwrap()
            .options,
        OpOptions::None
    );

    check_per_channel_filters(&graph);
}

#[test]
fn pretrained_resnet_quant() {
    let bytes = load("pretrainedResnet_quant");
    let model = Model::parse(&bytes).unwrap();
    assert_eq!(model.subgraph_count(), 1);
    let graph = model.subgraph(0).unwrap();
    check_graph(&graph);
    let ops = operators(&graph);
    let expected = BTreeMap::from([
        (BuiltinOp::CONV_2D.0, 9),
        (BuiltinOp::ADD.0, 3),
        (BuiltinOp::AVERAGE_POOL_2D.0, 1),
        (BuiltinOp::RESHAPE.0, 1),
        (BuiltinOp::FULLY_CONNECTED.0, 1),
        (BuiltinOp::SOFTMAX.0, 1),
    ]);
    assert_eq!(histogram(&ops), expected);

    let (input, output) = io(&graph);
    assert_eq!(input.dims.as_slice(), [1, 32, 32, 3]);
    assert_eq!(input.dtype, DataType::I8);
    assert_eq!(output.dims.as_slice(), [1, 10]);
    assert_eq!(output.dtype, DataType::I8);

    // Residual additions fuse a ReLU.
    for op in ops
        .iter()
        .filter(|op| op.code == OpCode::Builtin(BuiltinOp::ADD))
    {
        assert_eq!(
            op.options,
            OpOptions::Elementwise {
                activation: Activation::Relu
            }
        );
        assert_eq!(op.inputs.len(), 2);
    }
    check_per_channel_filters(&graph);
}

#[test]
fn vww_96_int8() {
    let bytes = load("vww_96_int8");
    let model = Model::parse(&bytes).unwrap();
    let graph = model.subgraph(0).unwrap();
    check_graph(&graph);
    let ops = operators(&graph);
    let expected = BTreeMap::from([
        (BuiltinOp::CONV_2D.0, 14),
        (BuiltinOp::DEPTHWISE_CONV_2D.0, 13),
        (BuiltinOp::AVERAGE_POOL_2D.0, 1),
        (BuiltinOp::RESHAPE.0, 1),
        (BuiltinOp::FULLY_CONNECTED.0, 1),
        (BuiltinOp::SOFTMAX.0, 1),
    ]);
    assert_eq!(histogram(&ops), expected);

    let (input, output) = io(&graph);
    assert_eq!(input.dims.as_slice(), [1, 96, 96, 3]);
    assert_eq!(output.dims.as_slice(), [1, 2]);

    let OpOptions::Conv2D(first) = ops[0].options else {
        panic!("{:?}", ops[0])
    };
    assert_eq!(first.activation, Activation::Relu);
    assert_eq!((first.stride_w, first.stride_h), (2, 2));
    check_per_channel_filters(&graph);
}

#[test]
fn ad01_int8() {
    let bytes = load("ad01_int8");
    let model = Model::parse(&bytes).unwrap();
    let graph = model.subgraph(0).unwrap();
    check_graph(&graph);
    let ops = operators(&graph);
    assert_eq!(
        histogram(&ops),
        BTreeMap::from([(BuiltinOp::FULLY_CONNECTED.0, 10)])
    );
    let (input, output) = io(&graph);
    for t in [input, output] {
        assert_eq!(t.dims.as_slice(), [1, 640]);
        assert_eq!(t.dtype, DataType::I8);
    }
    for op in &ops {
        let OpOptions::FullyConnected(o) = op.options else {
            panic!("{op:?}")
        };
        assert!(o.default_weights_format);
        assert!(!o.keep_num_dims);
        let weights = graph.tensor(op.inputs.get(1).unwrap() as usize).unwrap();
        assert_eq!(weights.dtype, DataType::I8);
        assert_eq!(weights.dims.rank(), 2);
        assert!(weights.data.is_some());
    }
}
