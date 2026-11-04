//! Field-level tests on models built by hand.

mod common;

use common::{Field, Value, finish, id, ints, s, table, tables};
use nanoglia_core::{
    Activation, BuiltinOp, Conv2DOptions, DataType, Detail, ErrorCode, OpCode, OpOptions, Padding,
};
use nanoglia_tflite::Model;

type Fields = Vec<(u16, Field)>;

/// A small valid model: one `CONV_2D` reading an activation and a constant filter.
struct Spec {
    version: u32,
    opcodes: Vec<Fields>,
    tensors: Vec<Fields>,
    operators: Vec<Fields>,
    inputs: Vec<i32>,
    outputs: Vec<i32>,
    buffers: Vec<Fields>,
    /// Bytes appended after the flatbuffer.
    tail: Vec<u8>,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            version: 3,
            opcodes: vec![vec![
                (id::opcode::DEPRECATED_BUILTIN_CODE, Field::I8(3)),
                (id::opcode::VERSION, Field::I32(5)),
                (id::opcode::BUILTIN_CODE, Field::I32(3)),
            ]],
            tensors: vec![
                vec![
                    (id::tensor::SHAPE, ints(&[1, 4, 4, 1])),
                    (id::tensor::TYPE, Field::I8(9)),
                    (id::tensor::NAME, s("in")),
                    (
                        id::tensor::QUANTIZATION,
                        table(vec![
                            (id::quant::SCALE, Field::Ref(Value::Floats(vec![0.5]))),
                            (id::quant::ZERO_POINT, Field::Ref(Value::Longs(vec![-1]))),
                        ]),
                    ),
                ],
                vec![
                    (id::tensor::SHAPE, ints(&[2, 1, 1, 1])),
                    (id::tensor::TYPE, Field::I8(9)),
                    (id::tensor::BUFFER, Field::U32(1)),
                    (id::tensor::NAME, s("filter")),
                    (
                        id::tensor::QUANTIZATION,
                        table(vec![
                            (
                                id::quant::SCALE,
                                Field::Ref(Value::Floats(vec![0.25, 0.125])),
                            ),
                            (id::quant::ZERO_POINT, Field::Ref(Value::Longs(vec![0, 0]))),
                            (id::quant::QUANTIZED_DIMENSION, Field::I32(0)),
                        ]),
                    ),
                ],
                vec![
                    (id::tensor::SHAPE, ints(&[1, 4, 4, 2])),
                    (id::tensor::TYPE, Field::I8(9)),
                    (id::tensor::NAME, s("out")),
                    (id::tensor::SHAPE_SIGNATURE, ints(&[-1, 4, 4, 2])),
                ],
            ],
            operators: vec![vec![
                (id::operator::OPCODE_INDEX, Field::U32(0)),
                (id::operator::INPUTS, ints(&[0, 1, -1])),
                (id::operator::OUTPUTS, ints(&[2])),
            ]],
            inputs: vec![0],
            outputs: vec![2],
            buffers: vec![
                vec![],
                vec![(id::buffer::DATA, Field::Ref(Value::Bytes(vec![7, 9])))],
            ],
            tail: Vec::new(),
        }
    }
}

impl Spec {
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = finish(&Value::Table(vec![
            (id::model::VERSION, Field::U32(self.version)),
            (id::model::OPERATOR_CODES, tables(self.opcodes.clone())),
            (
                id::model::SUBGRAPHS,
                tables(vec![vec![
                    (id::subgraph::TENSORS, tables(self.tensors.clone())),
                    (id::subgraph::INPUTS, ints(&self.inputs)),
                    (id::subgraph::OUTPUTS, ints(&self.outputs)),
                    (id::subgraph::OPERATORS, tables(self.operators.clone())),
                    (id::subgraph::NAME, s("main")),
                ]]),
            ),
            (id::model::DESCRIPTION, s("synthetic")),
            (id::model::BUFFERS, tables(self.buffers.clone())),
        ]));
        bytes.extend_from_slice(&self.tail);
        bytes
    }

    fn with_options(mut self, ty: u8, options: Fields) -> Self {
        self.operators[0].push((id::operator::BUILTIN_OPTIONS_TYPE, Field::U8(ty)));
        self.operators[0].push((id::operator::BUILTIN_OPTIONS, table(options)));
        self
    }

    fn with_opcode(mut self, code: i32) -> Self {
        self.opcodes[0] = vec![
            (
                id::opcode::DEPRECATED_BUILTIN_CODE,
                Field::I8(code.min(127) as i8),
            ),
            (id::opcode::BUILTIN_CODE, Field::I32(code)),
        ];
        self
    }
}

fn malformed(bytes: &[u8]) {
    let err = Model::parse(bytes).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidEncoding, "{err}");
    assert!(matches!(err.detail, Detail::MalformedModel { .. }), "{err}");
}

#[test]
fn reads_a_small_model() {
    let bytes = Spec::default().bytes();
    let model = Model::parse(&bytes).unwrap();
    assert_eq!(model.version(), 3);
    assert_eq!(model.description(), "synthetic");
    assert_eq!(model.subgraph_count(), 1);
    let graph = model.subgraph(0).unwrap();
    assert_eq!(graph.name(), "main");
    assert_eq!(graph.tensor_count(), 3);
    assert_eq!(graph.inputs().iter().collect::<Vec<_>>(), [0]);
    assert_eq!(graph.outputs().iter().collect::<Vec<_>>(), [2]);

    let input = graph.tensor(0).unwrap();
    assert_eq!(input.index, 0);
    assert_eq!(input.name, "in");
    assert_eq!(input.dtype, DataType::I8);
    assert_eq!(input.dims.as_slice(), [1, 4, 4, 1]);
    assert_eq!(input.quant.scale(), 0.5);
    assert_eq!(input.quant.zero_point(), -1);
    assert!(input.data.is_none());

    let filter = graph.tensor(1).unwrap();
    assert_eq!(filter.index, 1);
    assert_eq!(filter.data, Some(&[7u8, 9][..]));
    assert_eq!(filter.quant.scale.iter().collect::<Vec<_>>(), [0.25, 0.125]);

    // `shape_signature` is ignored.
    assert_eq!(graph.tensor(2).unwrap().dims.as_slice(), [1, 4, 4, 2]);
    assert!(!graph.tensor_is_variable(2).unwrap());

    let op = graph.operator(0).unwrap();
    assert_eq!(op.code, OpCode::Builtin(BuiltinOp::CONV_2D));
    assert_eq!(op.version, 5);
    assert_eq!(op.inputs.iter().collect::<Vec<_>>(), [0, 1, -1]);
    assert_eq!(op.outputs.iter().collect::<Vec<_>>(), [2]);
    // No options table: schema defaults.
    assert_eq!(
        op.options,
        OpOptions::Conv2D(Conv2DOptions {
            padding: Padding::Same,
            stride_w: 0,
            stride_h: 0,
            dilation_w: 1,
            dilation_h: 1,
            activation: Activation::None,
        })
    );

    for (result, index) in [
        (graph.tensor(3).map(|_| ()), 3),
        (graph.tensor_is_variable(3).map(|_| ()), 3),
        (graph.operator(1).map(|_| ()), 1),
        (model.subgraph(1).map(|_| ()), 1),
    ] {
        let err = result.unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidArgument);
        assert_eq!(err.detail, Detail::IndexOutOfRange { index });
    }
}

#[test]
fn checks_identifier_and_version() {
    let mut bytes = Spec::default().bytes();
    bytes[4] = b'X';
    assert!(!Model::is_tflite(&bytes));
    let err = Model::parse(&bytes).unwrap_err();
    assert_eq!(err.detail, Detail::UnknownEncoding);

    let bytes = Spec {
        version: 2,
        ..Spec::default()
    }
    .bytes();
    let err = Model::parse(&bytes).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedOperation);
    assert_eq!(err.detail, Detail::UnsupportedSchema { version: 2 });
}

#[test]
fn checks_indices() {
    let spec = Spec {
        inputs: vec![3],
        ..Spec::default()
    };
    malformed(&spec.bytes());

    let spec = Spec {
        outputs: vec![-1],
        ..Spec::default()
    };
    malformed(&spec.bytes());

    let mut spec = Spec::default();
    spec.operators[0][1] = (id::operator::INPUTS, ints(&[0, 3]));
    malformed(&spec.bytes());

    let mut spec = Spec::default();
    spec.operators[0][1] = (id::operator::INPUTS, ints(&[0, -2]));
    malformed(&spec.bytes());

    let mut spec = Spec::default();
    spec.operators[0][2] = (id::operator::OUTPUTS, ints(&[-1]));
    malformed(&spec.bytes());

    let mut spec = Spec::default();
    spec.operators[0][0] = (id::operator::OPCODE_INDEX, Field::U32(1));
    malformed(&spec.bytes());

    let mut spec = Spec::default();
    spec.tensors[1][2] = (id::tensor::BUFFER, Field::U32(2));
    malformed(&spec.bytes());
}

#[test]
fn buffers() {
    // Data stored after the flatbuffer, at an offset from the start of the file.
    let mut spec = Spec::default();
    spec.buffers[1] = vec![
        (id::buffer::OFFSET, Field::U64(0)),
        (id::buffer::SIZE, Field::U64(3)),
    ];
    let len = spec.bytes().len() as u64;
    spec.buffers[1][0] = (id::buffer::OFFSET, Field::U64(len + 1));
    spec.tail = vec![1, 2, 3, 4];
    let bytes = spec.bytes();
    let model = Model::parse(&bytes).unwrap();
    let filter = model.subgraph(0).unwrap().tensor(1).unwrap();
    assert_eq!(filter.data, Some(&[2u8, 3, 4][..]));

    // One byte past the end.
    spec.buffers[1][1] = (id::buffer::SIZE, Field::U64(4));
    malformed(&spec.bytes());
    spec.buffers[1][1] = (id::buffer::SIZE, Field::U64(u64::MAX));
    malformed(&spec.bytes());

    // Offsets 0 and 1 mean "use `data`".
    let mut spec = Spec::default();
    spec.buffers[1].push((id::buffer::OFFSET, Field::U64(1)));
    let bytes = spec.bytes();
    let model = Model::parse(&bytes).unwrap();
    let filter = model.subgraph(0).unwrap().tensor(1).unwrap();
    assert_eq!(filter.data, Some(&[7u8, 9][..]));

    // Empty data.
    let mut spec = Spec::default();
    spec.buffers[1] = vec![(id::buffer::DATA, Field::Ref(Value::Bytes(vec![])))];
    let bytes = spec.bytes();
    let model = Model::parse(&bytes).unwrap();
    assert!(model.subgraph(0).unwrap().tensor(1).unwrap().data.is_none());

    // Buffer 0 is the empty sentinel, even if it has data.
    let mut spec = Spec::default();
    spec.buffers[0] = vec![(id::buffer::DATA, Field::Ref(Value::Bytes(vec![1])))];
    let bytes = spec.bytes();
    let model = Model::parse(&bytes).unwrap();
    assert!(model.subgraph(0).unwrap().tensor(0).unwrap().data.is_none());
}

#[test]
fn operator_codes() {
    // `builtin_code` takes over above 127 (`PLACEHOLDER_FOR_GREATER_OP_CODES`).
    let bytes = Spec::default().with_opcode(150).bytes();
    let model = Model::parse(&bytes).unwrap();
    let op = model.subgraph(0).unwrap().operator(0).unwrap();
    assert_eq!(op.code, OpCode::Builtin(BuiltinOp(150)));
    assert_eq!(op.version, 1);
    assert_eq!(op.options, OpOptions::None);

    // Old models only have `deprecated_builtin_code`.
    let mut spec = Spec::default();
    spec.opcodes[0] = vec![(id::opcode::DEPRECATED_BUILTIN_CODE, Field::I8(25))];
    let bytes = spec.bytes();
    let model = Model::parse(&bytes).unwrap();
    let op = model.subgraph(0).unwrap().operator(0).unwrap();
    assert_eq!(op.code, OpCode::Builtin(BuiltinOp::SOFTMAX));
    assert_eq!(op.options, OpOptions::Softmax { beta: 0.0 });
}

#[test]
fn custom_operators() {
    let mut spec = Spec::default();
    spec.opcodes[0] = vec![
        (id::opcode::DEPRECATED_BUILTIN_CODE, Field::I8(32)),
        (id::opcode::CUSTOM_CODE, s("MyOp")),
        (id::opcode::BUILTIN_CODE, Field::I32(32)),
    ];
    spec.operators[0].push((
        id::operator::CUSTOM_OPTIONS,
        Field::Ref(Value::Bytes(vec![1, 2, 3])),
    ));
    let bytes = spec.bytes();
    let model = Model::parse(&bytes).unwrap();
    let op = model.subgraph(0).unwrap().operator(0).unwrap();
    assert_eq!(op.code, OpCode::Custom("MyOp"));
    assert_eq!(op.options, OpOptions::Custom(&[1, 2, 3]));

    // Large custom options, stored after the flatbuffer.
    spec.operators[0].push((id::operator::LARGE_CUSTOM_OPTIONS_OFFSET, Field::U64(0)));
    spec.operators[0].push((id::operator::LARGE_CUSTOM_OPTIONS_SIZE, Field::U64(2)));
    let len = spec.bytes().len() as u64;
    spec.operators[0][4] = (id::operator::LARGE_CUSTOM_OPTIONS_OFFSET, Field::U64(len));
    spec.tail = vec![5, 6];
    let bytes = spec.bytes();
    assert_eq!(bytes.len() as u64, len + 2);
    let model = Model::parse(&bytes).unwrap();
    let op = model.subgraph(0).unwrap().operator(0).unwrap();
    assert_eq!(op.options, OpOptions::Custom(&[5, 6]));
    spec.tail = vec![5];
    malformed(&spec.bytes());

    // A custom operator needs a name.
    let mut spec = Spec::default();
    spec.opcodes[0] = vec![(id::opcode::BUILTIN_CODE, Field::I32(32))];
    malformed(&spec.bytes());
}

#[test]
fn decodes_options() {
    let decode = |spec: Spec| {
        let bytes = spec.bytes();
        let model = Model::parse(&bytes).unwrap();
        let options = model.subgraph(0).unwrap().operator(0).unwrap().options;
        format!("{options:?}")
    };

    let conv = Spec::default().with_options(
        1,
        vec![
            (0, Field::I8(1)),
            (1, Field::I32(2)),
            (2, Field::I32(3)),
            (3, Field::I8(3)),
            (4, Field::I32(4)),
            (5, Field::I32(5)),
        ],
    );
    let bytes = conv.bytes();
    let model = Model::parse(&bytes).unwrap();
    assert_eq!(
        model.subgraph(0).unwrap().operator(0).unwrap().options,
        OpOptions::Conv2D(Conv2DOptions {
            padding: Padding::Valid,
            stride_w: 2,
            stride_h: 3,
            dilation_w: 4,
            dilation_h: 5,
            activation: Activation::Relu6,
        })
    );

    let dw = Spec::default().with_opcode(4).with_options(
        2,
        vec![(3, Field::I32(2)), (4, Field::I8(1)), (6, Field::I32(3))],
    );
    assert_eq!(
        decode(dw),
        "DepthwiseConv2D(DepthwiseConv2DOptions { padding: Same, stride_w: 0, stride_h: 0, \
         depth_multiplier: 2, dilation_w: 1, dilation_h: 3, activation: Relu })"
    );

    let fc = Spec::default()
        .with_opcode(9)
        .with_options(8, vec![(1, Field::I8(1)), (2, Field::U8(1))]);
    assert_eq!(
        decode(fc),
        "FullyConnected(FullyConnectedOptions { activation: None, default_weights_format: false, \
         keep_num_dims: true, asymmetric_quantize_inputs: false })"
    );
    // Absent table: `weights_format` defaults to `DEFAULT`.
    assert_eq!(
        decode(Spec::default().with_opcode(9)),
        "FullyConnected(FullyConnectedOptions { activation: None, default_weights_format: true, \
         keep_num_dims: false, asymmetric_quantize_inputs: false })"
    );

    let pool = Spec::default().with_opcode(17).with_options(
        5,
        vec![(3, Field::I32(2)), (4, Field::I32(3)), (5, Field::I8(2))],
    );
    assert_eq!(
        decode(pool),
        "Pool2D(Pool2DOptions { padding: Same, stride_w: 0, stride_h: 0, filter_w: 2, \
         filter_h: 3, activation: ReluN1To1 })"
    );

    for (code, ty) in [(0, 11), (41, 28), (18, 21), (42, 29)] {
        let spec = Spec::default()
            .with_opcode(code)
            .with_options(ty, vec![(0, Field::I8(1))]);
        assert_eq!(decode(spec), "Elementwise { activation: Relu }");
    }

    let concat = Spec::default()
        .with_opcode(2)
        .with_options(10, vec![(0, Field::I32(-1))]);
    assert_eq!(
        decode(concat),
        "Concatenation { axis: -1, activation: None }"
    );

    for code in [40, 74, 82, 89, 81] {
        let spec = Spec::default()
            .with_opcode(code)
            .with_options(27, vec![(0, Field::U8(1))]);
        assert_eq!(decode(spec), "Reducer { keep_dims: true }");
    }

    let leaky = Spec::default()
        .with_opcode(98)
        .with_options(75, vec![(0, Field::F32(0.25))]);
    assert_eq!(decode(leaky), "LeakyRelu { alpha: 0.25 }");

    let squeeze = Spec::default()
        .with_opcode(43)
        .with_options(30, vec![(0, ints(&[1, -1]))]);
    assert_eq!(decode(squeeze), "Squeeze { dims: [1, -1] }");
    assert_eq!(
        decode(Spec::default().with_opcode(43)),
        "Squeeze { dims: [] }"
    );

    let softmax = Spec::default()
        .with_opcode(25)
        .with_options(9, vec![(0, Field::F32(2.0))]);
    assert_eq!(decode(softmax), "Softmax { beta: 2.0 }");

    let strided_slice = Spec::default().with_opcode(45).with_options(
        32,
        vec![
            (0, Field::I32(1)),
            (1, Field::I32(2)),
            (2, Field::I32(4)),
            (3, Field::I32(8)),
            (4, Field::I32(16)),
            (5, Field::U8(1)),
        ],
    );
    assert_eq!(
        decode(strided_slice),
        "StridedSlice(StridedSliceOptions { begin_mask: 1, end_mask: 2, ellipsis_mask: 4, \
         new_axis_mask: 8, shrink_axis_mask: 16, offset: true })"
    );
    assert_eq!(
        decode(Spec::default().with_opcode(45)),
        "StridedSlice(StridedSliceOptions { begin_mask: 0, end_mask: 0, ellipsis_mask: 0, \
         new_axis_mask: 0, shrink_axis_mask: 0, offset: false })"
    );

    let gather = Spec::default()
        .with_opcode(36)
        .with_options(23, vec![(0, Field::I32(-1)), (1, Field::I32(1))]);
    assert_eq!(decode(gather), "Gather { axis: -1, batch_dims: 1 }");
    assert_eq!(
        decode(Spec::default().with_opcode(36)),
        "Gather { axis: 0, batch_dims: 0 }"
    );

    let split = Spec::default()
        .with_opcode(49)
        .with_options(35, vec![(0, Field::I32(3))]);
    assert_eq!(decode(split), "Split { num_splits: 3 }");
    let split_v = Spec::default()
        .with_opcode(102)
        .with_options(79, vec![(0, Field::I32(2))]);
    assert_eq!(decode(split_v), "Split { num_splits: 2 }");
    // SPLIT carrying `SplitVOptions` is malformed.
    malformed(
        &Spec::default()
            .with_opcode(49)
            .with_options(79, vec![])
            .bytes(),
    );

    let arg_max = Spec::default()
        .with_opcode(56)
        .with_options(40, vec![(0, Field::I8(4))]);
    assert_eq!(decode(arg_max), "ArgMinMax { output_type: I64 }");
    let arg_min = Spec::default()
        .with_opcode(79)
        .with_options(57, vec![(0, Field::I8(2))]);
    assert_eq!(decode(arg_min), "ArgMinMax { output_type: I32 }");
    // Absent table: the schema default, FLOAT32 (refused by the kernels, as in TFLite).
    assert_eq!(
        decode(Spec::default().with_opcode(56)),
        "ArgMinMax { output_type: F32 }"
    );
    // ARG_MAX carrying `ArgMinOptions` is malformed.
    malformed(
        &Spec::default()
            .with_opcode(56)
            .with_options(57, vec![])
            .bytes(),
    );

    // The deprecated `new_height` and `new_width` (fields 0 and 1) are ignored.
    let bilinear = Spec::default().with_opcode(23).with_options(
        15,
        vec![
            (0, Field::I32(9)),
            (1, Field::I32(9)),
            (2, Field::U8(1)),
            (3, Field::U8(0)),
        ],
    );
    assert_eq!(
        decode(bilinear),
        "Resize { align_corners: true, half_pixel_centers: false }"
    );
    let nearest = Spec::default()
        .with_opcode(97)
        .with_options(74, vec![(1, Field::U8(1))]);
    assert_eq!(
        decode(nearest),
        "Resize { align_corners: false, half_pixel_centers: true }"
    );
    assert_eq!(
        decode(Spec::default().with_opcode(97)),
        "Resize { align_corners: false, half_pixel_centers: false }"
    );
    // RESIZE_BILINEAR carrying `ResizeNearestNeighborOptions` is malformed.
    malformed(
        &Spec::default()
            .with_opcode(23)
            .with_options(74, vec![])
            .bytes(),
    );

    let transpose_conv = Spec::default().with_opcode(67).with_options(
        49,
        vec![
            (0, Field::I8(1)),
            (1, Field::I32(2)),
            (2, Field::I32(3)),
            (3, Field::I8(3)),
            (4, Field::I8(2)),
        ],
    );
    assert_eq!(
        decode(transpose_conv),
        "TransposeConv(TransposeConvOptions { padding: Valid, stride_w: 2, stride_h: 3, \
         activation: Relu6 })"
    );
    // Versions 1 to 3 have no fused activation.
    let transpose_conv = Spec::default()
        .with_opcode(67)
        .with_options(49, vec![(1, Field::I32(1)), (2, Field::I32(1))]);
    assert_eq!(
        decode(transpose_conv),
        "TransposeConv(TransposeConvOptions { padding: Same, stride_w: 1, stride_h: 1, \
         activation: None })"
    );

    // Operators whose options nanoglia does not decode.
    let reshape = Spec::default()
        .with_opcode(22)
        .with_options(17, vec![(0, ints(&[1, 32]))]);
    assert_eq!(decode(reshape), "None");
}

#[test]
fn rejects_options_of_another_operator() {
    // CONV_2D carrying `Pool2DOptions`.
    let spec = Spec::default().with_options(5, vec![]);
    malformed(&spec.bytes());
    // NONE is fine.
    let spec = Spec::default().with_options(0, vec![]);
    assert!(Model::parse(&spec.bytes()).is_ok());
}

#[test]
fn unknown_enum_values() {
    for options in [
        vec![(0, Field::I8(2))],  // padding
        vec![(3, Field::I8(6))],  // activation
        vec![(3, Field::I8(-1))], // activation
    ] {
        let bytes = Spec::default().with_options(1, options).bytes();
        let model = Model::parse(&bytes).unwrap();
        let err = model.subgraph(0).unwrap().operator(0).unwrap_err();
        assert_eq!(err.code, ErrorCode::UnsupportedOperation);
        assert_eq!(
            err.detail,
            Detail::UnsupportedOp {
                op_index: 0,
                opcode: 3
            }
        );
    }
    // Unknown fully-connected weights format, non-standard quantization contract.
    for options in [
        vec![(1, Field::I8(2))],
        vec![(5, Field::Ref(Value::Bytes(vec![0])))],
    ] {
        let bytes = Spec::default()
            .with_opcode(9)
            .with_options(8, options)
            .bytes();
        let model = Model::parse(&bytes).unwrap();
        let err = model.subgraph(0).unwrap().operator(0).unwrap_err();
        assert_eq!(
            err.detail,
            Detail::UnsupportedOp {
                op_index: 0,
                opcode: 9
            }
        );
    }
    // Unknown ARG_MAX output type, TRANSPOSE_CONV padding and activation.
    for (opcode, ty, options) in [
        (56, 40, vec![(0, Field::I8(23))]),
        (67, 49, vec![(0, Field::I8(2))]),
        (67, 49, vec![(3, Field::I8(7))]),
    ] {
        let bytes = Spec::default()
            .with_opcode(opcode)
            .with_options(ty, options)
            .bytes();
        let model = Model::parse(&bytes).unwrap();
        let err = model.subgraph(0).unwrap().operator(0).unwrap_err();
        assert_eq!(
            err.detail,
            Detail::UnsupportedOp {
                op_index: 0,
                opcode
            }
        );
    }
}

#[test]
fn options_fields_must_fit_their_table() {
    // A 1-byte field where the schema has a 4-byte one, at the end of the table: reading it would
    // run past the table, so the verifier refuses the model.
    let spec = Spec::default()
        .with_opcode(67)
        .with_options(49, vec![(0, Field::I8(0)), (2, Field::U8(1))]);
    malformed(&spec.bytes());
    // The same table with a 4-byte stride is fine.
    let spec = Spec::default()
        .with_opcode(67)
        .with_options(49, vec![(0, Field::I8(0)), (2, Field::I32(1))]);
    assert!(Model::parse(&spec.bytes()).is_ok());
}

#[test]
fn unsupported_tensors() {
    let unsupported = |spec: Spec| {
        let bytes = spec.bytes();
        let model = Model::parse(&bytes).unwrap();
        let err = model.subgraph(0).unwrap().tensor(2).unwrap_err();
        assert_eq!(err.code, ErrorCode::UnsupportedOperation);
        assert_eq!(err.detail, Detail::UnsupportedType { tensor: 2 });
    };
    for ty in [5, 8, 11, 13, 14, 19, 23, -1] {
        let mut spec = Spec::default();
        spec.tensors[2][1] = (id::tensor::TYPE, Field::I8(ty));
        unsupported(spec);
    }

    let mut spec = Spec::default();
    spec.tensors[2][0] = (id::tensor::SHAPE, ints(&[1; 7]));
    unsupported(spec);

    let mut spec = Spec::default();
    spec.tensors[2].push((id::tensor::SPARSITY, table(vec![])));
    unsupported(spec);

    let mut spec = Spec::default();
    spec.tensors[2].push((
        id::tensor::QUANTIZATION,
        table(vec![
            (id::quant::DETAILS_TYPE, Field::U8(1)),
            (id::quant::DETAILS, table(vec![])),
        ]),
    ));
    unsupported(spec);

    // Rank 6 is fine.
    let mut spec = Spec::default();
    spec.tensors[2][0] = (id::tensor::SHAPE, ints(&[1; 6]));
    let bytes = spec.bytes();
    let model = Model::parse(&bytes).unwrap();
    assert_eq!(model.subgraph(0).unwrap().tensor(2).unwrap().dims.rank(), 6);
}

#[test]
fn variable_tensors() {
    let mut spec = Spec::default();
    spec.tensors[2].push((id::tensor::IS_VARIABLE, Field::U8(1)));
    let bytes = spec.bytes();
    let model = Model::parse(&bytes).unwrap();
    assert!(model.subgraph(0).unwrap().tensor_is_variable(2).unwrap());
}

#[test]
fn strings_must_be_utf8_and_terminated() {
    let mut spec = Spec::default();
    spec.tensors[0][2] = (
        id::tensor::NAME,
        Field::Ref(Value::Str(vec![b'a', 0xff, b'b'])),
    );
    malformed(&spec.bytes());

    let bytes = Spec::default().bytes();
    // The description is "synthetic": drop its NUL terminator.
    let at = bytes.windows(10).position(|w| w == b"synthetic\0").unwrap();
    let mut bytes = bytes;
    bytes[at + 9] = b'!';
    malformed(&bytes);
}

#[test]
fn truncated_vtable_fields_are_rejected() {
    // A table whose vtable claims a field beyond its inline size.
    let bytes = Spec::default().bytes();
    // The root vtable starts right after the 8-byte header: patch the inline size.
    let mut bad = bytes.clone();
    bad[10..12].copy_from_slice(&5u16.to_le_bytes());
    malformed(&bad);
    // Odd vtable size.
    let mut bad = bytes;
    bad[8..10].copy_from_slice(&7u16.to_le_bytes());
    malformed(&bad);
}
