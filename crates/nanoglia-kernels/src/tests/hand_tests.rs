//! Hand-computed cases, vectors from the TFLite Micro kernel tests (`*_test.cc`) and checks of
//! what the backend refuses.

use super::*;
use nanoglia_core::{
    Activation, Conv2DOptions, DepthwiseConv2DOptions, FullyConnectedOptions, Padding,
    Pool2DOptions,
};

fn unary(
    code: BuiltinOp,
    x: TensorDesc<'static>,
    y: TensorDesc<'static>,
) -> OpDesc<'static, 'static> {
    op(code, OpOptions::None, &[Some(x)], &[y])
}

#[test]
fn backend_identity() {
    assert_eq!(BACKEND.name(), "rust");
    assert_eq!(BACKEND.kind(), nanoglia_core::backend::BackendKind::Cpu);
}

#[test]
fn reshape_squeeze_expand_dims_copy_bytes() {
    let data: Vec<i16> = (0..12).collect();
    let shape = konst(t(1, DataType::I32, &[2]), &[3i32, 4]);
    let reshape = op(
        BuiltinOp::RESHAPE,
        OpOptions::None,
        &[Some(t(0, DataType::I16, &[2, 6])), Some(shape)],
        &[t(2, DataType::I16, &[3, 4])],
    );
    assert_eq!(run1::<i16>(&reshape, &[buf(&data)]), data);

    let squeeze = op(
        BuiltinOp::SQUEEZE,
        OpOptions::Squeeze {
            dims: nanoglia_core::le::LeSlice::EMPTY,
        },
        &[Some(t(0, DataType::I16, &[1, 12, 1]))],
        &[t(1, DataType::I16, &[12])],
    );
    assert_eq!(run1::<i16>(&squeeze, &[buf(&data)]), data);

    let axis = konst(t(1, DataType::I32, &[]), &[0i32]);
    let expand = op(
        BuiltinOp::EXPAND_DIMS,
        OpOptions::None,
        &[Some(t(0, DataType::Bool, &[4])), Some(axis)],
        &[t(2, DataType::Bool, &[1, 4])],
    );
    assert_eq!(run1::<u8>(&expand, &[buf(&[1u8, 0, 1, 1])]), [1, 0, 1, 1]);

    // Element counts or types that differ are refused.
    let bad = unary(
        BuiltinOp::RESHAPE,
        t(0, DataType::I8, &[4]),
        t(1, DataType::I8, &[5]),
    );
    assert!(select(&bad).is_none());
    let bad = unary(
        BuiltinOp::RESHAPE,
        t(0, DataType::I8, &[4]),
        t(1, DataType::U8, &[4]),
    );
    assert!(select(&bad).is_none());
}

#[test]
fn quantize_hand_computed() {
    // `round(x / 0.5) + 10`, rounding half away from zero, clamped to `u8`.
    let x = t(0, DataType::F32, &[6]);
    let y = q(t(1, DataType::U8, &[6]), 0.5, 10);
    let op1 = unary(BuiltinOp::QUANTIZE, x, y);
    let input = [0.0f32, 0.25, -0.25, 1.0, -100.0, 1000.0];
    assert_eq!(run1::<u8>(&op1, &[buf(&input)]), [10, 11, 9, 12, 0, 255]);

    // Per-channel outputs are not supported.
    let y = qc(t(1, DataType::I8, &[2, 3]), &[0.5, 0.25], &[0, 0], 0);
    assert!(select(&unary(BuiltinOp::QUANTIZE, t(0, DataType::F32, &[2, 3]), y)).is_none());
    // Nor are int16 -> int16 conversions with zero points.
    let x = q(t(0, DataType::I16, &[2]), 0.5, 3);
    let y = q(t(1, DataType::I16, &[2]), 0.5, 0);
    assert!(select(&unary(BuiltinOp::QUANTIZE, x, y)).is_none());
}

#[test]
fn dequantize_per_channel_and_f16() {
    // Channels along axis 1: scale * (value - zero_point), computed in `f32`.
    let x = qc(
        t(0, DataType::I8, &[2, 3]),
        &[0.5, 0.25, 2.0],
        &[1, -2, 0],
        1,
    );
    let op1 = unary(BuiltinOp::DEQUANTIZE, x, t(1, DataType::F32, &[2, 3]));
    let out = run1::<f32>(&op1, &[buf(&[3i8, 2, -1, -1, 0, 4])]);
    assert_eq!(out, [1.0, 1.0, -2.0, -1.0, 0.5, 8.0]);

    let halves = [0x3c00u16, 0xc000, 0x3555, 0x0001, 0x7bff, 0xfc00, 0x0000];
    let op2 = unary(
        BuiltinOp::DEQUANTIZE,
        t(0, DataType::F16, &[7]),
        t(1, DataType::F32, &[7]),
    );
    let out = run1::<f32>(&op2, &[buf(&halves)]);
    assert_eq!(out[..3], [1.0, -2.0, 0.333_251_95]);
    assert_eq!(out[3], 2f32.powi(-24));
    assert_eq!(out[4], 65504.0);
    assert_eq!(out[5], f32::NEG_INFINITY);
    assert_eq!(out[6].to_bits(), 0);
}

#[test]
fn relu_float() {
    let input = [-3.0f32, -0.5, 0.0, 0.5, 3.0, 7.5];
    let cases: [(BuiltinOp, [f32; 6]); 3] = [
        (BuiltinOp::RELU, [0.0, 0.0, 0.0, 0.5, 3.0, 7.5]),
        (BuiltinOp::RELU6, [0.0, 0.0, 0.0, 0.5, 3.0, 6.0]),
        (BuiltinOp::RELU_N1_TO_1, [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0]),
    ];
    for (code, expected) in cases {
        let op1 = unary(code, t(0, DataType::F32, &[6]), t(1, DataType::F32, &[6]));
        assert_eq!(run1::<f32>(&op1, &[buf(&input)]), expected);
    }
    let leaky = op(
        BuiltinOp::LEAKY_RELU,
        OpOptions::LeakyRelu { alpha: 0.5 },
        &[Some(t(0, DataType::F32, &[6]))],
        &[t(1, DataType::F32, &[6])],
    );
    assert_eq!(
        run1::<f32>(&leaky, &[buf(&input)]),
        [-1.5, -0.25, 0.0, 0.5, 3.0, 7.5]
    );
}

#[test]
fn relu_int16() {
    // Symmetric int16 with rescaling by 2.
    let x = q(t(0, DataType::I16, &[4]), 0.5, 0);
    let y = q(t(1, DataType::I16, &[4]), 0.25, 0);
    let op1 = unary(BuiltinOp::RELU6, x, y);
    assert_eq!(
        run1::<i16>(&op1, &[buf(&[-4i16, 3, 11, 100])]),
        [0, 6, 22, 24]
    );
}

#[test]
fn logistic_requires_output_scale() {
    let x = q(t(0, DataType::I8, &[4]), 0.1, 0);
    let y = q(t(1, DataType::I8, &[4]), 1.0 / 255.0, -128);
    assert!(select(&unary(BuiltinOp::LOGISTIC, x, y)).is_none());
}

#[test]
fn mean_tflm_vectors() {
    // reduce_test.cc: MeanFloat4DKeepDims and the int8 variant (scale 1, zero point 0).
    let input: Vec<f32> = (1..=24).map(|v| v as f32).collect();
    let axis = konst(t(1, DataType::I32, &[2]), &[1i32, 2]);
    let op1 = op(
        BuiltinOp::MEAN,
        OpOptions::Reducer { keep_dims: true },
        &[Some(t(0, DataType::F32, &[2, 2, 3, 2])), Some(axis)],
        &[t(2, DataType::F32, &[2, 1, 1, 2])],
    );
    assert_eq!(run1::<f32>(&op1, &[buf(&input)]), [6.0, 7.0, 18.0, 19.0]);

    let input_i8: Vec<i8> = (1..=24).collect();
    let op2 = op(
        BuiltinOp::MEAN,
        OpOptions::Reducer { keep_dims: true },
        &[
            Some(q(t(0, DataType::I8, &[2, 2, 3, 2]), 1.0, 0)),
            Some(axis),
        ],
        &[q(t(2, DataType::I8, &[2, 1, 1, 2]), 1.0, 0)],
    );
    assert_eq!(run1::<i8>(&op2, &[buf(&input_i8)]), [6, 7, 18, 19]);

    // Duplicate and negative axes resolve like TFLite.
    let axis = konst(t(1, DataType::I32, &[3]), &[0i32, -4, 3]);
    let op3 = op(
        BuiltinOp::MEAN,
        OpOptions::Reducer { keep_dims: false },
        &[Some(t(0, DataType::F32, &[2, 2, 3, 2])), Some(axis)],
        &[t(2, DataType::F32, &[2, 3])],
    );
    assert_eq!(
        run1::<f32>(&op3, &[buf(&input)]),
        [7.5, 9.5, 11.5, 13.5, 15.5, 17.5]
    );

    // A non-constant axis is refused.
    let op4 = op(
        BuiltinOp::MEAN,
        OpOptions::Reducer { keep_dims: false },
        &[
            Some(t(0, DataType::F32, &[4])),
            Some(t(1, DataType::I32, &[1])),
        ],
        &[t(2, DataType::F32, &[])],
    );
    assert!(select(&op4).is_none());
}

/// A PAD of the 1x2x2x1 tensor of `pad_test.cc` to 3x2x4x1.
fn pad_op(
    dtype: DataType,
    value: Option<TensorDesc<'static>>,
    quant: Option<(f32, i64)>,
) -> OpDesc<'static, 'static> {
    let paddings = konst(t(1, DataType::I32, &[4, 2]), &[1i32, 1, 0, 0, 1, 1, 0, 0]);
    let mut x = t(0, dtype, &[1, 2, 2, 1]);
    let mut y = t(3, dtype, &[3, 2, 4, 1]);
    if let Some((s, z)) = quant {
        x = q(x, s, z);
        y = q(y, s, z);
    }
    let code = if value.is_some() {
        BuiltinOp::PADV2
    } else {
        BuiltinOp::PAD
    };
    let mut inputs = vec![Some(x), Some(paddings)];
    if value.is_some() {
        inputs.push(value);
    }
    op(code, OpOptions::None, &inputs, &[y])
}

#[test]
fn pad_tflm_vectors() {
    let golden = [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 0., 0., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.,
    ];
    let out = run1::<f32>(
        &pad_op(DataType::F32, None, None),
        &[buf(&[1.0f32, 2., 3., 4.])],
    );
    assert_eq!(out, golden);

    let value = konst(t(2, DataType::F32, &[]), &[42.0f32]);
    let out = run1::<f32>(
        &pad_op(DataType::F32, Some(value), None),
        &[buf(&[1.0f32, 2., 3., 4.])],
    );
    let expected: Vec<f32> = golden
        .iter()
        .map(|&g| if g == 0.0 { 42.0 } else { g })
        .collect();
    assert_eq!(out, expected);

    // int8 PAD fills with the output zero point.
    let out = run1::<i8>(
        &pad_op(DataType::I8, None, Some((0.5, -7))),
        &[buf(&[1i8, 2, 3, 4])],
    );
    let expected: Vec<i8> = golden
        .iter()
        .map(|&g| if g == 0.0 { -7 } else { g as i8 })
        .collect();
    assert_eq!(out, expected);

    // PADV2 with a pad value computed at run time.
    let value = q(t(2, DataType::I8, &[1]), 0.5, -7);
    let out = run1::<i8>(
        &pad_op(DataType::I8, Some(value), Some((0.5, -7))),
        &[buf(&[1i8, 2, 3, 4]), buf(&[42i8])],
    );
    let expected: Vec<i8> = golden
        .iter()
        .map(|&g| if g == 0.0 { 42 } else { g as i8 })
        .collect();
    assert_eq!(out, expected);

    // The pad value must share the output's quantization (pad_test.cc ExpectFailure cases).
    let value = konst(q(t(2, DataType::I8, &[1]), 0.25, -7), &[42i8]);
    assert!(select(&pad_op(DataType::I8, Some(value), Some((0.5, -7)))).is_none());

    // pad_test.cc Test4DFloat: 42 padded by one on every side, with int64 paddings.
    let paddings = konst(t(1, DataType::I64, &[4, 2]), &[1i64; 8]);
    let op4 = op(
        BuiltinOp::PAD,
        OpOptions::None,
        &[Some(t(0, DataType::F32, &[1, 1, 1, 1])), Some(paddings)],
        &[t(2, DataType::F32, &[3, 3, 3, 3])],
    );
    let out = run1::<f32>(&op4, &[buf(&[42.0f32])]);
    assert_eq!(out.iter().filter(|&&v| v == 0.0).count(), 80);
    assert_eq!(out[40], 42.0);
}

#[test]
fn concatenation() {
    // concatenation_test.cc TwoInputsAllAxesCombinations, axes 1, -1 and 0.
    let a = [1.0f32, 2., 3., 4., 5., 6.];
    let b = [7.0f32, 8., 9., 10., 11., 12.];
    for axis in [1, -1] {
        let op1 = op(
            BuiltinOp::CONCATENATION,
            OpOptions::Concatenation {
                axis,
                activation: Activation::None,
            },
            &[
                Some(t(0, DataType::F32, &[2, 3])),
                Some(t(1, DataType::F32, &[2, 3])),
            ],
            &[t(2, DataType::F32, &[2, 6])],
        );
        let out = run1::<f32>(&op1, &[buf(&a), buf(&b)]);
        assert_eq!(out, [1., 2., 3., 7., 8., 9., 4., 5., 6., 10., 11., 12.]);
    }
    let op0 = op(
        BuiltinOp::CONCATENATION,
        OpOptions::Concatenation {
            axis: 0,
            activation: Activation::None,
        },
        &[
            Some(t(0, DataType::F32, &[2, 3])),
            Some(t(1, DataType::F32, &[2, 3])),
        ],
        &[t(2, DataType::F32, &[4, 3])],
    );
    let out = run1::<f32>(&op0, &[buf(&a), buf(&b)]);
    assert_eq!(out, [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);

    // int8 with a constant input and identical quantization.
    let c = konst(q(t(1, DataType::I8, &[1, 1, 2]), 0.1, 3), &[9i8, 8]);
    let op8 = op(
        BuiltinOp::CONCATENATION,
        OpOptions::Concatenation {
            axis: 1,
            activation: Activation::None,
        },
        &[Some(q(t(0, DataType::I8, &[1, 2, 2]), 0.1, 3)), Some(c)],
        &[q(t(2, DataType::I8, &[1, 3, 2]), 0.1, 3)],
    );
    assert_eq!(
        run1::<i8>(&op8, &[buf(&[1i8, 2, 3, 4])]),
        [1, 2, 3, 4, 9, 8]
    );

    // Different quantization or a fused activation are refused.
    let op_bad = op(
        BuiltinOp::CONCATENATION,
        OpOptions::Concatenation {
            axis: 1,
            activation: Activation::None,
        },
        &[
            Some(q(t(0, DataType::I8, &[1, 2]), 0.1, 3)),
            Some(q(t(1, DataType::I8, &[1, 2]), 0.2, 3)),
        ],
        &[q(t(2, DataType::I8, &[1, 4]), 0.1, 3)],
    );
    assert!(select(&op_bad).is_none());
    let op_bad = op(
        BuiltinOp::CONCATENATION,
        OpOptions::Concatenation {
            axis: 1,
            activation: Activation::Relu,
        },
        &[
            Some(t(0, DataType::F32, &[1, 2])),
            Some(t(1, DataType::F32, &[1, 2])),
        ],
        &[t(2, DataType::F32, &[1, 4])],
    );
    assert!(select(&op_bad).is_none());
}

#[test]
fn maximum_minimum_broadcast() {
    // maximum_minimum_test.cc FloatWithBroadcastTest and Int32WithBroadcastTest.
    let a = [1.0f32, 0.0, -1.0, -2.0, -1.44, 11.0];
    for (code, expected) in [
        (BuiltinOp::MAXIMUM, [1.0f32, 0.5, 0.5, 0.5, 0.5, 11.0]),
        (BuiltinOp::MINIMUM, [0.5f32, 0.0, -1.0, -2.0, -1.44, 0.5]),
    ] {
        let op1 = op(
            code,
            OpOptions::None,
            &[
                Some(t(0, DataType::F32, &[3, 1, 2])),
                Some(konst(t(1, DataType::F32, &[1]), &[0.5f32])),
            ],
            &[t(2, DataType::F32, &[3, 1, 2])],
        );
        assert_eq!(run1::<f32>(&op1, &[buf(&a)]), expected);
    }
    let a = [1i32, 0, -1, -2, 3, 11];
    let op2 = op(
        BuiltinOp::MAXIMUM,
        OpOptions::None,
        &[
            Some(t(0, DataType::I32, &[3, 1, 2])),
            Some(t(1, DataType::I32, &[1])),
        ],
        &[t(2, DataType::I32, &[3, 1, 2])],
    );
    assert_eq!(
        run1::<i32>(&op2, &[buf(&a), buf(&[2i32])]),
        [2, 2, 2, 2, 3, 11]
    );

    // int8 compares raw values; broadcasting on both sides.
    let op3 = op(
        BuiltinOp::MINIMUM,
        OpOptions::None,
        &[
            Some(t(0, DataType::I8, &[2, 1])),
            Some(t(1, DataType::I8, &[1, 3])),
        ],
        &[t(2, DataType::I8, &[2, 3])],
    );
    let out = run1::<i8>(&op3, &[buf(&[0i8, -5]), buf(&[-1i8, 0, 1])]);
    assert_eq!(out, [-1, 0, 0, -5, -5, -5]);
}

/// A quantized `code` on `[2] x [2] -> [2]` tensors of type `dtype`, with the given (scale,
/// zero point) of both inputs and of the output.
fn quantized_binary_op(
    code: BuiltinOp,
    activation: Activation,
    dtype: DataType,
    quant: [(f32, i64); 3],
) -> OpDesc<'static, 'static> {
    op(
        code,
        OpOptions::Elementwise { activation },
        &[
            Some(q(t(0, dtype, &[2]), quant[0].0, quant[0].1)),
            Some(q(t(1, dtype, &[2]), quant[1].0, quant[1].1)),
        ],
        &[q(t(2, dtype, &[2]), quant[2].0, quant[2].1)],
    )
}

#[test]
fn quantized_binary_hand_computed() {
    let none = Activation::None;
    // (0.5 * (x + 1)) + (0.25 * y) in steps of 0.5, zero point 3.
    let add = quantized_binary_op(
        BuiltinOp::ADD,
        none,
        DataType::I8,
        [(0.5, -1), (0.25, 0), (0.5, 3)],
    );
    let out = run1::<i8>(&add, &[buf(&[1i8, 10]), buf(&[4i8, -8])]);
    assert_eq!(out, [3 + 2 + 2, 3 + 11 - 4]);
    let sub = quantized_binary_op(
        BuiltinOp::SUB,
        none,
        DataType::I8,
        [(0.5, -1), (0.25, 0), (0.5, 3)],
    );
    let out = run1::<i8>(&sub, &[buf(&[1i8, 127]), buf(&[4i8, -8])]);
    assert_eq!(out, [3 + 2 - 2, 127]); // saturates
    let mul = quantized_binary_op(
        BuiltinOp::MUL,
        Activation::Relu,
        DataType::I16,
        [(0.5, 0), (0.25, 0), (0.125, 0)],
    );
    let out = run1::<i16>(&mul, &[buf(&[3i16, -3]), buf(&[5i16, 5])]);
    assert_eq!(out, [15, 0]);
}

#[test]
fn quantized_binary_refusals() {
    let none = Activation::None;
    let ok = [(0.3, 0), (0.2, 0), (0.7, 0)];
    for code in [BuiltinOp::ADD, BuiltinOp::SUB, BuiltinOp::MUL] {
        for dtype in [DataType::I8, DataType::I16] {
            assert!(select(&quantized_binary_op(code, none, dtype, ok)).is_some());
            // Fused activations TFLite does not fuse.
            let tanh = quantized_binary_op(code, Activation::Tanh, dtype, ok);
            assert!(select(&tanh).is_none());
        }
        // int16 zero points must be 0, int8 ones in range.
        let zp = quantized_binary_op(code, none, DataType::I16, [(0.3, 1), (0.2, 0), (0.7, 0)]);
        assert!(select(&zp).is_none());
        let zp = quantized_binary_op(code, none, DataType::I8, [(0.3, 0), (0.2, 128), (0.7, 0)]);
        assert!(select(&zp).is_none());
        // Mixed types.
        let mut mixed = quantized_binary_op(code, none, DataType::I8, ok);
        mixed.inputs = Box::leak(
            vec![mixed.inputs[0], Some(q(t(1, DataType::I16, &[2]), 0.2, 0))].into_boxed_slice(),
        );
        assert!(select(&mixed).is_none());
    }
    // int16 ADD and SUB with power-of-two scales take TFLite's `pot_scale_int16` kernel.
    let pot = [(0.5, 0), (0.25, 0), (1.0 / 1024.0, 0)];
    assert!(
        select(&quantized_binary_op(
            BuiltinOp::ADD,
            none,
            DataType::I16,
            pot
        ))
        .is_none()
    );
    assert!(
        select(&quantized_binary_op(
            BuiltinOp::SUB,
            none,
            DataType::I16,
            pot
        ))
        .is_none()
    );
    assert!(
        select(&quantized_binary_op(
            BuiltinOp::MUL,
            none,
            DataType::I16,
            pot
        ))
        .is_some()
    );
    assert!(
        select(&quantized_binary_op(
            BuiltinOp::ADD,
            none,
            DataType::I8,
            pot
        ))
        .is_some()
    );
    // An output scale too small for the 20 bits of headroom of `ADD`.
    let narrow = [(1.0, 0), (1.0, 0), (1e-8, 0)];
    assert!(
        select(&quantized_binary_op(
            BuiltinOp::ADD,
            none,
            DataType::I8,
            narrow
        ))
        .is_none()
    );
    let sub = select(&quantized_binary_op(
        BuiltinOp::SUB,
        none,
        DataType::I8,
        narrow,
    ));
    assert_eq!(sub.is_some(), nanoglia_core::quant::SINGLE_ROUNDING);
    // A product scale too large to requantize.
    let huge = [(1e20, 0), (1e20, 0), (1e-20, 0)];
    assert!(
        select(&quantized_binary_op(
            BuiltinOp::MUL,
            none,
            DataType::I8,
            huge
        ))
        .is_none()
    );
}

#[test]
fn binary_shapes_and_types() {
    // A 6D broadcast: [2,1,1,1,1,3] + [1,1,1,1,2,1] -> [2,1,1,1,2,3].
    let op1 = op(
        BuiltinOp::ADD,
        OpOptions::Elementwise {
            activation: Activation::None,
        },
        &[
            Some(t(0, DataType::F32, &[2, 1, 1, 1, 1, 3])),
            Some(t(1, DataType::F32, &[1, 1, 1, 1, 2, 1])),
        ],
        &[t(2, DataType::F32, &[2, 1, 1, 1, 2, 3])],
    );
    let out = run1::<f32>(
        &op1,
        &[buf(&[1.0f32, 2., 3., 4., 5., 6.]), buf(&[10.0f32, 20.])],
    );
    assert_eq!(
        out,
        [11., 12., 13., 21., 22., 23., 14., 15., 16., 24., 25., 26.]
    );

    // A scalar on the left, with a fused ReLU.
    let op2 = op(
        BuiltinOp::SUB,
        OpOptions::Elementwise {
            activation: Activation::Relu,
        },
        &[
            Some(t(0, DataType::F32, &[])),
            Some(t(1, DataType::F32, &[4])),
        ],
        &[t(2, DataType::F32, &[4])],
    );
    let out = run1::<f32>(&op2, &[buf(&[1.0f32]), buf(&[-1.0f32, 0.5, 1.0, 3.0])]);
    assert_eq!(out, [2.0, 0.5, 0.0, 0.0]);

    // Incompatible shapes, uint8 ADD and a fused tanh are refused.
    let bad = op(
        BuiltinOp::MUL,
        OpOptions::Elementwise {
            activation: Activation::None,
        },
        &[
            Some(t(0, DataType::F32, &[2, 3])),
            Some(t(1, DataType::F32, &[2])),
        ],
        &[t(2, DataType::F32, &[2, 3])],
    );
    assert!(select(&bad).is_none());
    let bad = op(
        BuiltinOp::ADD,
        OpOptions::Elementwise {
            activation: Activation::None,
        },
        &[
            Some(q(t(0, DataType::U8, &[2]), 0.1, 0)),
            Some(q(t(1, DataType::U8, &[2]), 0.1, 0)),
        ],
        &[q(t(2, DataType::U8, &[2]), 0.1, 0)],
    );
    assert!(select(&bad).is_none());
    let bad = op(
        BuiltinOp::ADD,
        OpOptions::Elementwise {
            activation: Activation::Tanh,
        },
        &[
            Some(t(0, DataType::F32, &[2])),
            Some(t(1, DataType::F32, &[2])),
        ],
        &[t(2, DataType::F32, &[2])],
    );
    assert!(select(&bad).is_none());
}

fn conv_options(padding: Padding, stride: i32) -> OpOptions<'static> {
    OpOptions::Conv2D(Conv2DOptions {
        padding,
        stride_w: stride,
        stride_h: stride,
        dilation_w: 1,
        dilation_h: 1,
        activation: Activation::None,
    })
}

#[test]
fn conv_tflm_vectors() {
    // conv_test.cc SimpleTestFloat.
    let filter = [1.0f32, 2., 3., 4., -1., 1., -1., 1., -1., -1., 1., 1.];
    let w = konst(t(1, DataType::F32, &[3, 2, 2, 1]), &filter);
    let b = konst(t(2, DataType::F32, &[3]), &[1.0f32, 2., 3.]);
    let op1 = op(
        BuiltinOp::CONV_2D,
        conv_options(Padding::Valid, 2),
        &[Some(t(0, DataType::F32, &[2, 2, 4, 1])), Some(w), Some(b)],
        &[t(3, DataType::F32, &[2, 1, 2, 3])],
    );
    let input = [
        1.0f32, 1., 1., 1., 2., 2., 2., 2., 1., 2., 3., 4., 1., 2., 3., 4.,
    ];
    let out = run1::<f32>(&op1, &[buf(&input)]);
    assert_eq!(out, [18., 2., 5., 18., 2., 5., 17., 4., 3., 37., 4., 3.]);

    // A wrong output size and int8 tensors (CMSIS-NN's) are refused.
    let w = konst(t(1, DataType::F32, &[3, 2, 2, 1]), &[0.0f32; 12]);
    let bad = op(
        BuiltinOp::CONV_2D,
        conv_options(Padding::Valid, 2),
        &[Some(t(0, DataType::F32, &[2, 2, 4, 1])), Some(w)],
        &[t(3, DataType::F32, &[2, 1, 3, 3])],
    );
    assert!(select(&bad).is_none());
    let w = konst(q(t(1, DataType::I8, &[3, 2, 2, 1]), 0.1, 0), &[0i8; 12]);
    let bad = op(
        BuiltinOp::CONV_2D,
        conv_options(Padding::Valid, 2),
        &[Some(q(t(0, DataType::I8, &[2, 2, 4, 1]), 0.1, 0)), Some(w)],
        &[q(t(3, DataType::I8, &[2, 1, 2, 3]), 0.1, 0)],
    );
    assert!(select(&bad).is_none());
}

#[test]
fn grouped_conv() {
    // Two groups of one input channel, 1x1 filters: each output channel scales its group.
    let w = konst(t(1, DataType::F32, &[4, 1, 1, 1]), &[1.0f32, 2., 3., 4.]);
    let op1 = op(
        BuiltinOp::CONV_2D,
        conv_options(Padding::Same, 1),
        &[Some(t(0, DataType::F32, &[1, 1, 2, 2])), Some(w), None],
        &[t(3, DataType::F32, &[1, 1, 2, 4])],
    );
    let out = run1::<f32>(&op1, &[buf(&[1.0f32, 10., 2., 20.])]);
    assert_eq!(out, [1., 2., 30., 40., 2., 4., 60., 80.]);
}

#[test]
fn depthwise_tflm_vectors() {
    // depthwise_conv_test.cc SimpleTest.
    let filter = [
        1.0f32, 2., 3., 4., -9., 10., -11., 12., 5., 6., 7., 8., 13., -14., 15., -16.,
    ];
    let w = konst(t(1, DataType::F32, &[1, 2, 2, 4]), &filter);
    let b = konst(t(2, DataType::F32, &[1, 1, 1, 4]), &[1.0f32, 2., 3., 4.]);
    let options = OpOptions::DepthwiseConv2D(DepthwiseConv2DOptions {
        padding: Padding::Valid,
        stride_w: 1,
        stride_h: 1,
        depth_multiplier: 2,
        dilation_w: 1,
        dilation_h: 1,
        activation: Activation::None,
    });
    let op1 = op(
        BuiltinOp::DEPTHWISE_CONV_2D,
        options,
        &[Some(t(0, DataType::F32, &[1, 3, 2, 2])), Some(w), Some(b)],
        &[t(3, DataType::F32, &[1, 2, 1, 4])],
    );
    let input = [1.0f32, 2., 7., 8., 3., 4., 9., 10., 5., 6., 11., 12.];
    let out = run1::<f32>(&op1, &[buf(&input)]);
    assert_eq!(out, [71., -34., 99., -20., 91., -26., 127., -4.]);
}

fn fc_options(activation: Activation) -> OpOptions<'static> {
    OpOptions::FullyConnected(FullyConnectedOptions {
        activation,
        default_weights_format: true,
        keep_num_dims: false,
        asymmetric_quantize_inputs: false,
    })
}

#[test]
fn fully_connected_tflm_vectors() {
    // fully_connected_test.cc SimpleTest (with a fused ReLU).
    let weights: Vec<f32> = (0..3).flat_map(|_| (1..=10).map(|v| v as f32)).collect();
    let w = konst(t(1, DataType::F32, &[3, 10]), &weights);
    let b = konst(t(2, DataType::F32, &[3]), &[1.0f32, 2., 3.]);
    let op1 = op(
        BuiltinOp::FULLY_CONNECTED,
        fc_options(Activation::Relu),
        &[Some(t(0, DataType::F32, &[2, 10])), Some(w), Some(b)],
        &[t(3, DataType::F32, &[2, 3])],
    );
    let input = [
        1.0f32, 2., 3., 4., 5., 6., 7., 8., -9., -10., 1., 2., 3., 4., 5., 6., 7., -8., 9., -10.,
    ];
    assert_eq!(
        run1::<f32>(&op1, &[buf(&input)]),
        [24., 25., 26., 58., 59., 60.]
    );

    // Hybrid layers (int8 weights, f32 input) are refused.
    let w = konst(q(t(1, DataType::I8, &[3, 10]), 0.1, 0), &[0i8; 30]);
    let bad = op(
        BuiltinOp::FULLY_CONNECTED,
        fc_options(Activation::None),
        &[Some(t(0, DataType::F32, &[2, 10])), Some(w), None],
        &[t(3, DataType::F32, &[2, 3])],
    );
    assert!(select(&bad).is_none());
}

#[test]
fn pooling_tflm_vectors() {
    // pooling_test.cc SimpleAveragePoolTestFloat and SimpleMaxPoolTestFloat.
    let input = [0.0f32, 6., 2., 4., 3., 2., 10., 7.];
    for (code, expected) in [
        (BuiltinOp::AVERAGE_POOL_2D, [2.75f32, 5.75]),
        (BuiltinOp::MAX_POOL_2D, [6.0f32, 10.0]),
    ] {
        let options = OpOptions::Pool2D(Pool2DOptions {
            padding: Padding::Valid,
            stride_w: 2,
            stride_h: 2,
            filter_w: 2,
            filter_h: 2,
            activation: Activation::None,
        });
        let op1 = op(
            code,
            options,
            &[Some(t(0, DataType::F32, &[1, 2, 4, 1]))],
            &[t(1, DataType::F32, &[1, 1, 2, 1])],
        );
        assert_eq!(run1::<f32>(&op1, &[buf(&input)]), expected);
    }
}

#[test]
fn softmax_reference_values() {
    // TFLite softmax_test.cc: softmax of [1, 2, 3, 4, 5] and of its opposite.
    let op1 = op(
        BuiltinOp::SOFTMAX,
        OpOptions::Softmax { beta: 1.0 },
        &[Some(t(0, DataType::F32, &[2, 5]))],
        &[t(1, DataType::F32, &[2, 5])],
    );
    let input = [1.0f32, 2., 3., 4., 5., -1., -2., -3., -4., -5.];
    let out = run1::<f32>(&op1, &[buf(&input)]);
    let expected = [
        0.011_656_231,
        0.031_684_92,
        0.086_128_54,
        0.234_121_66,
        0.636_408_6,
        0.636_408_6,
        0.234_121_66,
        0.086_128_54,
        0.031_684_92,
        0.011_656_231,
    ];
    assert_near(&out, &expected, 1e-6);
}

#[test]
fn fully_connected_16x8_conditions() {
    use super::golden_tests::fully_connected_16x8_op;
    let none = Activation::None;
    // Without bias: the dot products of hand-made data.
    let mut op1 = fully_connected_16x8_op(&[0.005], none, None);
    assert!(select(&op1).is_some());
    // An int32 bias, which LiteRT reads according to an undecoded option.
    let int32_bias = konst(t(2, DataType::I32, &[6]), &[0i32; 6]);
    assert!(select(&fully_connected_16x8_op(&[0.005], none, Some(int32_bias))).is_none());
    // Per-channel scales along the wrong axis, weights with a zero point.
    let mut weights = op1.inputs[1].unwrap();
    weights = qc(weights, &[0.005; 24], &[0; 24], 1);
    let mut inputs = op1.inputs.to_vec();
    inputs[1] = Some(weights);
    let wrong_axis = OpDesc {
        inputs: Box::leak(inputs.into_boxed_slice()),
        ..op1
    };
    assert!(select(&wrong_axis).is_none());
    let mut inputs = op1.inputs.to_vec();
    inputs[1] = Some(qc(op1.inputs[1].unwrap(), &[0.005], &[3], 0));
    let shifted = OpDesc {
        inputs: Box::leak(inputs.into_boxed_slice()),
        ..op1
    };
    assert!(select(&shifted).is_none());
    // An input zero point.
    let mut inputs = op1.inputs.to_vec();
    inputs[0] = Some(q(t(0, DataType::I16, &[4, 24]), 0.0002, 5));
    op1.inputs = Box::leak(inputs.into_boxed_slice());
    assert!(select(&op1).is_none());
}

#[test]
fn softmax_i16_conditions() {
    let softmax = |input: TensorDesc<'static>, output: TensorDesc<'static>| {
        op(
            BuiltinOp::SOFTMAX,
            OpOptions::Softmax { beta: 1.0 },
            &[Some(input)],
            &[output],
        )
    };
    let x = |zero_point| q(t(0, DataType::I16, &[2, 5]), 0.001, zero_point);
    let y = |scale, zero_point| q(t(1, DataType::I16, &[2, 5]), scale, zero_point);
    let one = 1.0 / 32768.0;
    // Equal inputs share the probability evenly: 32768 / 5, rounded by the tables.
    let out = run1::<i16>(&softmax(x(0), y(one, 0)), &[buf(&[7i16; 10])]);
    assert!(out.iter().all(|&v| (6552..=6555).contains(&v)), "{out:?}");
    // The output must be quantized as [0, 1) in Q0.15, the input symmetric.
    assert!(select(&softmax(x(0), y(one * 1.002, 0))).is_none());
    assert!(select(&softmax(x(0), y(one, 1))).is_none());
    assert!(select(&softmax(x(3), y(one, 0))).is_none());
    let int8_output = q(t(1, DataType::I8, &[2, 5]), 1.0 / 256.0, -128);
    assert!(select(&softmax(x(0), int8_output)).is_none());
}

#[test]
fn eval_rejects_short_buffers() {
    let op1 = unary(
        BuiltinOp::RELU,
        t(0, DataType::F32, &[4]),
        t(1, DataType::F32, &[4]),
    );
    let err = run(&op1, &[&[0u8; 8]]).unwrap_err();
    assert_eq!(err.code, nanoglia_core::ErrorCode::InvalidArgument);
}

#[test]
fn misaligned_constants_are_reported() {
    let mut w = t(1, DataType::F32, &[1, 4]);
    let bytes = buf(&[0u8; 20]);
    w.data = Some(&bytes[1..17]);
    let op1 = op(
        BuiltinOp::FULLY_CONNECTED,
        fc_options(Activation::None),
        &[Some(t(0, DataType::F32, &[1, 4])), Some(w)],
        &[t(2, DataType::F32, &[1, 1])],
    );
    let err = (crate::fully_connected::FULLY_CONNECTED_F32.plan)(&op1).unwrap_err();
    assert_eq!(err.detail, Detail::Misaligned { tensor: 1 });
}

#[test]
fn unknown_operators_are_not_claimed() {
    let x = t(0, DataType::F32, &[4]);
    assert!(select(&unary(BuiltinOp::DEPTH_TO_SPACE, x, x)).is_none());
    let custom = OpDesc {
        index: 0,
        code: OpCode::Custom("Foo"),
        options: OpOptions::None,
        inputs: &[],
        outputs: &[],
    };
    assert!(select(&custom).is_none());
    // Softmax on int8 belongs to CMSIS-NN.
    let x = q(t(0, DataType::I8, &[4]), 0.1, 0);
    let y = q(t(1, DataType::I8, &[4]), 1.0 / 256.0, -128);
    let softmax = op(
        BuiltinOp::SOFTMAX,
        OpOptions::Softmax { beta: 1.0 },
        &[Some(x)],
        &[y],
    );
    assert!(select(&softmax).is_none());
}
