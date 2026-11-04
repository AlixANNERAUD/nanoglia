//! Bit-exact comparisons with outputs of the LiteRT reference kernels (see [`super::golden`]).

use super::golden::*;
use super::*;
use nanoglia_core::{
    Activation, Conv2DOptions, DepthwiseConv2DOptions, FullyConnectedOptions, Padding,
    Pool2DOptions,
};
use std::vec::Vec;

/// Runs a unary 8-bit operator on every input value.
fn sweep_i8(
    code: BuiltinOp,
    options: OpOptions<'static>,
    input: (f32, i64),
    output: (f32, i64),
) -> Vec<i8> {
    let x = q(t(0, DataType::I8, &[256]), input.0, input.1);
    let y = q(t(1, DataType::I8, &[256]), output.0, output.1);
    let op = op(code, options, &[Some(x)], &[y]);
    run1(&op, &[buf(&all_i8())])
}

#[test]
fn logistic_i8() {
    let out = |s, z| {
        sweep_i8(
            BuiltinOp::LOGISTIC,
            OpOptions::None,
            (s, z),
            (1.0 / 256.0, -128),
        )
    };
    assert_eq!(out(0.1, -5), LOGISTIC_I8_A);
    assert_eq!(out(0.0627, 3), LOGISTIC_I8_B);
}

#[test]
fn logistic_u8() {
    let x = q(t(0, DataType::U8, &[256]), 0.05, 120);
    let y = q(t(1, DataType::U8, &[256]), 1.0 / 256.0, 0);
    let op = op(BuiltinOp::LOGISTIC, OpOptions::None, &[Some(x)], &[y]);
    let input: Vec<u8> = (0..=255).collect();
    assert_eq!(run1::<u8>(&op, &[buf(&input)]), LOGISTIC_U8);
}

#[test]
fn tanh_i8() {
    let out = |s, z| sweep_i8(BuiltinOp::TANH, OpOptions::None, (s, z), (1.0 / 128.0, 0));
    assert_eq!(out(0.05, 2), TANH_I8_A);
    assert_eq!(out(0.25, -20), TANH_I8_B);
}

#[test]
fn hard_swish_i8() {
    let out = |i, o| sweep_i8(BuiltinOp::HARD_SWISH, OpOptions::None, i, o);
    assert_eq!(out((0.03, -10), (0.02, -100)), HARD_SWISH_I8_A);
    assert_eq!(out((0.2, 5), (0.1, -3)), HARD_SWISH_I8_B);
    assert_eq!(out((0.002, 0), (0.004, -128)), HARD_SWISH_I8_C);
}

#[test]
fn relu_family_i8() {
    let out = |code, i, o| sweep_i8(code, OpOptions::None, i, o);
    assert_eq!(out(BuiltinOp::RELU, (0.05, -3), (0.04, -128)), RELU_I8);
    assert_eq!(out(BuiltinOp::RELU6, (0.05, -3), (0.025, -128)), RELU6_I8);
    assert_eq!(
        out(BuiltinOp::RELU_N1_TO_1, (0.02, 0), (0.01, 0)),
        RELU_N1_TO_1_I8
    );
}

#[test]
fn leaky_relu_i8() {
    let out = |alpha| {
        sweep_i8(
            BuiltinOp::LEAKY_RELU,
            OpOptions::LeakyRelu { alpha },
            (0.05, -3),
            (0.04, 10),
        )
    };
    assert_eq!(out(0.2), LEAKY_RELU_I8_A);
    assert_eq!(out(-0.3), LEAKY_RELU_I8_B);
}

#[test]
fn quantize_from_f32() {
    let input = floats(&QUANTIZE_F32_IN);
    let x = t(0, DataType::F32, &[64]);
    let op8 = op(
        BuiltinOp::QUANTIZE,
        OpOptions::None,
        &[Some(x)],
        &[q(t(1, DataType::I8, &[64]), 0.1, 3)],
    );
    assert_eq!(run1::<i8>(&op8, &[buf(&input)]), QUANTIZE_F32_I8);
    let op16 = op(
        BuiltinOp::QUANTIZE,
        OpOptions::None,
        &[Some(x)],
        &[q(t(1, DataType::I16, &[64]), 0.001, 0)],
    );
    assert_eq!(run1::<i16>(&op16, &[buf(&input)]), QUANTIZE_F32_I16);
}

#[test]
fn requantize() {
    let all = all_i8();
    let x = q(t(0, DataType::I8, &[256]), 0.05, -3);
    let y = q(t(1, DataType::I8, &[256]), 0.07, 10);
    let op1 = op(BuiltinOp::QUANTIZE, OpOptions::None, &[Some(x)], &[y]);
    assert_eq!(run1::<i8>(&op1, &[buf(&all)]), REQUANTIZE_I8_I8);

    let input: Vec<u8> = (0..=255).collect();
    let x = q(t(0, DataType::U8, &[256]), 0.05, 128);
    let y = q(t(1, DataType::I8, &[256]), 0.05, 0);
    let op2 = op(BuiltinOp::QUANTIZE, OpOptions::None, &[Some(x)], &[y]);
    assert_eq!(run1::<i8>(&op2, &[buf(&input)]), REQUANTIZE_U8_I8);

    let x = q(t(0, DataType::I16, &[64]), 0.001, 0);
    let y = q(t(1, DataType::I8, &[64]), 0.05, -1);
    let op3 = op(BuiltinOp::QUANTIZE, OpOptions::None, &[Some(x)], &[y]);
    assert_eq!(
        run1::<i8>(&op3, &[buf(&REQUANTIZE_I16_IN)]),
        REQUANTIZE_I16_I8
    );
}

#[test]
fn dequantize() {
    let x = q(t(0, DataType::I8, &[256]), 0.0123, -7);
    let op1 = op(
        BuiltinOp::DEQUANTIZE,
        OpOptions::None,
        &[Some(x)],
        &[t(1, DataType::F32, &[256])],
    );
    assert_bits(
        &run1::<f32>(&op1, &[buf(&all_i8())]),
        &floats(&DEQUANTIZE_I8),
    );

    let x = q(t(0, DataType::I16, &[64]), 0.00031, 0);
    let op2 = op(
        BuiltinOp::DEQUANTIZE,
        OpOptions::None,
        &[Some(x)],
        &[t(1, DataType::F32, &[64])],
    );
    assert_bits(
        &run1::<f32>(&op2, &[buf(&REQUANTIZE_I16_IN)]),
        &floats(&DEQUANTIZE_I16),
    );
}

/// The shapes of a `MEAN`: input, constant axes, output and `keep_dims`.
struct MeanShape<'s>(&'s [i32], &'s [i32], &'s [i32], bool);

/// A `MEAN` on `dtype`, with the given quantization of the input and output (if any).
fn mean_op(
    dtype: DataType,
    MeanShape(input, axes, output, keep_dims): MeanShape<'_>,
    quant: Option<((f32, i64), (f32, i64))>,
) -> OpDesc<'static, 'static> {
    let (mut x, mut y) = (t(0, dtype, input), t(2, dtype, output));
    if let Some((qi, qo)) = quant {
        x = q(x, qi.0, qi.1);
        y = q(y, qo.0, qo.1);
    }
    let axis = konst(t(1, DataType::I32, &[axes.len() as i32]), axes);
    op(
        BuiltinOp::MEAN,
        OpOptions::Reducer { keep_dims },
        &[Some(x), Some(axis)],
        &[y],
    )
}

fn mean_i8(shape: MeanShape<'_>, qi: (f32, i64), qo: (f32, i64), data: &[i8]) -> Vec<i8> {
    run1(&mean_op(DataType::I8, shape, Some((qi, qo))), &[buf(data)])
}

#[test]
fn mean() {
    let shape = MeanShape(&[1, 4, 5, 3], &[1, 2], &[1, 1, 1, 3], true);
    let out = mean_i8(shape, (0.1, -5), (0.07, 2), &MEAN_I8_A_IN);
    assert_eq!(out, MEAN_I8_A_OUT);
    let shape = MeanShape(&[2, 3, 4], &[-1], &[2, 3], false);
    let out = mean_i8(shape, (0.05, 7), (0.05, 7), &MEAN_I8_B_IN);
    assert_eq!(out, MEAN_I8_B_OUT);
    let shape = MeanShape(&[2, 3, 4], &[0, 2], &[1, 3, 1], true);
    let out = mean_i8(shape, (0.02, -128), (0.3, 1), &MEAN_I8_C_IN);
    assert_eq!(out, MEAN_I8_C_OUT);

    let op = mean_op(
        DataType::F32,
        MeanShape(&[2, 3, 4], &[1], &[2, 4], false),
        None,
    );
    let out = run1::<f32>(&op, &[buf(&floats(&MEAN_F32_IN))]);
    assert_bits(&out, &floats(&MEAN_F32_OUT));
}

#[test]
fn conv_2d_f32() {
    let options = OpOptions::Conv2D(Conv2DOptions {
        padding: Padding::Same,
        stride_w: 1,
        stride_h: 2,
        dilation_w: 1,
        dilation_h: 1,
        activation: Activation::Relu6,
    });
    let w = konst(t(1, DataType::F32, &[4, 3, 3, 3]), &floats(&CONV_W));
    let b = konst(t(2, DataType::F32, &[4]), &floats(&CONV_B));
    let op = op(
        BuiltinOp::CONV_2D,
        options,
        &[Some(t(0, DataType::F32, &[1, 5, 6, 3])), Some(w), Some(b)],
        &[t(3, DataType::F32, &[1, 3, 6, 4])],
    );
    assert_bits(
        &run1::<f32>(&op, &[buf(&floats(&CONV_IN))]),
        &floats(&CONV_OUT),
    );
}

#[test]
fn depthwise_conv_2d_f32() {
    let options = OpOptions::DepthwiseConv2D(DepthwiseConv2DOptions {
        padding: Padding::Valid,
        stride_w: 1,
        stride_h: 1,
        depth_multiplier: 2,
        dilation_w: 2,
        dilation_h: 1,
        activation: Activation::None,
    });
    let w = konst(t(1, DataType::F32, &[1, 3, 3, 4]), &floats(&DW_W));
    let b = konst(t(2, DataType::F32, &[4]), &floats(&DW_B));
    let op = op(
        BuiltinOp::DEPTHWISE_CONV_2D,
        options,
        &[Some(t(0, DataType::F32, &[1, 6, 6, 2])), Some(w), Some(b)],
        &[t(3, DataType::F32, &[1, 4, 2, 4])],
    );
    assert_bits(&run1::<f32>(&op, &[buf(&floats(&DW_IN))]), &floats(&DW_OUT));
}

#[test]
fn fully_connected_f32() {
    let options = OpOptions::FullyConnected(FullyConnectedOptions {
        activation: Activation::None,
        default_weights_format: true,
        keep_num_dims: false,
        asymmetric_quantize_inputs: false,
    });
    let w = konst(t(1, DataType::F32, &[3, 10]), &floats(&FC_W));
    let b = konst(t(2, DataType::F32, &[3]), &floats(&FC_B));
    let op = op(
        BuiltinOp::FULLY_CONNECTED,
        options,
        &[Some(t(0, DataType::F32, &[2, 10])), Some(w), Some(b)],
        &[t(3, DataType::F32, &[2, 3])],
    );
    assert_bits(&run1::<f32>(&op, &[buf(&floats(&FC_IN))]), &floats(&FC_OUT));
}

/// A 16x8 `FULLY_CONNECTED` `[4, 24] x [6, 24]` with the weight scales `scales`.
pub(crate) fn fully_connected_16x8_op(
    scales: &[f32],
    activation: Activation,
    bias: Option<TensorDesc<'static>>,
) -> OpDesc<'static, 'static> {
    let options = OpOptions::FullyConnected(FullyConnectedOptions {
        activation,
        default_weights_format: true,
        keep_num_dims: false,
        asymmetric_quantize_inputs: false,
    });
    let zero_points = std::vec![0; scales.len()];
    let w = konst(
        qc(t(1, DataType::I8, &[6, 24]), scales, &zero_points, 0),
        &FC16_W,
    );
    op(
        BuiltinOp::FULLY_CONNECTED,
        options,
        &[
            Some(q(t(0, DataType::I16, &[4, 24]), 0.0002, 0)),
            Some(w),
            bias,
        ],
        &[q(t(3, DataType::I16, &[4, 6]), 0.0003, 0)],
    )
}

/// The 16x8 kernel requantizes like TensorFlow Lite Micro (`*_TFLM`: fixed point with a 16-bit
/// multiplier, computed with the integer formula), not like LiteRT's reference kernel (floating
/// point), which gives one more or one less on some outputs.
#[test]
fn fully_connected_16x8() {
    let bias = || Some(konst(t(2, DataType::I64, &[6]), &FC16_BIAS));
    let per_channel = [0.004, 0.006, 0.002, 0.01, 0.003, 0.008];
    for (scales, activation, tflm, litert) in [
        (
            &per_channel[..],
            Activation::Relu6,
            &FC16_PER_CHANNEL_TFLM,
            &FC16_PER_CHANNEL,
        ),
        (
            &[0.005],
            Activation::None,
            &FC16_PER_TENSOR_TFLM,
            &FC16_PER_TENSOR,
        ),
    ] {
        let op = fully_connected_16x8_op(scales, activation, bias());
        assert_eq!(select(&op).unwrap().name(), "rust/fully_connected_16x8");
        let out = run1::<i16>(&op, &[buf(&FC16_X)]);
        assert_eq!(&out, tflm);
        for (a, b) in out.iter().zip(litert) {
            assert!((i32::from(*a) - i32::from(*b)).abs() <= 1);
        }
    }
}

#[test]
fn pooling_f32() {
    let options = OpOptions::Pool2D(Pool2DOptions {
        padding: Padding::Same,
        stride_w: 2,
        stride_h: 2,
        filter_w: 3,
        filter_h: 3,
        activation: Activation::None,
    });
    for (code, expected) in [
        (BuiltinOp::AVERAGE_POOL_2D, &POOL_AVG_OUT),
        (BuiltinOp::MAX_POOL_2D, &POOL_MAX_OUT),
    ] {
        let op = op(
            code,
            options,
            &[Some(t(0, DataType::F32, &[1, 5, 5, 2]))],
            &[t(1, DataType::F32, &[1, 3, 3, 2])],
        );
        assert_bits(
            &run1::<f32>(&op, &[buf(&floats(&POOL_IN))]),
            &floats(expected),
        );
    }
}

#[test]
fn softmax_f32() {
    let op = op(
        BuiltinOp::SOFTMAX,
        OpOptions::Softmax { beta: 0.7 },
        &[Some(t(0, DataType::F32, &[2, 7]))],
        &[t(1, DataType::F32, &[2, 7])],
    );
    assert_bits(
        &run1::<f32>(&op, &[buf(&floats(&SOFTMAX_IN))]),
        &floats(&SOFTMAX_OUT),
    );
}

#[test]
fn softmax_i16_tables() {
    use crate::softmax::{LUT_SIZE, lut_populate_i16};
    let range = 65535.0;
    let mut lut = [0i16; LUT_SIZE];
    lut_populate_i16((10.0 / range, 32767), 2.0 / range, libm::exp, &mut lut);
    assert_eq!(lut, SOFTMAX_EXP_LUT);
    lut_populate_i16(
        (1.0 / range, -32768),
        2.0 / range,
        |x| 1.0 / (1.0 + x),
        &mut lut,
    );
    assert_eq!(lut, SOFTMAX_ONE_OVER_LUT);
}

#[test]
fn softmax_i16() {
    let rows16 = spread_i16();
    let (rows8, single, one_hot) = (
        &SOFTMAX_X_I16[..64],
        &SOFTMAX_X_I16[64..65],
        &SOFTMAX_X_I16[65..69],
    );
    for (shape, scale, beta, x, expected) in [
        (
            &[16, 16][..],
            0.000_121_986_9,
            1.0,
            &rows16[..],
            &SOFTMAX_I16_ROWS16[..],
        ),
        (&[8, 8], 0.000_7, 0.5, rows8, &SOFTMAX_I16_ROWS8),
        // A row of one element: the result saturates to 32767.
        (&[1, 1], 0.000_7, 1.0, single, &SOFTMAX_I16_SINGLE),
        (&[1, 4], 0.000_7, 1.0, one_hot, &SOFTMAX_I16_ONE_HOT),
    ] {
        let op = op(
            BuiltinOp::SOFTMAX,
            OpOptions::Softmax { beta },
            &[Some(q(t(0, DataType::I16, shape), scale, 0))],
            &[q(t(1, DataType::I16, shape), 1.0 / 32768.0, 0)],
        );
        assert_eq!(select(&op).unwrap().name(), "rust/softmax_i16");
        assert_eq!(run1::<i16>(&op, &[buf(x)]), expected, "{shape:?}");
    }
}

#[test]
fn binary_f32_broadcast() {
    for (code, activation, expected) in [
        (BuiltinOp::ADD, Activation::None, &BIN_ADD_OUT),
        (BuiltinOp::SUB, Activation::None, &BIN_SUB_OUT),
        (BuiltinOp::MUL, Activation::ReluN1To1, &BIN_MUL_OUT),
    ] {
        let op = op(
            code,
            OpOptions::Elementwise { activation },
            &[
                Some(t(0, DataType::F32, &[2, 3, 1])),
                Some(t(1, DataType::F32, &[1, 4])),
            ],
            &[t(2, DataType::F32, &[2, 3, 4])],
        );
        let out = run1::<f32>(&op, &[buf(&floats(&BIN_A)), buf(&floats(&BIN_B))]);
        assert_bits(&out, &floats(expected));
    }
}

#[test]
fn unary_f32() {
    let input = floats(&UNARY_IN);
    for (code, expected) in [
        (BuiltinOp::LOGISTIC, &LOGISTIC_F32_OUT),
        (BuiltinOp::TANH, &TANH_F32_OUT),
        (BuiltinOp::HARD_SWISH, &HARD_SWISH_F32_OUT),
    ] {
        let op = op(
            code,
            OpOptions::None,
            &[Some(t(0, DataType::F32, &[64]))],
            &[t(1, DataType::F32, &[64])],
        );
        assert_bits(&run1::<f32>(&op, &[buf(&input)]), &floats(expected));
    }
}

/// 256 int16 values spread over the whole range: `-32768 + 257 * i`.
fn spread_i16() -> Vec<i16> {
    (0..256).map(|i| (i * 257 - 32768) as i16).collect()
}

/// The shape and the quantization of a tensor.
type Quantized<'a> = (&'a [i32], (f32, i64));

/// Runs a quantized binary operator of type `dtype` on `a` and `b`, checking that the Rust kernel
/// runs it.
fn quantized_binary<T: Pod>(
    code: BuiltinOp,
    activation: Activation,
    dtype: DataType,
    [qa, qb, qo]: [Quantized<'_>; 3],
    a: &[T],
    b: &[T],
) -> Vec<T> {
    let op = op(
        code,
        OpOptions::Elementwise { activation },
        &[
            Some(q(t(0, dtype, qa.0), qa.1.0, qa.1.1)),
            Some(q(t(1, dtype, qb.0), qb.1.0, qb.1.1)),
        ],
        &[q(t(2, dtype, qo.0), qo.1.0, qo.1.1)],
    );
    assert_eq!(select(&op).unwrap().name(), "rust/binary_q");
    run1(&op, &[buf(a), buf(b)])
}

#[test]
fn quantized_binary_i8_broadcast() {
    let (qa, qb, qo) = (
        (0.015_610_263, -64),
        (0.012_902_341, 13),
        (0.025_976_034, -21),
    );
    let (a, b) = (all_i8(), BINARY_B_I8.to_vec());
    let (none, i8) = (Activation::None, DataType::I8);
    let (full, row): (&[i32], &[i32]) = (&[16, 16], &[16]);
    let add = quantized_binary(
        BuiltinOp::ADD,
        none,
        i8,
        [(full, qa), (row, qb), (full, qo)],
        &a,
        &b,
    );
    assert_eq!(add, ADD_I8);
    let relu6 = (full, (0.03, -128));
    let add = quantized_binary(
        BuiltinOp::ADD,
        Activation::Relu6,
        i8,
        [(full, qa), (row, qb), relu6],
        &a,
        &b,
    );
    assert_eq!(add, ADD_RELU6_I8);
    // [16, 1, 16] - [16] -> [16, 1, 16].
    let cube: &[i32] = &[16, 1, 16];
    let sub = quantized_binary(
        BuiltinOp::SUB,
        none,
        i8,
        [(cube, qa), (row, qb), (cube, qo)],
        &a,
        &b,
    );
    assert_eq!(sub, SUB_I8);
    // The first input broadcasts.
    let sub = quantized_binary(
        BuiltinOp::SUB,
        none,
        i8,
        [(row, qb), (full, qa), (full, qo)],
        &b,
        &a,
    );
    assert_eq!(sub, SUB_FIRST_BROADCAST_I8);
    let column: &[i32] = &[1, 16];
    for (act, expected) in [(none, &MUL_I8), (Activation::Relu, &MUL_RELU_I8)] {
        let shapes = [(full, qa), (column, qb), (full, (0.02, 5))];
        assert_eq!(
            &quantized_binary(BuiltinOp::MUL, act, i8, shapes, &a, &b),
            expected
        );
    }
}

#[test]
fn quantized_binary_i16_broadcast() {
    let (qa, qb) = ((9.112_61e-5, 0), (5.554_919e-5, 0));
    let (a, b) = (spread_i16(), BINARY_B_I16.to_vec());
    let (full, row): (&[i32], &[i32]) = (&[16, 16], &[16]);
    for (code, qo, expected) in [
        (BuiltinOp::ADD, 0.000_117_512_53, &ADD_I16),
        (BuiltinOp::SUB, 0.000_15, &SUB_I16),
        (BuiltinOp::MUL, 0.000_2, &MUL_I16),
    ] {
        let shapes = [(full, qa), (row, qb), (full, (qo, 0))];
        let out = quantized_binary(code, Activation::None, DataType::I16, shapes, &a, &b);
        assert_eq!(&out, expected, "{code:?}");
    }
}
