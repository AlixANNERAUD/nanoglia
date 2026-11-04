//! `TRANSPOSE_CONV`: a vector from the TFLite kernel tests (`transpose_conv_test.cc`), the int8
//! kernel against the float one, and what the backend refuses.

use super::*;
use nanoglia_core::{Activation, Padding, TransposeConvOptions};

fn options(padding: Padding, stride: i32, activation: Activation) -> OpOptions<'static> {
    OpOptions::TransposeConv(TransposeConvOptions {
        padding,
        stride_w: stride,
        stride_h: stride,
        activation,
    })
}

/// A transpose convolution of an activation `input` with constant `weights` and `bias`.
fn tconv(
    options: OpOptions<'static>,
    input: TensorDesc<'static>,
    weights: TensorDesc<'static>,
    bias: Option<TensorDesc<'static>>,
    output: TensorDesc<'static>,
) -> OpDesc<'static, 'static> {
    let shape = konst(t(1, DataType::I32, &[4]), output.dims.as_slice());
    let mut inputs = vec![Some(shape), Some(weights), Some(input)];
    if let Some(b) = bias {
        inputs.push(Some(b));
    }
    op(BuiltinOp::TRANSPOSE_CONV, options, &inputs, &[output])
}

fn iota(n: usize) -> Vec<f32> {
    (1..=n).map(|v| v as f32).collect()
}

#[test]
fn transpose_conv_f32_vector() {
    // transpose_conv_test.cc SimpleTest: 4x4 input, 3x3 filter, stride 1, SAME.
    let filter = konst(t(2, DataType::F32, &[1, 3, 3, 1]), &iota(9));
    let op = tconv(
        options(Padding::Same, 1, Activation::None),
        t(0, DataType::F32, &[1, 4, 4, 1]),
        filter,
        None,
        t(3, DataType::F32, &[1, 4, 4, 1]),
    );
    let expected = [
        29.0, 62.0, 83.0, 75.0, 99.0, 192.0, 237.0, 198.0, 207.0, 372.0, 417.0, 330.0, 263.0,
        446.0, 485.0, 365.0,
    ];
    assert_bits(&run1::<f32>(&op, &[buf(&iota(16))]), &expected);
}

#[test]
fn transpose_conv_int8_matches_float() {
    // Stride 2, SAME, 2x2 input of two channels, 3x3 filters for three output channels, a bias
    // and RELU: small integers, so the float result is exact and the int8 one is its rounding.
    let input: Vec<f32> = vec![1.0, -2.0, 3.0, 0.0, -1.0, 2.0, 4.0, -3.0];
    let filter: Vec<f32> = (0..3 * 3 * 3 * 2)
        .map(|v| ((v * 7) % 11) as f32 - 5.0)
        .collect();
    let bias = [3.0f32, -20.0, 0.0];
    let f = tconv(
        options(Padding::Same, 2, Activation::Relu),
        t(0, DataType::F32, &[1, 2, 2, 2]),
        konst(t(2, DataType::F32, &[3, 3, 3, 2]), &filter),
        Some(konst(t(4, DataType::F32, &[3]), &bias)),
        t(3, DataType::F32, &[1, 4, 4, 3]),
    );
    let expected = run1::<f32>(&f, &[buf(&input)]);

    // Input scale 0.5 (zero point 1), weights scale 1, output scale 2 (zero point -3): the
    // integer result is `relu(x) / 2` rounded half away from zero, plus the zero point.
    let q_input: Vec<i8> = input.iter().map(|v| (v / 0.5) as i8 + 1).collect();
    let q_filter: Vec<i8> = filter.iter().map(|&v| v as i8).collect();
    let q_bias: Vec<i32> = bias.iter().map(|&v| (v / 0.5) as i32).collect();
    let q = tconv(
        options(Padding::Same, 2, Activation::Relu),
        q(t(0, DataType::I8, &[1, 2, 2, 2]), 0.5, 1),
        qc(
            konst(t(2, DataType::I8, &[3, 3, 3, 2]), &q_filter),
            &[1.0, 1.0, 1.0],
            &[0, 0, 0],
            0,
        ),
        Some(konst(t(4, DataType::I32, &[3]), &q_bias)),
        q(t(3, DataType::I8, &[1, 4, 4, 3]), 2.0, -3),
    );
    let actual = run1::<i8>(&q, &[buf(&q_input)]);
    let rounded: Vec<i8> = expected
        .iter()
        .map(|&v| (nanoglia_core::quant::round_f32(v / 2.0) as i32 - 3).clamp(-128, 127) as i8)
        .collect();
    assert_eq!(actual, rounded);
}

#[test]
fn transpose_conv_refusals() {
    let input = || q(t(0, DataType::I8, &[1, 2, 2, 2]), 0.5, 1);
    let weights = || {
        qc(
            konst(t(2, DataType::I8, &[3, 3, 3, 2]), &[1i8; 54]),
            &[1.0, 1.0, 1.0],
            &[0, 0, 0],
            0,
        )
    };
    let output = |dims: &[i32]| q(t(3, DataType::I8, dims), 2.0, -3);
    let same2 = options(Padding::Same, 2, Activation::None);
    let ok = tconv(same2, input(), weights(), None, output(&[1, 4, 4, 3]));
    assert!(select(&ok).is_some());
    // A 3x3 output is also a SAME stride 2 transpose of a 2x2 input.
    let odd = tconv(same2, input(), weights(), None, output(&[1, 3, 3, 3]));
    assert!(select(&odd).is_some());

    let mut refused = vec![
        // The output is not what the forward convolution would read the input from.
        tconv(same2, input(), weights(), None, output(&[1, 5, 4, 3])),
        tconv(
            options(Padding::Valid, 2, Activation::None),
            input(),
            weights(),
            None,
            output(&[1, 4, 4, 3]),
        ),
        // Stride 0, unsupported activation.
        tconv(
            options(Padding::Same, 0, Activation::None),
            input(),
            weights(),
            None,
            output(&[1, 4, 4, 3]),
        ),
        tconv(
            options(Padding::Same, 2, Activation::Tanh),
            input(),
            weights(),
            None,
            output(&[1, 4, 4, 3]),
        ),
        // An int8 bias, per-channel weights along another axis, weights with a zero point.
        tconv(
            same2,
            input(),
            weights(),
            Some(konst(t(4, DataType::I8, &[3]), &[0i8; 3])),
            output(&[1, 4, 4, 3]),
        ),
        tconv(
            same2,
            input(),
            qc(
                konst(t(2, DataType::I8, &[3, 3, 3, 2]), &[1i8; 54]),
                &[1.0, 1.0, 1.0],
                &[0, 0, 0],
                1,
            ),
            None,
            output(&[1, 4, 4, 3]),
        ),
        tconv(
            same2,
            input(),
            qc(
                konst(t(2, DataType::I8, &[3, 3, 3, 2]), &[1i8; 54]),
                &[1.0],
                &[1],
                0,
            ),
            None,
            output(&[1, 4, 4, 3]),
        ),
        // Hybrid: float input with int8 weights.
        tconv(
            same2,
            t(0, DataType::F32, &[1, 2, 2, 2]),
            weights(),
            None,
            t(3, DataType::F32, &[1, 4, 4, 3]),
        ),
        // Weights for another number of input channels.
        tconv(
            same2,
            input(),
            qc(
                konst(t(2, DataType::I8, &[3, 3, 3, 1]), &[1i8; 27]),
                &[1.0],
                &[0],
                0,
            ),
            None,
            output(&[1, 4, 4, 3]),
        ),
    ];
    // An output shape computed at run time, or of another type.
    refused.push(op(
        BuiltinOp::TRANSPOSE_CONV,
        same2,
        &[
            Some(t(1, DataType::I32, &[4])),
            Some(weights()),
            Some(input()),
        ],
        &[output(&[1, 4, 4, 3])],
    ));
    refused.push(op(
        BuiltinOp::TRANSPOSE_CONV,
        same2,
        &[
            Some(konst(t(1, DataType::I64, &[4]), &[1i64, 4, 4, 3])),
            Some(weights()),
            Some(input()),
        ],
        &[output(&[1, 4, 4, 3])],
    ));
    for (i, op) in refused.iter().enumerate() {
        assert!(select(op).is_none(), "case {i}");
    }
}
