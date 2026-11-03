//! 16x8 `CONV_2D` and `FULLY_CONNECTED`: int16 activations (zero point 0), int8 weights, int64
//! bias.

use nanoglia_core::{
    Activation, BuiltinOp, Conv2DOptions, DataType, FullyConnectedOptions, OpOptions, Padding,
};

use crate::harness::{ACTIVATIONS, Op, Rng, Tensor};
use crate::reference::{self, ConvParams};

/// A 16x8 convolution test case: NHWC input, `[out_c, kh, kw, in_c]` filter.
struct ConvCase {
    input: [i32; 4],
    filter: [i32; 4],
    stride: (i32, i32),
    dilation: (i32, i32),
    padding: Padding,
    per_channel: bool,
    bias: bool,
}

fn conv_options(case: &ConvCase, activation: Activation) -> OpOptions<'static> {
    OpOptions::Conv2D(Conv2DOptions {
        padding: case.padding,
        stride_h: case.stride.0,
        stride_w: case.stride.1,
        dilation_h: case.dilation.0,
        dilation_w: case.dilation.1,
        activation,
    })
}

fn run_conv(case: &ConvCase, activation: Activation, seed: u64) {
    let mut rng = Rng::new(seed);
    let [batches, in_h, in_w, _] = case.input;
    let [out_c, kh, kw, kc] = case.filter;
    let same = case.padding == Padding::Same;
    let (out_h, pad_h) = reference::padding(same, in_h, kh, case.stride.0, case.dilation.0);
    let (out_w, pad_w) = reference::padding(same, in_w, kw, case.stride.1, case.dilation.1);
    let output_dims = [batches, out_h, out_w, out_c];

    let input_scale = rng.float(1e-4, 1e-3);
    let channels = if case.per_channel { out_c as usize } else { 1 };
    let filter_scales: Vec<f32> = (0..channels).map(|_| rng.float(0.002, 0.02)).collect();
    let max_filter_scale = filter_scales.iter().copied().fold(0.0, f32::max);
    let depth = (kh * kw * kc) as f32;
    let output_scale = input_scale * max_filter_scale * depth.sqrt() * 80.0 * rng.float(0.5, 1.5);

    let input_values = rng.i16s(case.input.iter().product::<i32>() as usize);
    let filter_values = rng.i8s(case.filter.iter().product::<i32>() as usize);
    let bias_values: Vec<i64> = (0..out_c)
        .map(|_| rng.range(-2_000_000, 2_000_000) as i64)
        .collect();

    let input =
        Tensor::activation(DataType::I16, &case.input, input_scale, 0).with_data_i16(&input_values);
    let filter = Tensor::weights(&case.filter, &filter_values, &filter_scales, 0);
    let bias = Tensor::bias_i64(&bias_values);
    let output = Tensor::activation(DataType::I16, &output_dims, output_scale, 0);
    let op = Op::new(BuiltinOp::CONV_2D, conv_options(case, activation))
        .input(&input)
        .input(&filter)
        .optional(case.bias.then_some(&bias))
        .output(&output);
    assert_eq!(op.kernel_name(), "cmsis-nn/conv2d_s16");

    let params = ConvParams {
        input_offset: 0,
        output_offset: 0,
        stride: case.stride,
        dilation: case.dilation,
        padding: (pad_h, pad_w),
        activation: reference::activation_range_i16(activation, output_scale),
    };
    let multipliers =
        reference::channel_multipliers(input_scale, &filter_scales, output_scale, out_c as usize);
    let want = reference::conv_per_channel_16x8(
        &params,
        &multipliers,
        case.input,
        &input_values,
        case.filter,
        &filter_values,
        case.bias.then_some(&bias_values[..]),
        output_dims,
    );
    let got = op.run_i16();
    assert_eq!(got, want, "seed {seed}");
    if activation == Activation::None {
        let mut distinct = want.clone();
        distinct.sort();
        distinct.dedup();
        assert!(distinct.len() > want.len().min(8) / 2, "degenerate output");
    }
}

#[test]
fn conv_16x8() {
    let cases = [
        ConvCase {
            input: [1, 8, 8, 4],
            filter: [8, 3, 3, 4],
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        // An odd number of output pixels (CMSIS-NN's leftover column path), no bias.
        ConvCase {
            input: [2, 5, 7, 3],
            filter: [5, 3, 2, 3],
            stride: (2, 1),
            dilation: (1, 2),
            padding: Padding::Valid,
            per_channel: false,
            bias: false,
        },
        // 511 filter values per channel, the most CMSIS-NN accumulates safely.
        ConvCase {
            input: [1, 3, 3, 73],
            filter: [3, 1, 7, 73],
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
    ];
    for (i, case) in cases.iter().enumerate() {
        for (j, activation) in ACTIVATIONS.into_iter().enumerate() {
            run_conv(case, activation, 300 + (i * 8 + j) as u64);
        }
    }
}

#[test]
fn conv_16x8_refusals() {
    let input = Tensor::activation(DataType::I16, &[1, 4, 4, 2], 0.001, 0);
    let output = Tensor::activation(DataType::I16, &[1, 4, 4, 3], 0.002, 0);
    let filter = Tensor::weights(&[3, 3, 3, 2], &[1; 54], &[0.01], 0);
    let case = ConvCase {
        input: [1, 4, 4, 2],
        filter: [3, 3, 3, 2],
        stride: (1, 1),
        dilation: (1, 1),
        padding: Padding::Same,
        per_channel: false,
        bias: false,
    };
    let selected = |input: &Tensor, filter: &Tensor, bias: Option<&Tensor>, output: &Tensor| {
        Op::new(BuiltinOp::CONV_2D, conv_options(&case, Activation::None))
            .input(input)
            .input(filter)
            .optional(bias)
            .output(output)
            .selected()
    };
    assert!(selected(&input, &filter, None, &output));
    assert!(selected(
        &input,
        &filter,
        Some(&Tensor::bias_i64(&[1, 2, 3])),
        &output
    ));
    // int32 bias: LiteRT reads it according to an option nanoglia does not decode.
    assert!(!selected(
        &input,
        &filter,
        Some(&Tensor::bias(&[1, 2, 3])),
        &output
    ));
    // Nonzero zero points.
    let shifted = Tensor::activation(DataType::I16, &[1, 4, 4, 2], 0.001, 1);
    assert!(!selected(&shifted, &filter, None, &output));
    let shifted = Tensor::activation(DataType::I16, &[1, 4, 4, 3], 0.002, -1);
    assert!(!selected(&input, &filter, None, &shifted));
    // Grouped convolution.
    let grouped = Tensor::weights(&[3, 3, 3, 1], &[1; 27], &[0.01], 0);
    assert!(!selected(&input, &grouped, None, &output));
    // More than 511 filter values per output channel.
    let wide_input = Tensor::activation(DataType::I16, &[1, 4, 4, 57], 0.001, 0);
    let wide = Tensor::weights(&[3, 3, 3, 57], &[1; 1539], &[0.01], 0);
    assert!(!selected(&wide_input, &wide, None, &output));
    // A requantization shift of 8 or more.
    let tiny_output = Tensor::activation(DataType::I16, &[1, 4, 4, 3], 1e-8, 0);
    assert!(!selected(&input, &filter, None, &tiny_output));
}

fn fc_op<'t>(
    input: &'t Tensor,
    filter: &'t Tensor,
    bias: Option<&'t Tensor>,
    output: &'t Tensor,
    activation: Activation,
) -> Op<'t> {
    let options = FullyConnectedOptions {
        activation,
        default_weights_format: true,
        keep_num_dims: false,
        asymmetric_quantize_inputs: false,
    };
    Op::new(
        BuiltinOp::FULLY_CONNECTED,
        OpOptions::FullyConnected(options),
    )
    .input(input)
    .input(filter)
    .optional(bias)
    .output(output)
}

fn run_fc(batches: usize, accum_depth: usize, output_depth: usize, bias: bool, seed: u64) {
    let mut rng = Rng::new(seed);
    let activation = ACTIVATIONS[(seed % 4) as usize];
    let input_scale = rng.float(1e-4, 1e-3);
    let filter_scale = rng.float(0.002, 0.02);
    let output_scale =
        input_scale * filter_scale * (accum_depth as f32).sqrt() * 80.0 * rng.float(0.5, 1.5);
    let input_values = rng.i16s(batches * accum_depth);
    let filter_values = rng.i8s(output_depth * accum_depth);
    let bias_values: Vec<i64> = (0..output_depth)
        .map(|_| rng.range(-2_000_000, 2_000_000) as i64)
        .collect();
    let input = Tensor::activation(
        DataType::I16,
        &[batches as i32, accum_depth as i32],
        input_scale,
        0,
    )
    .with_data_i16(&input_values);
    let filter = Tensor::weights(
        &[output_depth as i32, accum_depth as i32],
        &filter_values,
        &[filter_scale],
        0,
    );
    let bias_tensor = Tensor::bias_i64(&bias_values);
    let output = Tensor::activation(
        DataType::I16,
        &[batches as i32, output_depth as i32],
        output_scale,
        0,
    );
    let op = fc_op(
        &input,
        &filter,
        bias.then_some(&bias_tensor),
        &output,
        activation,
    );
    assert_eq!(op.kernel_name(), "cmsis-nn/fully_connected_s16");
    let multipliers =
        reference::channel_multipliers(input_scale, &[filter_scale], output_scale, output_depth);
    let want = reference::fully_connected_16x8(
        reference::activation_range_i16(activation, output_scale),
        &multipliers,
        batches,
        accum_depth,
        output_depth,
        &input_values,
        &filter_values,
        bias.then_some(&bias_values[..]),
    );
    assert_eq!(op.run_i16(), want, "seed {seed}");
}

#[test]
fn fully_connected_16x8() {
    run_fc(2, 16, 8, true, 0);
    run_fc(1, 37, 5, false, 1);
    run_fc(3, 600, 7, true, 2); // more columns than CMSIS-NN accumulates in 32 bits
    run_fc(4, 1, 1, true, 3);
}

#[test]
fn fully_connected_16x8_refusals() {
    let input = Tensor::activation(DataType::I16, &[2, 4], 0.001, 0);
    let output = Tensor::activation(DataType::I16, &[2, 3], 0.002, 0);
    let filter = Tensor::weights(&[3, 4], &[1; 12], &[0.01], 0);
    let none = Activation::None;
    assert!(fc_op(&input, &filter, None, &output, none).selected());
    // Per-channel weights (refused by TFLite Micro's CMSIS-NN kernel too).
    let per_channel = Tensor::weights(&[3, 4], &[1; 12], &[0.01, 0.02, 0.03], 0);
    assert!(!fc_op(&input, &per_channel, None, &output, none).selected());
    // int32 bias.
    let bias = Tensor::bias(&[1, 2, 3]);
    assert!(!fc_op(&input, &filter, Some(&bias), &output, none).selected());
    // Nonzero zero points.
    let shifted = Tensor::activation(DataType::I16, &[2, 4], 0.001, 2);
    assert!(!fc_op(&shifted, &filter, None, &output, none).selected());
}
