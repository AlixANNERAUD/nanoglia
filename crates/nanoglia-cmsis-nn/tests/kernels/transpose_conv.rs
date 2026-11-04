use nanoglia_core::{Activation, BuiltinOp, DataType, OpOptions, Padding, TransposeConvOptions};

use crate::harness::{ACTIVATIONS, Op, Rng, Tensor};
use crate::reference::{self, ConvParams};

/// The input size TFLite expects for an output of size `out` (`ComputeOutSize` of the forward
/// convolution) and the padding before the output.
fn forward(same: bool, out: i32, filter: i32, stride: i32) -> (i32, i32) {
    let input = if same {
        (out + stride - 1) / stride
    } else {
        (out + stride - filter) / stride
    };
    let total = ((input - 1) * stride + filter - out).max(0);
    (input, total / 2)
}

struct Case {
    batches: i32,
    in_c: i32,
    out_c: i32,
    filter: (i32, i32),
    stride: (i32, i32),
    padding: Padding,
    output: (i32, i32),
    per_channel: bool,
    bias: bool,
    activation: Activation,
}

/// Runs `case` if the backend claims it, and checks it against the reference kernel. Returns
/// whether the backend claimed it.
fn run(case: &Case, seed: u64) -> bool {
    let mut rng = Rng::new(seed);
    let same = case.padding == Padding::Same;
    let (in_h, pad_h) = forward(same, case.output.0, case.filter.0, case.stride.0);
    let (in_w, pad_w) = forward(same, case.output.1, case.filter.1, case.stride.1);
    assert!(in_h > 0 && in_w > 0);
    let input_dims = [case.batches, in_h, in_w, case.in_c];
    let filter_dims = [case.out_c, case.filter.0, case.filter.1, case.in_c];
    let output_dims = [case.batches, case.output.0, case.output.1, case.out_c];

    let input_scale = rng.float(0.01, 0.1);
    let input_zero_point = rng.range(-128, 127);
    let channels = if case.per_channel {
        case.out_c as usize
    } else {
        1
    };
    let filter_scales: Vec<f32> = (0..channels).map(|_| rng.float(0.002, 0.02)).collect();
    let max_filter_scale = filter_scales.iter().copied().fold(0.0, f32::max);
    let depth = (case.filter.0 * case.filter.1 * case.in_c) as f32;
    let output_scale = input_scale * max_filter_scale * depth.sqrt() * 40.0 * rng.float(0.5, 1.5);
    let output_zero_point = rng.range(-50, 50);

    let input_values = rng.i8s(input_dims.iter().product::<i32>() as usize);
    let filter_values = rng.i8s(filter_dims.iter().product::<i32>() as usize);
    let bias_values = rng.i32s(case.out_c as usize, -5000, 5000);

    let shape_bytes: Vec<u8> = output_dims.iter().flat_map(|d| d.to_le_bytes()).collect();
    let shape = Tensor::constant(DataType::I32, &[4], &shape_bytes);
    let input = Tensor::activation(DataType::I8, &input_dims, input_scale, input_zero_point)
        .with_data_i8(&input_values);
    let filter = Tensor::weights(&filter_dims, &filter_values, &filter_scales, 0);
    let bias = Tensor::bias(&bias_values);
    let output = Tensor::activation(DataType::I8, &output_dims, output_scale, output_zero_point);
    let options = TransposeConvOptions {
        padding: case.padding,
        stride_h: case.stride.0,
        stride_w: case.stride.1,
        activation: case.activation,
    };
    let op = Op::new(BuiltinOp::TRANSPOSE_CONV, OpOptions::TransposeConv(options))
        .input(&shape)
        .input(&filter)
        .input(&input)
        .optional(case.bias.then_some(&bias))
        .output(&output);
    if !op.selected() {
        return false;
    }
    let got = op.run();

    let params = ConvParams {
        input_offset: -input_zero_point,
        output_offset: output_zero_point,
        stride: case.stride,
        dilation: (1, 1),
        padding: (pad_h, pad_w),
        activation: reference::activation_range(case.activation, output_scale, output_zero_point),
    };
    let multipliers = reference::channel_multipliers(
        input_scale,
        &filter_scales,
        output_scale,
        case.out_c as usize,
    );
    let want = reference::transpose_conv_per_channel(
        &params,
        &multipliers,
        input_dims,
        &input_values,
        filter_dims,
        &filter_values,
        case.bias.then_some(&bias_values[..]),
        output_dims,
    );
    let name = format!(
        "seed {seed}: filter {:?} stride {:?} {:?} output {:?} in_c {}",
        case.filter, case.stride, case.padding, case.output, case.in_c
    );
    assert_eq!(got, want, "{name}");
    true
}

/// Every output size of two input sizes, with filters smaller than, equal to and larger than the
/// strides, both CMSIS-NN paths (up to 16 input channels, and more with strides up to 2), and both
/// paddings. Whatever the backend claims must match the reference kernel; the convolution of the
/// upsampled input (strides up to 2, more than 16 input channels) takes every size.
#[test]
fn transpose_conv_matches_the_reference() {
    let mut seed = 0;
    let (mut claimed, mut refused) = (0, 0);
    for stride in [(1, 1), (2, 2), (1, 2), (3, 2), (3, 3)] {
        for filter in [(1, 1), (2, 2), (3, 3), (4, 3), (2, 5)] {
            for padding in [Padding::Same, Padding::Valid] {
                for in_c in [3, 17] {
                    for input in [(2, 3), (3, 2)] {
                        // The output sizes whose forward convolution reads `input`.
                        let sizes = |i: usize, input: i32| {
                            let (f, s) = if i == 0 {
                                (filter.0, stride.0)
                            } else {
                                (filter.1, stride.1)
                            };
                            (1..=input * s + f)
                                .filter(|&o| forward(padding == Padding::Same, o, f, s).0 == input)
                                .collect::<Vec<_>>()
                        };
                        for out_h in sizes(0, input.0) {
                            for out_w in sizes(1, input.1) {
                                seed += 1;
                                let case = Case {
                                    batches: 1 + (seed % 2) as i32,
                                    in_c,
                                    out_c: 2 + (seed % 3) as i32,
                                    filter,
                                    stride,
                                    padding,
                                    output: (out_h, out_w),
                                    per_channel: seed % 5 != 0,
                                    bias: seed % 3 != 0,
                                    activation: ACTIVATIONS[(seed % 4) as usize],
                                };
                                let reverse = in_c > 16 && stride.0 <= 2 && stride.1 <= 2;
                                if run(&case, seed) {
                                    claimed += 1;
                                } else {
                                    assert!(!reverse, "seed {seed}: refused");
                                    refused += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // The rows or columns of `arm_transpose_conv_s8` do not fit some output sizes: the Rust
    // backend runs them.
    assert!(
        claimed > 400 && refused > 0,
        "{claimed} claimed, {refused} refused"
    );
}

#[test]
fn transpose_conv_paths() {
    let case = |in_c, stride, output, filter, padding| Case {
        batches: 1,
        in_c,
        out_c: 4,
        filter,
        stride,
        padding,
        output,
        per_channel: true,
        bias: true,
        activation: Activation::None,
    };
    // `arm_transpose_conv_s8`, and the convolution of the upsampled input.
    assert!(run(&case(8, (2, 2), (8, 8), (3, 3), Padding::Same), 1));
    assert!(run(&case(24, (2, 2), (8, 8), (3, 3), Padding::Same), 2));
    // Rows past the output: a 2x2 stride 2 SAME transpose of 4 rows into 7.
    assert!(!run(&case(8, (2, 2), (7, 7), (2, 2), Padding::Same), 3));
    // The upsampled-input convolution handles it.
    assert!(run(&case(24, (2, 2), (7, 7), (2, 2), Padding::Same), 4));
    // Columns past the rolling buffer: the largest VALID output.
    assert!(!run(&case(8, (2, 2), (10, 10), (3, 3), Padding::Valid), 5));
    // Stride larger than the filter, VALID: rows past the output.
    assert!(!run(&case(8, (3, 3), (8, 8), (2, 2), Padding::Valid), 6));
    // A horizontal stride larger than the filter width and the vertical stride: CMSIS-NN's
    // buffer size function uses the vertical stride for the width of its rows.
    assert!(run(&case(8, (1, 3), (4, 10), (1, 1), Padding::Valid), 7));
}

#[test]
fn transpose_conv_scratch_covers_the_rolling_buffer() {
    // Input 4x4, stride (1, 3), 1x1 filter, 4 output channels: rows of
    // `(4 - 1) * 3 + max(1, 3)` columns, one row, 4 bytes per value.
    let shape_bytes: Vec<u8> = [1i32, 4, 10, 4]
        .iter()
        .flat_map(|d| d.to_le_bytes())
        .collect();
    let shape = Tensor::constant(DataType::I32, &[4], &shape_bytes);
    let input = Tensor::activation(DataType::I8, &[1, 4, 4, 8], 0.05, 0);
    let filter = Tensor::weights(&[4, 1, 1, 8], &[1; 32], &[0.01], 0);
    let output = Tensor::activation(DataType::I8, &[1, 4, 10, 4], 0.1, 0);
    let options = TransposeConvOptions {
        padding: Padding::Valid,
        stride_h: 1,
        stride_w: 3,
        activation: Activation::None,
    };
    let op = Op::new(BuiltinOp::TRANSPOSE_CONV, OpOptions::TransposeConv(options))
        .input(&shape)
        .input(&filter)
        .input(&input)
        .output(&output);
    let scratch = op.with_desc(|desc| {
        use nanoglia_core::backend::Backend;
        let kernel = nanoglia_cmsis_nn::BACKEND.select(desc).expect("refused");
        kernel.plan(desc).expect("plan").scratch
    });
    assert!(scratch >= 12 * 4 * 4, "{scratch}");
}
