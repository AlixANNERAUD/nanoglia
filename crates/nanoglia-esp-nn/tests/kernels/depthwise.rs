use nanoglia_core::{Activation, BuiltinOp, DataType, DepthwiseConv2DOptions, OpOptions, Padding};

use crate::harness::{ACTIVATIONS, Op, Rng, Tensor, assert_spread};
use crate::reference::{self, ConvParams};

/// A depthwise convolution test case: NHWC input, `(kh, kw)` filter.
struct Case {
    input: [i32; 4],
    kernel: (i32, i32),
    depth_multiplier: i32,
    stride: (i32, i32),
    dilation: (i32, i32),
    padding: Padding,
    per_channel: bool,
    bias: bool,
}

fn run(case: &Case, activation: Activation, seed: u64, check_spread: bool) {
    let mut rng = Rng::new(seed);
    let [batches, in_h, in_w, in_c] = case.input;
    let out_c = in_c * case.depth_multiplier;
    let filter_dims = [1, case.kernel.0, case.kernel.1, out_c];
    let same = case.padding == Padding::Same;
    let (out_h, pad_h) =
        reference::padding(same, in_h, case.kernel.0, case.stride.0, case.dilation.0);
    let (out_w, pad_w) =
        reference::padding(same, in_w, case.kernel.1, case.stride.1, case.dilation.1);
    let output_dims = [batches, out_h, out_w, out_c];

    let input_scale = rng.float(0.01, 0.1);
    let input_zero_point = rng.range(-128, 127);
    let channels = if case.per_channel { out_c as usize } else { 1 };
    let filter_scales: Vec<f32> = (0..channels).map(|_| rng.float(0.002, 0.02)).collect();
    let max_filter_scale = filter_scales.iter().copied().fold(0.0, f32::max);
    let depth = (case.kernel.0 * case.kernel.1) as f32;
    let output_scale = input_scale * max_filter_scale * depth.sqrt() * 60.0 * rng.float(0.5, 1.5);
    let output_zero_point = rng.range(-50, 50);

    let input_values = rng.i8s(case.input.iter().product::<i32>() as usize);
    let filter_values = rng.i8s(filter_dims.iter().product::<i32>() as usize);
    let bias_values = rng.i32s(out_c as usize, -3000, 3000);

    let input = Tensor::activation(DataType::I8, &case.input, input_scale, input_zero_point)
        .with_data_i8(&input_values);
    let filter = Tensor::weights(&filter_dims, &filter_values, &filter_scales, 3);
    let bias = Tensor::bias(&bias_values);
    let output = Tensor::activation(DataType::I8, &output_dims, output_scale, output_zero_point);

    let options = DepthwiseConv2DOptions {
        padding: case.padding,
        stride_h: case.stride.0,
        stride_w: case.stride.1,
        depth_multiplier: case.depth_multiplier,
        dilation_h: case.dilation.0,
        dilation_w: case.dilation.1,
        activation,
    };
    let op = Op::new(
        BuiltinOp::DEPTHWISE_CONV_2D,
        OpOptions::DepthwiseConv2D(options),
    )
    .input(&input)
    .input(&filter)
    .optional(case.bias.then_some(&bias))
    .output(&output);
    if case.dilation != (1, 1) {
        // ESP-NN ignores dilations: the next backend runs these.
        assert!(!op.selected(), "seed {seed}");
        return;
    }
    assert_eq!(op.kernel_name(), "esp-nn/depthwise_conv2d_s8");
    let got = op.run();

    let params = ConvParams {
        input_offset: -input_zero_point,
        output_offset: output_zero_point,
        stride: case.stride,
        dilation: case.dilation,
        padding: (pad_h, pad_w),
        activation: reference::activation_range(activation, output_scale, output_zero_point),
    };
    let multipliers =
        reference::channel_multipliers(input_scale, &filter_scales, output_scale, out_c as usize);
    let want = reference::depthwise_conv_per_channel(
        &params,
        case.depth_multiplier,
        &multipliers,
        case.input,
        &input_values,
        filter_dims,
        &filter_values,
        case.bias.then_some(&bias_values[..]),
        output_dims,
    );
    assert_eq!(got, want, "seed {seed}");
    if check_spread && activation == Activation::None {
        assert_spread(&want);
    }
}

fn run_all(case: &Case, seed: u64) {
    for (i, activation) in ACTIVATIONS.into_iter().enumerate() {
        run(case, activation, seed * 16 + i as u64, true);
    }
}

#[test]
fn depthwise_3x3_same() {
    // Unit multiplier, 3x3, padding <= 1: arm_depthwise_conv_3x3_s8 without Helium.
    run_all(
        &Case {
            input: [1, 7, 6, 8],
            kernel: (3, 3),
            depth_multiplier: 1,
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        1,
    );
}

#[test]
fn depthwise_strided_3x3() {
    run_all(
        &Case {
            input: [1, 10, 10, 5],
            kernel: (3, 3),
            depth_multiplier: 1,
            stride: (2, 2),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        2,
    );
}

#[test]
fn depthwise_5x5_opt() {
    // Unit multiplier, larger padding: arm_depthwise_conv_s8_opt.
    run_all(
        &Case {
            input: [1, 9, 8, 12],
            kernel: (5, 5),
            depth_multiplier: 1,
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: false,
        },
        3,
    );
}

#[test]
fn depthwise_multiplier() {
    run_all(
        &Case {
            input: [1, 8, 7, 3],
            kernel: (3, 2),
            depth_multiplier: 2,
            stride: (2, 1),
            dilation: (1, 1),
            padding: Padding::Valid,
            per_channel: true,
            bias: true,
        },
        4,
    );
    // A single input channel with many outputs (converted to a convolution on Helium).
    run_all(
        &Case {
            input: [1, 6, 6, 1],
            kernel: (3, 3),
            depth_multiplier: 8,
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        5,
    );
}

#[test]
fn depthwise_dilated() {
    run_all(
        &Case {
            input: [1, 9, 9, 4],
            kernel: (3, 3),
            depth_multiplier: 1,
            stride: (1, 1),
            dilation: (2, 2),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        6,
    );
}

#[test]
fn depthwise_batched_per_tensor() {
    run_all(
        &Case {
            input: [2, 5, 6, 4],
            kernel: (3, 3),
            depth_multiplier: 1,
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: false,
            bias: true,
        },
        7,
    );
}

#[test]
fn depthwise_random_shapes() {
    let mut rng = Rng::new(8);
    for seed in 0..40 {
        let case = Case {
            input: [
                rng.range(1, 2),
                rng.range(1, 9),
                rng.range(1, 9),
                rng.range(1, 10),
            ],
            kernel: (rng.range(1, 4), rng.range(1, 4)),
            depth_multiplier: rng.range(1, 3),
            stride: (rng.range(1, 3), rng.range(1, 3)),
            dilation: (rng.range(1, 2), rng.range(1, 2)),
            padding: if rng.range(0, 1) == 0 {
                Padding::Same
            } else {
                Padding::Valid
            },
            per_channel: rng.range(0, 3) != 0,
            bias: rng.range(0, 3) != 0,
        };
        let fits = |input: i32, filter: i32, dilation: i32| (filter - 1) * dilation < input;
        if case.padding == Padding::Valid
            && !(fits(case.input[1], case.kernel.0, case.dilation.0)
                && fits(case.input[2], case.kernel.1, case.dilation.1))
        {
            continue;
        }
        let activation = ACTIVATIONS[rng.range(0, 3) as usize];
        run(&case, activation, 1000 + seed, false);
    }
}
