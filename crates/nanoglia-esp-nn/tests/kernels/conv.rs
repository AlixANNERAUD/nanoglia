use nanoglia_core::{Activation, BuiltinOp, Conv2DOptions, DataType, OpOptions, Padding};

use crate::harness::{ACTIVATIONS, Op, Rng, Tensor};
use crate::reference::{self, ConvParams};

/// A convolution test case: NHWC input, `[out_c, kh, kw, in_c / groups]` filter.
struct Case {
    input: [i32; 4],
    filter: [i32; 4],
    stride: (i32, i32),
    dilation: (i32, i32),
    padding: Padding,
    per_channel: bool,
    bias: bool,
}

fn run(case: &Case, activation: Activation, seed: u64, check_spread: bool) {
    let mut rng = Rng::new(seed);
    let [batches, in_h, in_w, _] = case.input;
    let [out_c, kh, kw, kc] = case.filter;
    let same = case.padding == Padding::Same;
    let (out_h, pad_h) = reference::padding(same, in_h, kh, case.stride.0, case.dilation.0);
    let (out_w, pad_w) = reference::padding(same, in_w, kw, case.stride.1, case.dilation.1);
    let output_dims = [batches, out_h, out_w, out_c];

    let input_scale = rng.float(0.01, 0.1);
    let input_zero_point = rng.range(-128, 127);
    let channels = if case.per_channel { out_c as usize } else { 1 };
    let filter_scales: Vec<f32> = (0..channels).map(|_| rng.float(0.002, 0.02)).collect();
    let max_filter_scale = filter_scales.iter().copied().fold(0.0, f32::max);
    let depth = (kh * kw * kc) as f32;
    let output_scale = input_scale * max_filter_scale * depth.sqrt() * 60.0 * rng.float(0.5, 1.5);
    let output_zero_point = rng.range(-50, 50);

    let input_values = rng.i8s(case.input.iter().product::<i32>() as usize);
    let filter_values = rng.i8s(case.filter.iter().product::<i32>() as usize);
    let bias_values = rng.i32s(out_c as usize, -5000, 5000);

    let input = Tensor::activation(DataType::I8, &case.input, input_scale, input_zero_point)
        .with_data_i8(&input_values);
    let filter = Tensor::weights(&case.filter, &filter_values, &filter_scales, 0);
    let bias = Tensor::bias(&bias_values);
    let output = Tensor::activation(DataType::I8, &output_dims, output_scale, output_zero_point);

    let options = Conv2DOptions {
        padding: case.padding,
        stride_h: case.stride.0,
        stride_w: case.stride.1,
        dilation_h: case.dilation.0,
        dilation_w: case.dilation.1,
        activation,
    };
    let op = Op::new(BuiltinOp::CONV_2D, OpOptions::Conv2D(options))
        .input(&input)
        .input(&filter)
        .optional(case.bias.then_some(&bias))
        .output(&output);
    if case.dilation != (1, 1) {
        // ESP-NN ignores dilations: the next backend runs these.
        assert!(!op.selected(), "seed {seed}");
        return;
    }
    assert_eq!(op.kernel_name(), "esp-nn/conv2d_s8");
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
    let want = reference::conv_per_channel(
        &params,
        &multipliers,
        case.input,
        &input_values,
        case.filter,
        &filter_values,
        case.bias.then_some(&bias_values[..]),
        output_dims,
    );
    assert_eq!(got, want, "seed {seed}");
    if check_spread && activation == Activation::None {
        crate::harness::assert_spread(&want);
    }
}

fn run_all(case: &Case, seed: u64) {
    for (i, activation) in ACTIVATIONS.into_iter().enumerate() {
        run(case, activation, seed * 16 + i as u64, true);
    }
}

#[test]
fn conv_3x3_same() {
    run_all(
        &Case {
            input: [1, 5, 5, 3],
            filter: [4, 3, 3, 3],
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
fn conv_strided_valid_per_tensor_no_bias() {
    run_all(
        &Case {
            input: [1, 9, 7, 5],
            filter: [6, 3, 2, 5],
            stride: (2, 3),
            dilation: (1, 1),
            padding: Padding::Valid,
            per_channel: false,
            bias: false,
        },
        2,
    );
}

#[test]
fn conv_same_even_input_asymmetric_padding() {
    run_all(
        &Case {
            input: [1, 8, 6, 4],
            filter: [3, 4, 4, 4],
            stride: (2, 2),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        3,
    );
}

#[test]
fn conv_1x1_unit_stride() {
    run_all(
        &Case {
            input: [1, 4, 5, 8],
            filter: [7, 1, 1, 8],
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Valid,
            per_channel: true,
            bias: true,
        },
        4,
    );
}

#[test]
fn conv_1x1_strided() {
    run_all(
        &Case {
            input: [2, 7, 7, 6],
            filter: [5, 1, 1, 6],
            stride: (2, 2),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        5,
    );
}

#[test]
fn conv_1xn() {
    // Height 1 and stride * channels a multiple of 4: arm_convolve_1_x_n_s8.
    run_all(
        &Case {
            input: [1, 1, 12, 4],
            filter: [6, 1, 3, 4],
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        6,
    );
    run_all(
        &Case {
            input: [1, 1, 15, 8],
            filter: [3, 1, 5, 8],
            stride: (1, 2),
            dilation: (1, 1),
            padding: Padding::Valid,
            per_channel: false,
            bias: false,
        },
        7,
    );
}

#[test]
fn conv_dilated() {
    run_all(
        &Case {
            input: [1, 9, 9, 3],
            filter: [4, 3, 3, 3],
            stride: (1, 1),
            dilation: (2, 3),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        8,
    );
}

#[test]
fn conv_batched() {
    run_all(
        &Case {
            input: [3, 6, 5, 3],
            filter: [5, 3, 3, 3],
            stride: (1, 2),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        9,
    );
}

#[test]
fn conv_grouped_without_bias() {
    run_all(
        &Case {
            input: [1, 5, 5, 6],
            filter: [9, 3, 3, 2],
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: false,
        },
        13,
    );
}

#[test]
fn conv_grouped() {
    run_all(
        &Case {
            input: [1, 6, 6, 4],
            filter: [6, 3, 3, 2],
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        10,
    );
}

#[test]
fn conv_wide() {
    run_all(
        &Case {
            input: [1, 10, 10, 16],
            filter: [32, 3, 3, 16],
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        11,
    );
}

#[test]
fn conv_random_shapes() {
    let mut rng = Rng::new(12);
    for seed in 0..40 {
        let in_c = rng.range(1, 9);
        let groups = if in_c % 2 == 0 && rng.range(0, 3) == 0 {
            2
        } else {
            1
        };
        let filter_c = in_c / groups;
        let out_c = groups * rng.range(1, 6);
        let case = Case {
            input: [rng.range(1, 2), rng.range(1, 9), rng.range(1, 9), in_c],
            filter: [out_c, rng.range(1, 4), rng.range(1, 4), filter_c],
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
        // VALID padding needs the dilated filter to fit in the input.
        let fits = |input: i32, filter: i32, dilation: i32| (filter - 1) * dilation < input;
        if case.padding == Padding::Valid
            && !(fits(case.input[1], case.filter[1], case.dilation.0)
                && fits(case.input[2], case.filter[2], case.dilation.1))
        {
            continue;
        }
        let activation = ACTIVATIONS[rng.range(0, 3) as usize];
        run(&case, activation, 1000 + seed, false);
    }
}
