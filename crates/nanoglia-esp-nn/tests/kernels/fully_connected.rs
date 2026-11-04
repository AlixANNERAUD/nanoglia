use nanoglia_core::{Activation, BuiltinOp, DataType, FullyConnectedOptions, OpOptions};

use crate::harness::{ACTIVATIONS, Op, Rng, Tensor, assert_spread};
use crate::reference;

/// A fully-connected test case: the input is flattened to `[batches, accum_depth]`.
struct Case<'a> {
    input: &'a [i32],
    output: &'a [i32],
    accum_depth: i32,
    per_channel: bool,
    bias: bool,
}

fn run(case: &Case<'_>, activation: Activation, seed: u64, check_spread: bool) {
    let mut rng = Rng::new(seed);
    let output_depth = *case.output.last().unwrap();
    let batches = case.output.iter().product::<i32>() / output_depth;
    assert_eq!(
        case.input.iter().product::<i32>(),
        batches * case.accum_depth
    );
    let filter_dims = [output_depth, case.accum_depth];

    let input_scale = rng.float(0.01, 0.1);
    let input_zero_point = rng.range(-128, 127);
    let channels = if case.per_channel {
        output_depth as usize
    } else {
        1
    };
    let filter_scales: Vec<f32> = (0..channels).map(|_| rng.float(0.002, 0.02)).collect();
    let max_filter_scale = filter_scales.iter().copied().fold(0.0, f32::max);
    let output_scale = input_scale
        * max_filter_scale
        * (case.accum_depth as f32).sqrt()
        * 60.0
        * rng.float(0.5, 1.5);
    let output_zero_point = rng.range(-50, 50);

    let input_values = rng.i8s((batches * case.accum_depth) as usize);
    let filter_values = rng.i8s((output_depth * case.accum_depth) as usize);
    let bias_values = rng.i32s(output_depth as usize, -5000, 5000);

    let input = Tensor::activation(DataType::I8, case.input, input_scale, input_zero_point)
        .with_data_i8(&input_values);
    let filter = Tensor::weights(&filter_dims, &filter_values, &filter_scales, 0);
    let bias = Tensor::bias(&bias_values);
    let output = Tensor::activation(DataType::I8, case.output, output_scale, output_zero_point);

    let options = FullyConnectedOptions {
        activation,
        default_weights_format: true,
        keep_num_dims: case.output.len() > 2,
        asymmetric_quantize_inputs: false,
    };
    let op = Op::new(
        BuiltinOp::FULLY_CONNECTED,
        OpOptions::FullyConnected(options),
    )
    .input(&input)
    .input(&filter)
    .optional(case.bias.then_some(&bias))
    .output(&output);
    assert_eq!(op.kernel_name(), "esp-nn/fully_connected_s8");
    let got = op.run();

    let multipliers = reference::channel_multipliers(
        input_scale,
        &filter_scales,
        output_scale,
        output_depth as usize,
    );
    let want = reference::fully_connected(
        -input_zero_point,
        output_zero_point,
        reference::activation_range(activation, output_scale, output_zero_point),
        &multipliers,
        batches as usize,
        case.accum_depth as usize,
        output_depth as usize,
        &input_values,
        &filter_values,
        case.bias.then_some(&bias_values[..]),
    );
    assert_eq!(got, want, "seed {seed}");
    if check_spread && activation == Activation::None {
        assert_spread(&want);
    }
}

fn run_all(case: &Case<'_>, seed: u64) {
    for (i, activation) in ACTIVATIONS.into_iter().enumerate() {
        run(case, activation, seed * 16 + i as u64, true);
    }
}

#[test]
fn fc_per_tensor() {
    run_all(
        &Case {
            input: &[2, 10],
            output: &[2, 5],
            accum_depth: 10,
            per_channel: false,
            bias: true,
        },
        1,
    );
}

#[test]
fn fc_per_channel() {
    run_all(
        &Case {
            input: &[1, 16],
            output: &[1, 8],
            accum_depth: 16,
            per_channel: true,
            bias: true,
        },
        2,
    );
}

#[test]
fn fc_no_bias() {
    run_all(
        &Case {
            input: &[3, 7],
            output: &[3, 9],
            accum_depth: 7,
            per_channel: false,
            bias: false,
        },
        3,
    );
}

#[test]
fn fc_flattened_input() {
    // A rank-4 input flattened to [1, 12].
    run_all(
        &Case {
            input: &[1, 2, 2, 3],
            output: &[1, 7],
            accum_depth: 12,
            per_channel: true,
            bias: true,
        },
        4,
    );
}

#[test]
fn fc_keep_num_dims_as_1x1_convolution() {
    // Output rank 3 and depth multiple of 4: arm_convolve_1x1_s8_fast.
    run_all(
        &Case {
            input: &[2, 3, 8],
            output: &[2, 3, 5],
            accum_depth: 8,
            per_channel: false,
            bias: true,
        },
        5,
    );
    run_all(
        &Case {
            input: &[1, 4, 12],
            output: &[1, 4, 6],
            accum_depth: 12,
            per_channel: true,
            bias: false,
        },
        6,
    );
}

#[test]
fn fc_keep_num_dims_unaligned_depth() {
    run_all(
        &Case {
            input: &[2, 2, 6],
            output: &[2, 2, 3],
            accum_depth: 6,
            per_channel: false,
            bias: true,
        },
        7,
    );
}

#[test]
fn fc_large() {
    // Like the MLPerf Tiny anomaly-detection autoencoder layers.
    run_all(
        &Case {
            input: &[1, 640],
            output: &[1, 128],
            accum_depth: 640,
            per_channel: false,
            bias: true,
        },
        8,
    );
}

#[test]
fn fc_random_shapes() {
    let mut rng = Rng::new(9);
    for seed in 0..40 {
        let batches = rng.range(1, 4);
        let accum_depth = rng.range(1, 40);
        let output_depth = rng.range(1, 20);
        let (input, output) = if rng.range(0, 1) == 0 {
            (vec![batches, accum_depth], vec![batches, output_depth])
        } else {
            (
                vec![1, batches, accum_depth],
                vec![1, batches, output_depth],
            )
        };
        let case = Case {
            input: &input,
            output: &output,
            accum_depth,
            per_channel: rng.range(0, 1) == 0,
            bias: rng.range(0, 3) != 0,
        };
        let activation = ACTIVATIONS[rng.range(0, 3) as usize];
        run(&case, activation, 1000 + seed, false);
    }
}
