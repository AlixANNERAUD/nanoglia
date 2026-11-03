use nanoglia_core::{Activation, BuiltinOp, DataType, OpOptions, Padding, Pool2DOptions};

use crate::harness::{ACTIVATIONS, Op, Rng, Tensor};
use crate::reference::{self, PoolParams};

struct Case {
    input: [i32; 4],
    filter: (i32, i32),
    stride: (i32, i32),
    padding: Padding,
}

fn run(case: &Case, average: bool, activation: Activation, seed: u64) {
    let mut rng = Rng::new(seed);
    let [batches, in_h, in_w, channels] = case.input;
    let same = case.padding == Padding::Same;
    let (out_h, pad_h) = reference::padding(same, in_h, case.filter.0, case.stride.0, 1);
    let (out_w, pad_w) = reference::padding(same, in_w, case.filter.1, case.stride.1, 1);
    let output_dims = [batches, out_h, out_w, channels];
    let scale = rng.float(0.01, 0.1);
    let zero_point = rng.range(-60, 60);
    let input_values = rng.i8s(case.input.iter().product::<i32>() as usize);

    let input = Tensor::activation(DataType::I8, &case.input, scale, zero_point)
        .with_data_i8(&input_values);
    let output = Tensor::activation(DataType::I8, &output_dims, scale, zero_point);
    let options = Pool2DOptions {
        padding: case.padding,
        stride_h: case.stride.0,
        stride_w: case.stride.1,
        filter_h: case.filter.0,
        filter_w: case.filter.1,
        activation,
    };
    let code = if average {
        BuiltinOp::AVERAGE_POOL_2D
    } else {
        BuiltinOp::MAX_POOL_2D
    };
    let op = Op::new(code, OpOptions::Pool2D(options))
        .input(&input)
        .output(&output);
    let got = op.run();

    let params = PoolParams {
        stride: case.stride,
        filter: case.filter,
        padding: (pad_h, pad_w),
        activation: reference::activation_range(activation, scale, zero_point),
    };
    let want = if average {
        reference::average_pool(&params, case.input, &input_values, output_dims)
    } else {
        reference::max_pool(&params, case.input, &input_values, output_dims)
    };
    assert_eq!(got, want, "seed {seed}, average {average}");
}

fn run_all(case: &Case, seed: u64) {
    for average in [true, false] {
        for (i, activation) in ACTIVATIONS.into_iter().enumerate() {
            run(case, average, activation, seed * 16 + i as u64);
        }
    }
}

#[test]
fn pool_2x2_valid() {
    run_all(
        &Case {
            input: [1, 8, 8, 4],
            filter: (2, 2),
            stride: (2, 2),
            padding: Padding::Valid,
        },
        1,
    );
}

#[test]
fn pool_3x3_same_partial_windows() {
    run_all(
        &Case {
            input: [2, 7, 6, 5],
            filter: (3, 3),
            stride: (2, 2),
            padding: Padding::Same,
        },
        2,
    );
}

#[test]
fn pool_global() {
    // Global average pooling, as at the end of the MLPerf Tiny convolutional models.
    run_all(
        &Case {
            input: [1, 5, 5, 64],
            filter: (5, 5),
            stride: (1, 1),
            padding: Padding::Valid,
        },
        3,
    );
}

#[test]
fn pool_overlapping_rectangular() {
    run_all(
        &Case {
            input: [1, 9, 11, 3],
            filter: (2, 4),
            stride: (1, 3),
            padding: Padding::Same,
        },
        4,
    );
}

#[test]
fn pool_random_shapes() {
    let mut rng = Rng::new(5);
    for seed in 0..60 {
        let case = Case {
            input: [
                rng.range(1, 2),
                rng.range(1, 10),
                rng.range(1, 10),
                rng.range(1, 12),
            ],
            filter: (rng.range(1, 4), rng.range(1, 4)),
            stride: (rng.range(1, 3), rng.range(1, 3)),
            padding: if rng.range(0, 1) == 0 {
                Padding::Same
            } else {
                Padding::Valid
            },
        };
        if case.padding == Padding::Valid
            && (case.filter.0 > case.input[1] || case.filter.1 > case.input[2])
        {
            continue;
        }
        let activation = ACTIVATIONS[rng.range(0, 3) as usize];
        run(&case, rng.range(0, 1) == 0, activation, 1000 + seed);
    }
}
