//! 16x8 `DEPTHWISE_CONV_2D` (int16 activations with zero point 0, int8 weights, int64 bias) and
//! int16 `AVERAGE_POOL_2D` / `MAX_POOL_2D`, against ports of the TFLite reference kernels.

use nanoglia_core::{
    Activation, BuiltinOp, DataType, DepthwiseConv2DOptions, OpOptions, Padding, Pool2DOptions,
};

use crate::harness::{ACTIVATIONS, Op, Rng, Tensor};
use crate::reference::{self, ConvParams, PoolParams};

/// A 16x8 depthwise convolution test case: NHWC input, `(kh, kw)` filter.
struct DepthwiseCase {
    input: [i32; 4],
    kernel: (i32, i32),
    depth_multiplier: i32,
    stride: (i32, i32),
    dilation: (i32, i32),
    padding: Padding,
    per_channel: bool,
    bias: bool,
}

fn depthwise_options(case: &DepthwiseCase, activation: Activation) -> OpOptions<'static> {
    OpOptions::DepthwiseConv2D(DepthwiseConv2DOptions {
        padding: case.padding,
        stride_h: case.stride.0,
        stride_w: case.stride.1,
        depth_multiplier: case.depth_multiplier,
        dilation_h: case.dilation.0,
        dilation_w: case.dilation.1,
        activation,
    })
}

fn run_depthwise(
    case: &DepthwiseCase,
    activation: Activation,
    seed: u64,
    bias_range: i64,
    check_spread: bool,
) {
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

    let input_scale = rng.float(1e-4, 1e-3);
    let channels = if case.per_channel { out_c as usize } else { 1 };
    let filter_scales: Vec<f32> = (0..channels).map(|_| rng.float(0.002, 0.02)).collect();
    let max_filter_scale = filter_scales.iter().copied().fold(0.0, f32::max);
    let depth = (case.kernel.0 * case.kernel.1) as f32;
    let output_scale = input_scale * max_filter_scale * depth.sqrt() * 80.0 * rng.float(0.5, 1.5);

    let input_values = rng.i16s(case.input.iter().product::<i32>() as usize);
    let filter_values = rng.i8s(filter_dims.iter().product::<i32>() as usize);
    let bias_values: Vec<i64> = (0..out_c)
        .map(|_| {
            let r = rng.range(-1_000_000, 1_000_000) as i64;
            r * bias_range / 1_000_000
        })
        .collect();

    let input =
        Tensor::activation(DataType::I16, &case.input, input_scale, 0).with_data_i16(&input_values);
    let filter = Tensor::weights(&filter_dims, &filter_values, &filter_scales, 3);
    let bias = Tensor::bias_i64(&bias_values);
    let output = Tensor::activation(DataType::I16, &output_dims, output_scale, 0);
    let op = Op::new(
        BuiltinOp::DEPTHWISE_CONV_2D,
        depthwise_options(case, activation),
    )
    .input(&input)
    .input(&filter)
    .optional(case.bias.then_some(&bias))
    .output(&output);
    assert_eq!(op.kernel_name(), "cmsis-nn/depthwise_conv2d_s16");

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
    let want = reference::depthwise_conv_per_channel_16x8(
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
    let got = op.run_i16();
    assert_eq!(got, want, "seed {seed}");
    if check_spread && activation == Activation::None {
        let mut distinct = want.clone();
        distinct.sort();
        distinct.dedup();
        assert!(distinct.len() > want.len().min(8) / 2, "degenerate output");
    }
}

fn run_depthwise_all(case: &DepthwiseCase, seed: u64) {
    for (i, activation) in ACTIVATIONS.into_iter().enumerate() {
        run_depthwise(case, activation, seed * 16 + i as u64, 2_000_000, true);
    }
}

#[test]
fn depthwise_16x8_fast_path_shapes() {
    // Depth multiplier 1, no dilation, fewer than 512 filter values: arm_depthwise_conv_fast_s16
    // with the DSP or Helium extension (its generic code on the host). 8 channels (blocks of 4)
    // and 5 (a leftover channel), an odd number of filter values, padding on every side.
    run_depthwise_all(
        &DepthwiseCase {
            input: [1, 8, 7, 8],
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
    run_depthwise_all(
        &DepthwiseCase {
            input: [2, 9, 9, 5],
            kernel: (3, 2),
            depth_multiplier: 1,
            stride: (2, 2),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: false,
            bias: false,
        },
        2,
    );
}

#[test]
fn depthwise_16x8_generic_path_shapes() {
    // arm_depthwise_conv_s16: depth multipliers above 1, dilation, and a 23x23 filter (529
    // values per channel, too many for the fast kernel).
    run_depthwise_all(
        &DepthwiseCase {
            input: [1, 8, 7, 3],
            kernel: (3, 2),
            depth_multiplier: 2,
            stride: (2, 1),
            dilation: (1, 1),
            padding: Padding::Valid,
            per_channel: true,
            bias: true,
        },
        3,
    );
    run_depthwise_all(
        &DepthwiseCase {
            input: [1, 9, 9, 4],
            kernel: (3, 3),
            depth_multiplier: 1,
            stride: (1, 1),
            dilation: (2, 2),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        4,
    );
    run_depthwise_all(
        &DepthwiseCase {
            input: [1, 24, 25, 2],
            kernel: (23, 23),
            depth_multiplier: 1,
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: true,
            bias: true,
        },
        5,
    );
    run_depthwise_all(
        &DepthwiseCase {
            input: [1, 6, 6, 1],
            kernel: (3, 3),
            depth_multiplier: 8,
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding::Same,
            per_channel: false,
            bias: false,
        },
        6,
    );
}

#[test]
fn depthwise_16x8_large_bias() {
    // Biases of up to 2^36, still within the accumulator bound of the kernel: the outputs
    // saturate, through CMSIS-NN's 32-bit intermediate value.
    let case = DepthwiseCase {
        input: [1, 5, 5, 4],
        kernel: (3, 3),
        depth_multiplier: 1,
        stride: (1, 1),
        dilation: (1, 1),
        padding: Padding::Same,
        per_channel: true,
        bias: true,
    };
    run_depthwise(&case, Activation::None, 7, 1 << 36, false);
}

#[test]
fn depthwise_16x8_random_shapes() {
    let mut rng = Rng::new(9);
    for seed in 0..40 {
        let case = DepthwiseCase {
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
        // Small images give few distinct outputs: no spread check.
        run_depthwise(&case, activation, 2000 + seed, 2_000_000, false);
    }
}

#[test]
fn depthwise_16x8_refusals() {
    let input = Tensor::activation(DataType::I16, &[1, 4, 4, 2], 0.001, 0);
    let filter = Tensor::weights(&[1, 3, 3, 2], &[1; 18], &[0.01, 0.02], 3);
    let output = Tensor::activation(DataType::I16, &[1, 4, 4, 2], 0.002, 0);
    let case = DepthwiseCase {
        input: [1, 4, 4, 2],
        kernel: (3, 3),
        depth_multiplier: 1,
        stride: (1, 1),
        dilation: (1, 1),
        padding: Padding::Same,
        per_channel: true,
        bias: true,
    };
    let selected = |input: &Tensor, filter: &Tensor, bias: Option<&Tensor>, output: &Tensor| {
        Op::new(
            BuiltinOp::DEPTHWISE_CONV_2D,
            depthwise_options(&case, Activation::None),
        )
        .input(input)
        .input(filter)
        .optional(bias)
        .output(output)
        .selected()
    };
    let bias = Tensor::bias_i64(&[100, -100]);
    assert!(selected(&input, &filter, None, &output));
    assert!(selected(&input, &filter, Some(&bias), &output));
    // int32 bias.
    assert!(!selected(
        &input,
        &filter,
        Some(&Tensor::bias(&[1, 2])),
        &output
    ));
    // Nonzero zero points.
    let shifted = Tensor::activation(DataType::I16, &[1, 4, 4, 2], 0.001, 3);
    assert!(!selected(&shifted, &filter, None, &output));
    let shifted_output = Tensor::activation(DataType::I16, &[1, 4, 4, 2], 0.002, -1);
    assert!(!selected(&input, &filter, None, &shifted_output));
    // A bias computed at run time cannot be bounded.
    let mut runtime_bias = Tensor::bias_i64(&[100, -100]);
    runtime_bias.constant = false;
    assert!(!selected(&input, &filter, Some(&runtime_bias), &output));
    // A bias so large that CMSIS-NN's 32-bit intermediate would overflow where TFLite's 64-bit
    // one does not.
    let huge_bias = Tensor::bias_i64(&[1 << 50, 0]);
    assert!(!selected(&input, &filter, Some(&huge_bias), &output));
    // A requantization shift of 8 or more.
    let tiny_output = Tensor::activation(DataType::I16, &[1, 4, 4, 2], 1e-8, 0);
    assert!(!selected(&input, &filter, None, &tiny_output));
    // A window origin beyond the int16 range of CMSIS-NN.
    let wide_input = Tensor::activation(DataType::I16, &[1, 1, 40000, 1], 0.001, 0);
    let wide_output = Tensor::activation(DataType::I16, &[1, 1, 40000, 1], 0.002, 0);
    let point = Tensor::weights(&[1, 1, 1, 1], &[1], &[0.01], 3);
    let pointwise = |input: &Tensor, output: &Tensor| {
        Op::new(
            BuiltinOp::DEPTHWISE_CONV_2D,
            OpOptions::DepthwiseConv2D(DepthwiseConv2DOptions {
                padding: Padding::Valid,
                stride_h: 1,
                stride_w: 1,
                depth_multiplier: 1,
                dilation_h: 1,
                dilation_w: 1,
                activation: Activation::None,
            }),
        )
        .input(input)
        .input(&point)
        .output(output)
        .selected()
    };
    assert!(!pointwise(&wide_input, &wide_output));
    let ok_input = Tensor::activation(DataType::I16, &[1, 1, 32768, 1], 0.001, 0);
    let ok_output = Tensor::activation(DataType::I16, &[1, 1, 32768, 1], 0.002, 0);
    assert!(pointwise(&ok_input, &ok_output));
}

/// An int16 pooling test case.
struct PoolCase {
    input: [i32; 4],
    filter: (i32, i32),
    stride: (i32, i32),
    padding: Padding,
}

fn pool_op<'t>(
    case: &PoolCase,
    average: bool,
    activation: Activation,
    input: &'t Tensor,
    output: &'t Tensor,
) -> Op<'t> {
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
    Op::new(code, OpOptions::Pool2D(options))
        .input(input)
        .output(output)
}

fn run_pool(case: &PoolCase, average: bool, activation: Activation, seed: u64) {
    let mut rng = Rng::new(seed);
    let [batches, in_h, in_w, channels] = case.input;
    let same = case.padding == Padding::Same;
    let (out_h, pad_h) = reference::padding(same, in_h, case.filter.0, case.stride.0, 1);
    let (out_w, pad_w) = reference::padding(same, in_w, case.filter.1, case.stride.1, 1);
    let output_dims = [batches, out_h, out_w, channels];
    let scale = rng.float(1e-4, 1e-3);
    let input_values = rng.i16s(case.input.iter().product::<i32>() as usize);
    let input =
        Tensor::activation(DataType::I16, &case.input, scale, 0).with_data_i16(&input_values);
    let output = Tensor::activation(DataType::I16, &output_dims, scale, 0);
    let op = pool_op(case, average, activation, &input, &output);
    let name = if average {
        "cmsis-nn/average_pool2d_s16"
    } else {
        "cmsis-nn/max_pool2d_s16"
    };
    assert_eq!(op.kernel_name(), name);
    let params = PoolParams {
        stride: case.stride,
        filter: case.filter,
        padding: (pad_h, pad_w),
        activation: reference::activation_range_i16(activation, scale),
    };
    let want = reference::pool_i16(&params, average, case.input, &input_values, output_dims);
    assert_eq!(op.run_i16(), want, "seed {seed}, average {average}");
}

fn run_pool_all(case: &PoolCase, seed: u64) {
    for average in [true, false] {
        for (i, activation) in ACTIVATIONS.into_iter().enumerate() {
            run_pool(case, average, activation, seed * 16 + i as u64);
        }
    }
}

#[test]
fn pool_s16_shapes() {
    run_pool_all(
        &PoolCase {
            input: [1, 8, 8, 4],
            filter: (2, 2),
            stride: (2, 2),
            padding: Padding::Valid,
        },
        1,
    );
    run_pool_all(
        &PoolCase {
            input: [2, 7, 6, 5],
            filter: (3, 3),
            stride: (2, 2),
            padding: Padding::Same,
        },
        2,
    );
    run_pool_all(
        &PoolCase {
            input: [1, 9, 11, 3],
            filter: (2, 4),
            stride: (1, 3),
            padding: Padding::Same,
        },
        3,
    );
    // Global pooling.
    run_pool_all(
        &PoolCase {
            input: [1, 5, 5, 64],
            filter: (5, 5),
            stride: (1, 1),
            padding: Padding::Valid,
        },
        4,
    );
}

#[test]
fn pool_s16_extreme_values() {
    // Windows of -32768 only and of 32767 only: the sums and their rounding stay in 32 bits.
    let case = PoolCase {
        input: [1, 16, 16, 2],
        filter: (16, 16),
        stride: (1, 1),
        padding: Padding::Valid,
    };
    let values: Vec<i16> = (0..512)
        .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
        .collect();
    let input = Tensor::activation(DataType::I16, &case.input, 0.001, 0).with_data_i16(&values);
    let output = Tensor::activation(DataType::I16, &[1, 1, 1, 2], 0.001, 0);
    for average in [true, false] {
        let op = pool_op(&case, average, Activation::None, &input, &output);
        assert_eq!(op.run_i16(), vec![i16::MIN, i16::MAX], "average {average}");
    }
}

#[test]
fn pool_s16_random_shapes() {
    let mut rng = Rng::new(11);
    for seed in 0..60 {
        let case = PoolCase {
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
        run_pool(&case, rng.range(0, 1) == 0, activation, 3000 + seed);
    }
}

#[test]
fn pool_s16_refusals() {
    let case = PoolCase {
        input: [1, 4, 4, 2],
        filter: (2, 2),
        stride: (2, 2),
        padding: Padding::Valid,
    };
    let input = Tensor::activation(DataType::I16, &[1, 4, 4, 2], 0.001, 0);
    let output = Tensor::activation(DataType::I16, &[1, 2, 2, 2], 0.001, 0);
    let selected = |average, input: &Tensor, output: &Tensor| {
        pool_op(&case, average, Activation::None, input, output).selected()
    };
    for average in [true, false] {
        assert!(selected(average, &input, &output));
        // Nonzero zero points (shared by the input and the output).
        let shifted_input = Tensor::activation(DataType::I16, &[1, 4, 4, 2], 0.001, 5);
        let shifted_output = Tensor::activation(DataType::I16, &[1, 2, 2, 2], 0.001, 5);
        assert!(!selected(average, &shifted_input, &shifted_output));
        // Different scales.
        let rescaled = Tensor::activation(DataType::I16, &[1, 2, 2, 2], 0.002, 0);
        assert!(!selected(average, &input, &rescaled));
        // An int8 output.
        let int8 = Tensor::activation(DataType::I8, &[1, 2, 2, 2], 0.001, 0);
        assert!(!selected(average, &input, &int8));
    }
    // 65536 values in an average window could overflow the 32-bit sum; max pooling is fine.
    let big = PoolCase {
        input: [1, 256, 256, 1],
        filter: (256, 256),
        stride: (1, 1),
        padding: Padding::Valid,
    };
    let big_input = Tensor::activation(DataType::I16, &big.input, 0.001, 0);
    let big_output = Tensor::activation(DataType::I16, &[1, 1, 1, 1], 0.001, 0);
    assert!(!pool_op(&big, true, Activation::None, &big_input, &big_output).selected());
    assert!(pool_op(&big, false, Activation::None, &big_input, &big_output).selected());
    let largest = PoolCase {
        input: [1, 255, 257, 1],
        filter: (255, 257),
        stride: (1, 1),
        padding: Padding::Valid,
    };
    let largest_input = Tensor::activation(DataType::I16, &largest.input, 0.001, 0);
    assert!(
        pool_op(
            &largest,
            true,
            Activation::None,
            &largest_input,
            &big_output
        )
        .selected()
    );
}
