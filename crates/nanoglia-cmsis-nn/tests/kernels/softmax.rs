use nanoglia_core::{BuiltinOp, DataType, OpOptions};

use crate::harness::{Op, Rng, Tensor};
use crate::reference;

fn run(dims: &[i32], beta: f32, input_scale: f32, seed: u64) {
    let mut rng = Rng::new(seed);
    let len = dims.iter().product::<i32>() as usize;
    let depth = *dims.last().unwrap() as usize;
    let input_zero_point = rng.range(-128, 127);
    let values = rng.i8s(len);
    let input =
        Tensor::activation(DataType::I8, dims, input_scale, input_zero_point).with_data_i8(&values);
    let output = Tensor::activation(DataType::I8, dims, 1.0 / 256.0, -128);
    let op = Op::new(BuiltinOp::SOFTMAX, OpOptions::Softmax { beta })
        .input(&input)
        .output(&output);
    let got = op.run();
    let want = reference::softmax(beta, input_scale, &values, depth);
    assert_eq!(got, want, "dims {dims:?} beta {beta} scale {input_scale}");
}

#[test]
fn softmax_rows() {
    run(&[1, 10], 1.0, 0.1, 1);
    run(&[3, 7], 1.0, 0.05, 2);
    run(&[2, 2, 5], 0.5, 1.0 / 16.0, 3);
    run(&[1, 12], 2.0, 0.2, 4);
    // MLPerf Tiny keyword spotting: 12 classes.
    run(&[1, 12], 1.0, 0.087_169_37, 5);
}

#[test]
fn softmax_wide_and_flat_rows() {
    run(&[4, 100], 1.0, 0.03, 6);
    run(&[1, 1], 1.0, 0.1, 7);
    // Tiny scale: most differences stay above diff_min.
    run(&[2, 33], 1.0, 0.004, 8);
    // Large scale: most differences fall below diff_min.
    run(&[2, 33], 1.0, 0.9, 9);
}

#[test]
fn softmax_random() {
    let mut rng = Rng::new(10);
    for seed in 0..50 {
        let rows = rng.range(1, 4);
        let depth = rng.range(1, 40);
        let beta = rng.float(0.2, 3.0);
        let scale = rng.float(0.005, 0.5);
        run(&[rows, depth], beta, scale, 1000 + seed);
    }
}
