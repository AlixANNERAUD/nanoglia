use nanoglia_core::{Activation, BuiltinOp, DataType, OpOptions};

use crate::harness::{ACTIVATIONS, Op, Rng, Tensor, assert_spread};
use crate::reference;

fn run(dims: &[i32], add: bool, activation: Activation, seed: u64) {
    let mut rng = Rng::new(seed);
    let len = dims.iter().product::<i32>() as usize;
    let (s1, z1) = (rng.float(0.01, 0.2), rng.range(-128, 127));
    let (s2, z2) = (rng.float(0.01, 0.2), rng.range(-128, 127));
    let so = if add {
        (s1 + s2) * rng.float(0.8, 1.6)
    } else {
        s1 * s2 * 128.0 * rng.float(0.3, 1.5)
    };
    let zo = rng.range(-60, 60);
    let a = rng.i8s(len);
    let b = rng.i8s(len);
    let input1 = Tensor::activation(DataType::I8, dims, s1, z1).with_data_i8(&a);
    let input2 = Tensor::activation(DataType::I8, dims, s2, z2).with_data_i8(&b);
    let output = Tensor::activation(DataType::I8, dims, so, zo);
    let code = if add { BuiltinOp::ADD } else { BuiltinOp::MUL };
    let op = Op::new(code, OpOptions::Elementwise { activation })
        .input(&input1)
        .input(&input2)
        .output(&output);
    let got = op.run();
    let want = if add {
        let p = reference::add_params(s1, z1, s2, z2, so, zo, activation);
        reference::add(&p, &a, &b)
    } else {
        reference::mul(s1, z1, s2, z2, so, zo, activation, &a, &b)
    };
    assert_eq!(got, want, "seed {seed}, add {add}");
    if activation == Activation::None && len >= 16 {
        assert_spread(&want);
    }
}

#[test]
fn add_shapes() {
    for (i, dims) in [&[1, 4, 4, 8][..], &[37], &[2, 3, 5], &[1, 1, 1, 1]]
        .into_iter()
        .enumerate()
    {
        for (j, activation) in ACTIVATIONS.into_iter().enumerate() {
            run(dims, true, activation, (i * 16 + j) as u64);
        }
    }
}

#[test]
fn mul_shapes() {
    for (i, dims) in [&[1, 4, 4, 8][..], &[37], &[2, 3, 5], &[1, 1, 1, 1]]
        .into_iter()
        .enumerate()
    {
        for (j, activation) in ACTIVATIONS.into_iter().enumerate() {
            run(dims, false, activation, 100 + (i * 16 + j) as u64);
        }
    }
}

fn run_i16(dims: &[i32], add: bool, activation: Activation, seed: u64) {
    let mut rng = Rng::new(seed);
    let len = dims.iter().product::<i32>() as usize;
    let (s1, s2) = (rng.float(1e-5, 1e-3), rng.float(1e-5, 1e-3));
    let so = if add {
        (s1 + s2) * rng.float(0.5, 1.2)
    } else {
        s1 * s2 * 32768.0 * rng.float(0.2, 1.2)
    };
    let (a, b) = (rng.i16s(len), rng.i16s(len));
    let input1 = Tensor::activation(DataType::I16, dims, s1, 0).with_data_i16(&a);
    let input2 = Tensor::activation(DataType::I16, dims, s2, 0).with_data_i16(&b);
    let output = Tensor::activation(DataType::I16, dims, so, 0);
    let code = if add { BuiltinOp::ADD } else { BuiltinOp::MUL };
    let op = Op::new(code, OpOptions::Elementwise { activation })
        .input(&input1)
        .input(&input2)
        .output(&output);
    let name = if add {
        "cmsis-nn/add_s16"
    } else {
        "cmsis-nn/mul_s16"
    };
    assert_eq!(op.kernel_name(), name);
    let want = if add {
        reference::add_i16((s1, s2, so), activation, &a, &b)
    } else {
        reference::mul_i16((s1, s2, so), activation, &a, &b)
    };
    assert_eq!(op.run_i16(), want, "seed {seed}, add {add}");
}

#[test]
fn elementwise_i16() {
    let mut rng = Rng::new(7);
    for seed in 0..40 {
        let dims = [
            rng.range(1, 2),
            rng.range(1, 9),
            rng.range(1, 9),
            rng.range(1, 9),
        ];
        let activation = ACTIVATIONS[rng.range(0, 3) as usize];
        run_i16(&dims, seed % 2 == 0, activation, 2000 + seed);
    }
}

#[test]
fn elementwise_i16_refusals() {
    let dims = [1, 2, 2, 3];
    let refused = |code, q: [(f32, i32); 3]| {
        let input1 = Tensor::activation(DataType::I16, &dims, q[0].0, q[0].1);
        let input2 = Tensor::activation(DataType::I16, &dims, q[1].0, q[1].1);
        let output = Tensor::activation(DataType::I16, &dims, q[2].0, q[2].1);
        !Op::new(
            code,
            OpOptions::Elementwise {
                activation: Activation::None,
            },
        )
        .input(&input1)
        .input(&input2)
        .output(&output)
        .selected()
    };
    // Nonzero zero points (CMSIS-NN ignores the int16 offsets).
    for code in [BuiltinOp::ADD, BuiltinOp::MUL] {
        assert!(refused(code, [(0.001, 3), (0.002, 0), (0.003, 0)]));
        assert!(refused(code, [(0.001, 0), (0.002, 0), (0.003, -1)]));
        assert!(!refused(code, [(0.001, 0), (0.002, 0), (0.003, 0)]));
    }
    // Power-of-two scales: TFLite may run its `pot_scale_int16` kernel.
    let pot = [(1.0 / 1024.0, 0), (1.0 / 2048.0, 0), (1.0 / 512.0, 0)];
    assert!(refused(BuiltinOp::ADD, pot));
    assert!(!refused(BuiltinOp::MUL, pot));
}

#[test]
fn elementwise_random() {
    let mut rng = Rng::new(3);
    for seed in 0..60 {
        let dims = [
            rng.range(1, 3),
            rng.range(1, 9),
            rng.range(1, 9),
            rng.range(1, 9),
        ];
        let activation = ACTIVATIONS[rng.range(0, 3) as usize];
        run(&dims, rng.range(0, 1) == 0, activation, 1000 + seed);
    }
}
