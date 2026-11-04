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
    let name = if add {
        "esp-nn/add_s8"
    } else {
        "esp-nn/mul_s8"
    };
    assert_eq!(op.kernel_name(), name);
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

#[test]
fn elementwise_refusals() {
    let activation = Activation::None;
    // int16 and broadcasting are left to the other backends.
    let a = Tensor::activation(DataType::I16, &[1, 2, 2, 3], 0.001, 0);
    let b = Tensor::activation(DataType::I16, &[1, 2, 2, 3], 0.002, 0);
    let o = Tensor::activation(DataType::I16, &[1, 2, 2, 3], 0.003, 0);
    for code in [BuiltinOp::ADD, BuiltinOp::MUL] {
        let op = Op::new(code, OpOptions::Elementwise { activation })
            .input(&a)
            .input(&b)
            .output(&o);
        assert!(!op.selected());
    }
    let a = Tensor::activation(DataType::I8, &[1, 2, 2, 3], 0.1, 0);
    let b = Tensor::activation(DataType::I8, &[3], 0.1, 0);
    let o = Tensor::activation(DataType::I8, &[1, 2, 2, 3], 0.2, 0);
    for code in [BuiltinOp::ADD, BuiltinOp::MUL] {
        let op = Op::new(code, OpOptions::Elementwise { activation })
            .input(&a)
            .input(&b)
            .output(&o);
        assert!(!op.selected());
    }
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
