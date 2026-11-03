use nanoglia_core::{BuiltinOp, DataType, OpOptions};

use crate::harness::{Op, Rng, Tensor};

/// `reference_ops::MaximumMinimumBroadcastSlow` for equal shapes or a single-value operand.
fn reference(a: &[i8], b: &[i8], maximum: bool) -> Vec<i8> {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| {
            let x = a[if a.len() == 1 { 0 } else { i }];
            let y = b[if b.len() == 1 { 0 } else { i }];
            if maximum { x.max(y) } else { x.min(y) }
        })
        .collect()
}

fn run(dims1: &[i32], dims2: &[i32], out: &[i32], maximum: bool, seed: u64) {
    let mut rng = Rng::new(seed);
    let a = rng.i8s(dims1.iter().product::<i32>() as usize);
    let b = rng.i8s(dims2.iter().product::<i32>() as usize);
    let input1 = Tensor::activation(DataType::I8, dims1, 0.1, 3).with_data_i8(&a);
    let input2 = Tensor::activation(DataType::I8, dims2, 0.1, 3).with_data_i8(&b);
    let output = Tensor::activation(DataType::I8, out, 0.1, 3);
    let code = if maximum {
        BuiltinOp::MAXIMUM
    } else {
        BuiltinOp::MINIMUM
    };
    let op = Op::new(code, OpOptions::None)
        .input(&input1)
        .input(&input2)
        .output(&output);
    assert_eq!(op.run(), reference(&a, &b, maximum), "{dims1:?} {dims2:?}");
}

#[test]
fn minmax_same_shape_and_scalar() {
    for (i, maximum) in [true, false].into_iter().enumerate() {
        let seed = 10 * i as u64;
        run(&[1, 4, 4, 8], &[1, 4, 4, 8], &[1, 4, 4, 8], maximum, seed);
        run(&[37], &[37], &[37], maximum, seed + 1);
        run(&[2, 3, 5], &[1], &[2, 3, 5], maximum, seed + 2);
        run(&[1, 1, 1], &[3, 1, 6], &[3, 1, 6], maximum, seed + 3);
        run(&[1], &[1], &[1], maximum, seed + 4);
    }
}

#[test]
fn minmax_general_broadcast_is_left_to_other_backends() {
    let a = Tensor::activation(DataType::I8, &[2, 3], 0.1, 0);
    let b = Tensor::activation(DataType::I8, &[1, 3], 0.1, 0);
    let out = Tensor::activation(DataType::I8, &[2, 3], 0.1, 0);
    assert!(
        !Op::new(BuiltinOp::MAXIMUM, OpOptions::None)
            .input(&a)
            .input(&b)
            .output(&out)
            .selected()
    );
}
