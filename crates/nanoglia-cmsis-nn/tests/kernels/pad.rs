use nanoglia_core::{BuiltinOp, DataType, OpOptions};

use crate::harness::{Op, Rng, Tensor};

/// `reference_ops::Pad`: copies the input into the padded output, filling with `pad_value`.
fn reference_pad(dims: &[i32], paddings: &[(i32, i32)], input: &[i8], pad_value: i8) -> Vec<i8> {
    let out_dims: Vec<i32> = dims
        .iter()
        .zip(paddings)
        .map(|(d, (b, a))| d + b + a)
        .collect();
    let len = out_dims.iter().product::<i32>() as usize;
    let mut output = vec![pad_value; len];
    for (i, &v) in input.iter().enumerate() {
        // Unravel the input index, shift by the leading paddings, ravel in the output.
        let mut rest = i;
        let mut coords = vec![0i32; dims.len()];
        for axis in (0..dims.len()).rev() {
            coords[axis] = (rest % dims[axis] as usize) as i32;
            rest /= dims[axis] as usize;
        }
        let mut o = 0usize;
        for axis in 0..dims.len() {
            o = o * out_dims[axis] as usize + (coords[axis] + paddings[axis].0) as usize;
        }
        output[o] = v;
    }
    output
}

fn run(dims: &[i32], paddings: &[(i32, i32)], v2: bool, i64_paddings: bool, seed: u64) {
    let mut rng = Rng::new(seed);
    let len = dims.iter().product::<i32>() as usize;
    let values = rng.i8s(len);
    let zero_point = rng.range(-128, 127);
    let out_dims: Vec<i32> = dims
        .iter()
        .zip(paddings)
        .map(|(d, (b, a))| d + b + a)
        .collect();
    let input = Tensor::activation(DataType::I8, dims, 0.05, zero_point).with_data_i8(&values);
    let output = Tensor::activation(DataType::I8, &out_dims, 0.05, zero_point);
    let flat: Vec<i64> = paddings
        .iter()
        .flat_map(|&(b, a)| [b as i64, a as i64])
        .collect();
    let paddings_tensor = if i64_paddings {
        let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
        Tensor::constant(DataType::I64, &[dims.len() as i32, 2], &bytes)
    } else {
        let bytes: Vec<u8> = flat
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect();
        Tensor::constant(DataType::I32, &[dims.len() as i32, 2], &bytes)
    };
    let constant_value = rng.range(-128, 127) as i8;
    let constant = Tensor::constant(DataType::I8, &[1], &[constant_value as u8])
        .with_zero_points(&[zero_point as i64]);
    let code = if v2 { BuiltinOp::PADV2 } else { BuiltinOp::PAD };
    let mut op = Op::new(code, OpOptions::None)
        .input(&input)
        .input(&paddings_tensor);
    if v2 {
        op = op.input(&constant);
    }
    let op = op.output(&output);
    let pad_value = if v2 { constant_value } else { zero_point as i8 };
    assert_eq!(
        op.run(),
        reference_pad(dims, paddings, &values, pad_value),
        "dims {dims:?} paddings {paddings:?}"
    );
}

#[test]
fn pad_ranks() {
    run(
        &[1, 4, 4, 3],
        &[(0, 0), (1, 1), (2, 1), (0, 0)],
        false,
        false,
        1,
    );
    run(
        &[2, 3, 5, 2],
        &[(1, 0), (0, 2), (1, 1), (1, 2)],
        false,
        true,
        2,
    );
    run(&[3, 4], &[(2, 1), (0, 3)], true, false, 3);
    run(&[7], &[(3, 2)], true, true, 4);
    run(&[2, 3, 4], &[(0, 0), (0, 0), (0, 0)], false, false, 5);
}

#[test]
fn pad_random() {
    let mut rng = Rng::new(6);
    for seed in 0..40 {
        let rank = rng.range(1, 4) as usize;
        let dims: Vec<i32> = (0..rank).map(|_| rng.range(1, 5)).collect();
        let paddings: Vec<(i32, i32)> = (0..rank)
            .map(|_| (rng.range(0, 2), rng.range(0, 2)))
            .collect();
        let v2 = rng.range(0, 1) == 0;
        let i64_paddings = rng.range(0, 1) == 0;
        run(&dims, &paddings, v2, i64_paddings, 1000 + seed);
    }
}

#[test]
fn pad_conditions() {
    let input = Tensor::activation(DataType::I8, &[1, 4], 0.1, 0);
    let output = Tensor::activation(DataType::I8, &[1, 6], 0.1, 0);
    let negative: Vec<u8> = [0i32, 0, 3, -1]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let paddings = Tensor::constant(DataType::I32, &[2, 2], &negative);
    assert!(
        !Op::new(BuiltinOp::PAD, OpOptions::None)
            .input(&input)
            .input(&paddings)
            .output(&output)
            .selected()
    );
    // Paddings computed at run time.
    let mut dynamic = Tensor::constant(DataType::I32, &[2, 2], &[0; 16]);
    dynamic.constant = false;
    assert!(
        !Op::new(BuiltinOp::PAD, OpOptions::None)
            .input(&input)
            .input(&dynamic)
            .output(&output)
            .selected()
    );
}
