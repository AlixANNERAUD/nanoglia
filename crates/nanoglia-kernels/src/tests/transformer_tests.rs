//! `BATCH_MATMUL`, `GELU` and `L2_NORMALIZATION`: kernel names, hand-computed results, the
//! rounding of long sums, ranks beyond TFLite's, and what the backend refuses. Golden cases cover
//! the results against LiteRT.

use super::*;
use nanoglia_core::Activation;

fn bmm_options(adj_x: bool, adj_y: bool) -> OpOptions<'static> {
    OpOptions::BatchMatMul {
        adj_x,
        adj_y,
        asymmetric_quantize_inputs: false,
    }
}

fn bmm(
    options: OpOptions<'static>,
    lhs: TensorDesc<'static>,
    rhs: TensorDesc<'static>,
    out: TensorDesc<'static>,
) -> OpDesc<'static, 'static> {
    op(
        BuiltinOp::BATCH_MATMUL,
        options,
        &[Some(lhs), Some(rhs)],
        &[out],
    )
}

fn f32s(index: u32, dims: &[i32]) -> TensorDesc<'static> {
    t(index, DataType::F32, dims)
}

/// Small integers: every product and sum is exact, so any order of rounding gives these.
fn ramp(n: usize, start: i32) -> Vec<f32> {
    (0..n as i32)
        .map(|v| ((v * 5 + start) % 7 - 3) as f32)
        .collect()
}

/// `lhs x rhs` of row-major matrices, in exact integer arithmetic.
fn matmul(lhs: &[f32], rhs: &[f32], rows: usize, depth: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = (0..depth)
                .map(|k| lhs[r * depth + k] * rhs[k * cols + c])
                .sum();
        }
    }
    out
}

/// The transpose of a row-major `rows x cols` matrix.
fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = m[r * cols + c];
        }
    }
    out
}

#[test]
fn batch_matmul_adjoints() {
    let (rows, depth, cols) = (3, 4, 2);
    let lhs = ramp(rows * depth, 1);
    let rhs = ramp(depth * cols, 4);
    let expected = matmul(&lhs, &rhs, rows, depth, cols);
    let lhs_t = transpose(&lhs, rows, depth);
    let rhs_t = transpose(&rhs, depth, cols);
    for (adj_x, adj_y) in [(false, false), (true, false), (false, true), (true, true)] {
        let (l, l_dims) = if adj_x {
            (&lhs_t, [4, 3])
        } else {
            (&lhs, [3, 4])
        };
        let (r, r_dims) = if adj_y {
            (&rhs_t, [2, 4])
        } else {
            (&rhs, [4, 2])
        };
        let op = bmm(
            bmm_options(adj_x, adj_y),
            f32s(0, &l_dims),
            f32s(1, &r_dims),
            f32s(2, &[3, 2]),
        );
        assert_eq!(select(&op).unwrap().name(), "rust/batch_matmul_f32");
        assert_bits(&run1::<f32>(&op, &[buf(l), buf(r)]), &expected);
        // A constant right-hand side gives the same.
        let op = bmm(
            bmm_options(adj_x, adj_y),
            f32s(0, &l_dims),
            konst(f32s(1, &r_dims), r),
            f32s(2, &[3, 2]),
        );
        assert_bits(&run1::<f32>(&op, &[buf(l)]), &expected);
    }
}

#[test]
fn batch_matmul_broadcasts_up_to_rank_six() {
    // [2, 1, 3, 1, 2, 3] x [1, 2, 1, 2, 3, 2]: rank 6, beyond TFLite's 5, broadcasting both ways.
    let (rows, depth, cols) = (2, 3, 2);
    let lhs = ramp(2 * 3 * rows * depth, 0);
    let rhs = ramp(2 * 2 * depth * cols, 3);
    let op = bmm(
        bmm_options(false, false),
        f32s(0, &[2, 1, 3, 1, 2, 3]),
        f32s(1, &[1, 2, 1, 2, 3, 2]),
        f32s(2, &[2, 2, 3, 2, 2, 2]),
    );
    let mut expected = Vec::new();
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..3 {
                for d in 0..2 {
                    let l = (a * 3 + c) * rows * depth;
                    let r = (b * 2 + d) * depth * cols;
                    expected.extend(matmul(
                        &lhs[l..l + rows * depth],
                        &rhs[r..r + depth * cols],
                        rows,
                        depth,
                        cols,
                    ));
                }
            }
        }
    }
    assert_bits(&run1::<f32>(&op, &[buf(&lhs), buf(&rhs)]), &expected);
}

#[test]
fn batch_matmul_rounds_by_blocks_of_eight() {
    // 1 * (1 + 2^-12)^2 - 1 summed over a depth of 9 then of 8: the ninth term is fused, the
    // eighth is not.
    let a = 1.0 + 1.0 / 4096.0;
    for (depth, expected) in [
        (9usize, 1.0 / 2048.0 + 1.0 / 16_777_216.0),
        (8, 1.0 / 2048.0),
    ] {
        let mut lhs = vec![0.0f32; depth];
        let mut rhs = vec![0.0f32; depth];
        (lhs[0], rhs[0]) = (-1.0, 1.0);
        (lhs[depth - 1], rhs[depth - 1]) = (a, a);
        let op = bmm(
            bmm_options(false, false),
            f32s(0, &[1, depth as i32]),
            f32s(1, &[depth as i32, 1]),
            f32s(2, &[1, 1]),
        );
        assert_eq!(run1::<f32>(&op, &[buf(&lhs), buf(&rhs)]), [expected]);
    }
}

#[test]
fn batch_matmul_empty_depth() {
    let op = bmm(
        bmm_options(false, false),
        f32s(0, &[2, 0]),
        f32s(1, &[0, 3]),
        f32s(2, &[2, 3]),
    );
    assert_eq!(run1::<f32>(&op, &[&[], &[]]), [0.0; 6]);
    // No rows: four billion batches of nothing, which must not be iterated.
    let op = bmm(
        bmm_options(false, false),
        f32s(0, &[65535, 65535, 0, 3]),
        f32s(1, &[3, 4]),
        f32s(2, &[65535, 65535, 0, 4]),
    );
    assert!(run1::<f32>(&op, &[&[], buf(&[0.0f32; 12])]).is_empty());
}

#[test]
fn batch_matmul_refusals() {
    let refused = |op: OpDesc<'static, 'static>| assert!(select(&op).is_none());
    let ok =
        |options, l: &[i32], r: &[i32], o: &[i32]| bmm(options, f32s(0, l), f32s(1, r), f32s(2, o));
    // The hybrid option is ignored on float inputs, as TFLite does.
    let hybrid_option = OpOptions::BatchMatMul {
        adj_x: false,
        adj_y: false,
        asymmetric_quantize_inputs: true,
    };
    assert!(select(&ok(hybrid_option, &[2, 3], &[3, 4], &[2, 4])).is_some());
    // Mismatched depth, wrong output shape, rank 1, batch dimensions that do not broadcast.
    refused(ok(bmm_options(false, false), &[2, 3], &[4, 4], &[2, 4]));
    refused(ok(bmm_options(false, false), &[2, 3], &[3, 4], &[4, 2]));
    refused(ok(bmm_options(false, false), &[2, 3], &[3, 4], &[1, 2, 4]));
    refused(ok(bmm_options(false, false), &[3], &[3, 4], &[4]));
    refused(ok(
        bmm_options(false, false),
        &[2, 2, 3],
        &[3, 3, 4],
        &[3, 2, 4],
    ));
    refused(ok(bmm_options(true, false), &[2, 3], &[3, 4], &[2, 4]));
    // Quantized, hybrid (float and int8) and integer products; missing options.
    let i8_lhs = q(t(0, DataType::I8, &[2, 3]), 0.5, 0);
    let i8_rhs = q(t(1, DataType::I8, &[3, 4]), 0.5, 0);
    let i8_out = q(t(2, DataType::I8, &[2, 4]), 0.5, 0);
    refused(bmm(bmm_options(false, false), i8_lhs, i8_rhs, i8_out));
    refused(bmm(
        bmm_options(false, false),
        f32s(0, &[2, 3]),
        i8_rhs,
        f32s(2, &[2, 4]),
    ));
    refused(bmm(
        bmm_options(false, false),
        t(0, DataType::I32, &[2, 3]),
        t(1, DataType::I32, &[3, 4]),
        t(2, DataType::I32, &[2, 4]),
    ));
    refused(ok(OpOptions::None, &[2, 3], &[3, 4], &[2, 4]));
}

fn gelu(approximate: bool, dims: &[i32]) -> OpDesc<'static, 'static> {
    op(
        BuiltinOp::GELU,
        OpOptions::Gelu { approximate },
        &[Some(f32s(0, dims))],
        &[f32s(1, dims)],
    )
}

#[test]
fn gelu_values() {
    // Inputs and LiteRT 2.3's outputs (Linux AArch64 wheel), as bit patterns: the first four go
    // through Eigen's packet code, the last three through its scalar code.
    let x = floats(&[
        0x0000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x40a0_0000,
        0xc0a0_0000,
        0x42c8_0000,
        0xc2c8_0000,
    ]);
    let cases = [
        (
            true,
            [
                0x0000_0000,
                0x3f57_585c,
                0xbe22_9e90,
                0x40a0_0000,
                0x8000_0000,
                0x42c8_0000,
                0x8000_0000,
            ],
        ),
        (
            false,
            [
                0x0000_0000,
                0x3f57_625f,
                0xbe22_7686,
                0x409f_fffd,
                0xb5c0_5e5d,
                0x42c8_0000,
                0x8000_0000,
            ],
        ),
    ];
    for (approximate, expected) in cases {
        let op = gelu(approximate, &[7]);
        assert_eq!(select(&op).unwrap().name(), "rust/gelu_f32");
        assert_bits(&run1::<f32>(&op, &[buf(&x)]), &floats(&expected));
    }
}

#[test]
fn gelu_refusals() {
    let refused = |op: OpDesc<'static, 'static>| assert!(select(&op).is_none());
    // Quantized, mismatched shapes, missing options.
    refused(op(
        BuiltinOp::GELU,
        OpOptions::Gelu { approximate: true },
        &[Some(q(t(0, DataType::I8, &[4]), 0.1, 0))],
        &[q(t(1, DataType::I8, &[4]), 0.1, 0)],
    ));
    refused(op(
        BuiltinOp::GELU,
        OpOptions::Gelu { approximate: false },
        &[Some(f32s(0, &[4]))],
        &[f32s(1, &[2, 2])],
    ));
    refused(op(
        BuiltinOp::GELU,
        OpOptions::None,
        &[Some(f32s(0, &[4]))],
        &[f32s(1, &[4])],
    ));
}

fn l2_norm(activation: Activation, dims: &[i32]) -> OpDesc<'static, 'static> {
    op(
        BuiltinOp::L2_NORMALIZATION,
        OpOptions::L2Norm { activation },
        &[Some(f32s(0, dims))],
        &[f32s(1, dims)],
    )
}

#[test]
fn l2_normalization_values() {
    let op = l2_norm(Activation::None, &[3, 2]);
    assert_eq!(select(&op).unwrap().name(), "rust/l2_normalization_f32");
    // A 3-4-5 triangle, a zero vector, and a vector whose norm is below 1e-6.
    let y = run1::<f32>(&op, &[buf(&[3.0f32, -4.0, 0.0, 0.0, 3e-7, 4e-7])]);
    assert_eq!(y[..4], [0.6, -0.8, 0.0, 0.0]);
    assert_eq!(y[4..], [3e-7 / 1e-6, 4e-7 / 1e-6]);
    // Rank 1.
    let op = l2_norm(Activation::None, &[2]);
    assert_eq!(run1::<f32>(&op, &[buf(&[0.0f32, -2.0])]), [0.0, -1.0]);
}

#[test]
fn l2_normalization_refusals() {
    let refused = |op: OpDesc<'static, 'static>| assert!(select(&op).is_none());
    // TFLite refuses fused activations; quantized types, scalars and missing options.
    refused(l2_norm(Activation::Relu, &[4]));
    refused(op(
        BuiltinOp::L2_NORMALIZATION,
        OpOptions::L2Norm {
            activation: Activation::None,
        },
        &[Some(q(t(0, DataType::I8, &[4]), 0.1, 0))],
        &[q(t(1, DataType::I8, &[4]), 1.0 / 128.0, 0)],
    ));
    refused(l2_norm(Activation::None, &[]));
    refused(op(
        BuiltinOp::L2_NORMALIZATION,
        OpOptions::None,
        &[Some(f32s(0, &[4]))],
        &[f32s(1, &[4])],
    ));
}
