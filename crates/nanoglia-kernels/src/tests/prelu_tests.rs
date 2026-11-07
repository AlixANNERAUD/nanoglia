//! `PRELU`: kernel names, a hand-computed case and what the backend refuses. Golden cases cover
//! the results.

use super::*;

fn prelu(
    x: TensorDesc<'static>,
    alpha: TensorDesc<'static>,
    y: TensorDesc<'static>,
) -> OpDesc<'static, 'static> {
    op(
        BuiltinOp::PRELU,
        OpOptions::None,
        &[Some(x), Some(alpha)],
        &[y],
    )
}

#[test]
fn prelu_kernels() {
    let f = prelu(
        t(0, DataType::F32, &[2, 2]),
        t(1, DataType::F32, &[2]),
        t(2, DataType::F32, &[2, 2]),
    );
    assert_eq!(select(&f).unwrap().name(), "rust/prelu_f32");
    let y = run1::<f32>(&f, &[buf(&[-2.0f32, 3.0, 0.0, -4.0]), buf(&[0.5f32, -1.0])]);
    assert_bits(&y, &[-1.0, 3.0, 0.0, 4.0]);

    // int8: x = 0.5 * (q + 2), alpha = 0.25 * (q - 1), y = 0.125 * (q - 3).
    let i8op = prelu(
        q(t(0, DataType::I8, &[2]), 0.5, -2),
        q(t(1, DataType::I8, &[1]), 0.25, 1),
        q(t(2, DataType::I8, &[2]), 0.125, 3),
    );
    assert_eq!(select(&i8op).unwrap().name(), "rust/prelu_q");
    // x = 1.0 -> 8 + 3; x = -2.0 with alpha 0.5 -> -1.0 -> -8 + 3.
    assert_eq!(run1::<i8>(&i8op, &[buf(&[0i8, -6]), buf(&[3i8])]), [11, -5]);
}

#[test]
fn prelu_refusals() {
    let i8t = |i, dims: &[i32], s, z| q(t(i, DataType::I8, dims), s, z);
    let refused = |op: OpDesc<'static, 'static>| assert!(select(&op).is_none());
    // The output has the input's shape: alpha may not broadcast it further.
    refused(prelu(
        t(0, DataType::F32, &[2]),
        t(1, DataType::F32, &[3, 2]),
        t(2, DataType::F32, &[3, 2]),
    ));
    // int16 needs zero points 0 and an int8 alpha; uint8 a uint8 alpha.
    let i16t = |i, z| q(t(i, DataType::I16, &[2]), 0.01, z);
    refused(prelu(i16t(0, 1), i8t(1, &[2], 0.1, 0), i16t(2, 0)));
    refused(prelu(i16t(0, 0), i16t(1, 0), i16t(2, 0)));
    refused(prelu(
        q(t(0, DataType::U8, &[2]), 0.1, 128),
        i8t(1, &[2], 0.1, 0),
        q(t(2, DataType::U8, &[2]), 0.1, 128),
    ));
    // Zero points outside the type, multipliers shifting left by more than 30 bits.
    refused(prelu(
        i8t(0, &[2], 0.1, 300),
        i8t(1, &[2], 0.1, 0),
        i8t(2, &[2], 0.1, 0),
    ));
    refused(prelu(
        i8t(0, &[2], 1.0e10, 0),
        i8t(1, &[2], 0.1, 0),
        i8t(2, &[2], 1.0e-10, 0),
    ));
    // Mixed float and quantized tensors.
    refused(prelu(
        t(0, DataType::F32, &[2]),
        i8t(1, &[2], 0.1, 0),
        t(2, DataType::F32, &[2]),
    ));
}
