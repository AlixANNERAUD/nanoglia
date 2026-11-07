//! Comparisons, `SELECT_V2` and `CAST`: kernel names, TensorFlow Lite Micro's float-to-integer
//! conversion of special values, and what the backend refuses. Golden cases cover the results.

use super::*;

fn cast(x: TensorDesc<'static>, y: TensorDesc<'static>) -> OpDesc<'static, 'static> {
    op(BuiltinOp::CAST, OpOptions::None, &[Some(x)], &[y])
}

#[test]
fn cast_saturates_like_tensorflow_lite_micro() {
    let op = cast(t(0, DataType::F32, &[6]), t(1, DataType::I8, &[6]));
    assert_eq!(select(&op).unwrap().name(), "rust/cast");
    let x = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -128.9,
        127.9,
        -0.7,
    ];
    assert_eq!(run1::<i8>(&op, &[buf(&x)]), [0, 127, -128, -128, 127, 0]);
    let to_u32 = cast(t(0, DataType::F32, &[3]), t(1, DataType::U32, &[3]));
    assert_eq!(
        run1::<u32>(&to_u32, &[buf(&[-5.0f32, 4.0e9, 1.0e10])]),
        [0, 4_000_000_000, u32::MAX]
    );
    // Integers keep their low bits; bools are 0 or 1; NaN is true.
    let narrow = cast(t(0, DataType::I64, &[2]), t(1, DataType::U8, &[2]));
    assert_eq!(
        run1::<u8>(&narrow, &[buf(&[-1i64, 0x1_0000_0102])]),
        [255, 2]
    );
    let to_bool = cast(t(0, DataType::F32, &[3]), t(1, DataType::Bool, &[3]));
    assert_eq!(
        run1::<u8>(&to_bool, &[buf(&[0.0f32, -0.0, f32::NAN])]),
        [0, 0, 1]
    );
}

#[test]
fn logic_refusals() {
    let refused = |op: OpDesc<'static, 'static>| assert!(select(&op).is_none());
    let cmp = |code, a: TensorDesc<'static>, b: TensorDesc<'static>, dims: &[i32]| {
        op(
            code,
            OpOptions::None,
            &[Some(a), Some(b)],
            &[t(2, DataType::Bool, dims)],
        )
    };
    let f = cmp(
        BuiltinOp::LESS,
        t(0, DataType::F32, &[2]),
        t(1, DataType::F32, &[2]),
        &[2],
    );
    assert_eq!(select(&f).unwrap().name(), "rust/comparison");
    // int16 (TFLite's support is inconsistent), bool ordering, scales of 1 or more, mixed
    // types, a non-bool output.
    refused(cmp(
        BuiltinOp::GREATER,
        q(t(0, DataType::I16, &[2]), 0.5, 0),
        q(t(1, DataType::I16, &[2]), 0.5, 0),
        &[2],
    ));
    refused(cmp(
        BuiltinOp::LESS,
        t(0, DataType::Bool, &[2]),
        t(1, DataType::Bool, &[2]),
        &[2],
    ));
    refused(cmp(
        BuiltinOp::EQUAL,
        q(t(0, DataType::I8, &[2]), 1.5, 0),
        q(t(1, DataType::I8, &[2]), 0.5, 0),
        &[2],
    ));
    refused(cmp(
        BuiltinOp::EQUAL,
        t(0, DataType::F32, &[2]),
        t(1, DataType::I32, &[2]),
        &[2],
    ));
    refused(op(
        BuiltinOp::EQUAL,
        OpOptions::None,
        &[
            Some(t(0, DataType::F32, &[2])),
            Some(t(1, DataType::F32, &[2])),
        ],
        &[t(2, DataType::I8, &[2])],
    ));

    let select_op = |c: &[i32], x: &[i32], y: &[i32], out: &[i32]| {
        op(
            BuiltinOp::SELECT_V2,
            OpOptions::None,
            &[
                Some(t(0, DataType::Bool, c)),
                Some(t(1, DataType::F32, x)),
                Some(t(2, DataType::F32, y)),
            ],
            &[t(3, DataType::F32, out)],
        )
    };
    let ok = select_op(&[2, 1], &[3], &[1], &[2, 3]);
    assert_eq!(select(&ok).unwrap().name(), "rust/select_v2");
    let y = run1::<f32>(
        &ok,
        &[buf(&[1u8, 0]), buf(&[1.0f32, 2.0, 3.0]), buf(&[9.0f32])],
    );
    assert_bits(&y, &[1.0, 2.0, 3.0, 9.0, 9.0, 9.0]);
    // The output must be the broadcast shape of the three inputs.
    refused(select_op(&[2, 1], &[3], &[1], &[4, 2, 3]));
    refused(select_op(&[2, 1], &[3], &[1], &[2, 4]));
    refused(select_op(&[2], &[3], &[1], &[2, 3]));

    // CAST of types TFLite Micro does not have.
    refused(cast(t(0, DataType::F16, &[2]), t(1, DataType::F32, &[2])));
    refused(cast(t(0, DataType::F32, &[2]), t(1, DataType::I32, &[3])));
}
