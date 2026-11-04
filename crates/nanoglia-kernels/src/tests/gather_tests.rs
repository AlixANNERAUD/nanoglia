//! `GATHER`: vectors from the TFLite kernel tests (`gather_test.cc`), run-time index errors, and
//! what the backend refuses.

use super::*;
use nanoglia_core::ErrorCode;

fn gather_op(
    shape: &[i32],
    indices: TensorDesc<'static>,
    axis: i32,
    batch_dims: i32,
    out: &[i32],
) -> OpDesc<'static, 'static> {
    op(
        BuiltinOp::GATHER,
        OpOptions::Gather { axis, batch_dims },
        &[Some(t(0, DataType::F32, shape)), Some(indices)],
        &[t(2, DataType::F32, out)],
    )
}

#[test]
fn gather_vectors_and_errors() {
    let x: Vec<f32> = (0..12).map(|v| v as f32).collect();
    // gather_test.cc Shuffle (axis 0) and Axis1 with int32, int64 and int16 indices.
    let i32s = konst(t(1, DataType::I32, &[2]), &[1i32, 0]);
    let op0 = gather_op(&[2, 6], i32s, 0, 0, &[2, 6]);
    assert_eq!(
        run1::<f32>(&op0, &[buf(&x)]),
        [6., 7., 8., 9., 10., 11., 0., 1., 2., 3., 4., 5.]
    );
    let i64s = t(1, DataType::I64, &[3]);
    let op1 = gather_op(&[3, 4], i64s, -1, 0, &[3, 3]);
    assert_eq!(
        run1::<f32>(&op1, &[buf(&x), buf(&[3i64, 0, 3])]),
        [3., 0., 3., 7., 4., 7., 11., 8., 11.]
    );
    let i16s = t(1, DataType::I16, &[2, 1]);
    let op2 = gather_op(&[2, 2, 3], i16s, 1, 1, &[2, 1, 3]);
    assert_eq!(
        run1::<f32>(&op2, &[buf(&x), buf(&[1i16, 0])]),
        [3., 4., 5., 6., 7., 8.]
    );

    // Indices outside the axis fail at evaluation, reporting their position.
    for bad in [[0i32, 3], [-1, 0]] {
        let op3 = gather_op(&[3, 4], t(1, DataType::I32, &[2]), 0, 0, &[2, 4]);
        let err = run(&op3, &[buf(&x), buf(&bad)]).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidArgument);
        let position = if bad[0] < 0 { 0 } else { 1 };
        assert_eq!(err.detail, Detail::IndexOutOfRange { index: position });
    }

    // Refused: batch_dims above the axis, wrong output shape or type, int8 indices, no options.
    let idx = || t(1, DataType::I32, &[2]);
    assert!(select(&gather_op(&[2, 3], idx(), 0, 1, &[2, 3])).is_none());
    assert!(select(&gather_op(&[2, 3], idx(), 0, 0, &[2, 2])).is_none());
    assert!(select(&gather_op(&[2, 3], idx(), 2, 0, &[2, 3])).is_none());
    assert!(select(&gather_op(&[2, 3], t(1, DataType::I8, &[2]), 0, 0, &[2, 3])).is_none());
    let mut wrong_type = gather_op(&[2, 3], idx(), 0, 0, &[2, 3]);
    wrong_type.outputs = Box::leak(vec![t(2, DataType::I32, &[2, 3])].into_boxed_slice());
    assert!(select(&wrong_type).is_none());
    let mut no_options = gather_op(&[2, 3], idx(), 0, 0, &[2, 3]);
    no_options.options = OpOptions::None;
    assert!(select(&no_options).is_none());
}
