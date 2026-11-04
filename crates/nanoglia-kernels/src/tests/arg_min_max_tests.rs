//! `ARG_MAX` and `ARG_MIN`: ties, NaN, axes, index types, and what the backend refuses.

use super::*;

fn arg_op<A: Pod>(
    max: bool,
    dtype: DataType,
    shape: &[i32],
    axis_type: DataType,
    axis: A,
    out_type: DataType,
    out_shape: &[i32],
) -> OpDesc<'static, 'static> {
    let code = if max {
        BuiltinOp::ARG_MAX
    } else {
        BuiltinOp::ARG_MIN
    };
    op(
        code,
        OpOptions::ArgMinMax {
            output_type: out_type,
        },
        &[
            Some(t(0, dtype, shape)),
            Some(konst(t(1, axis_type, &[1]), &[axis])),
        ],
        &[t(2, out_type, out_shape)],
    )
}

#[test]
fn arg_max_and_min_keep_the_first_extreme() {
    // [2, 3]: ties along both axes.
    let x: [f32; 6] = [1.0, 3.0, 3.0, 3.0, -1.0, 3.0];
    let max_last = arg_op(
        true,
        DataType::F32,
        &[2, 3],
        DataType::I32,
        1i32,
        DataType::I32,
        &[2],
    );
    assert_eq!(run1::<i32>(&max_last, &[buf(&x)]), [1, 0]);
    let min_last = arg_op(
        false,
        DataType::F32,
        &[2, 3],
        DataType::I32,
        -1i32,
        DataType::I32,
        &[2],
    );
    assert_eq!(run1::<i32>(&min_last, &[buf(&x)]), [0, 1]);
    // Along axis 0, with int64 axis and indices.
    let max_first = arg_op(
        true,
        DataType::F32,
        &[2, 3],
        DataType::I64,
        0i64,
        DataType::I64,
        &[3],
    );
    assert_eq!(run1::<i64>(&max_first, &[buf(&x)]), [1, 0, 0]);
    let min_first = arg_op(
        false,
        DataType::F32,
        &[2, 3],
        DataType::I64,
        -2i64,
        DataType::I32,
        &[3],
    );
    assert_eq!(run1::<i32>(&min_first, &[buf(&x)]), [0, 1, 0]);
}

#[test]
fn arg_max_skips_nan() {
    let nan = f32::NAN;
    let x: [f32; 9] = [nan, 1.0, 2.0, 5.0, nan, 4.0, nan, nan, nan];
    let max = arg_op(
        true,
        DataType::F32,
        &[3, 3],
        DataType::I32,
        1i32,
        DataType::I32,
        &[3],
    );
    assert_eq!(run1::<i32>(&max, &[buf(&x)]), [2, 0, 0]);
    let min = arg_op(
        false,
        DataType::F32,
        &[3, 3],
        DataType::I32,
        1i32,
        DataType::I32,
        &[3],
    );
    assert_eq!(run1::<i32>(&min, &[buf(&x)]), [1, 2, 0]);
    // -0.0 and +0.0 are equal: the first one wins.
    let zeros: [f32; 2] = [-0.0, 0.0];
    let max = arg_op(
        true,
        DataType::F32,
        &[2],
        DataType::I32,
        0i32,
        DataType::I32,
        &[],
    );
    assert_eq!(run1::<i32>(&max, &[buf(&zeros)]), [0]);
}

#[test]
fn arg_max_middle_axis_of_integer_types() {
    // [2, 3, 2], axis 1.
    let values: [i32; 12] = [5, -7, 9, 2, 9, 8, -1, 0, -1, 4, -3, 4];
    let max_expected = [1, 2, 0, 1];
    let min_expected = [0, 0, 2, 0];
    for dtype in [DataType::I8, DataType::I16, DataType::I32] {
        let input = match dtype {
            DataType::I8 => buf(&values.map(|v| v as i8)),
            DataType::I16 => buf(&values.map(|v| v as i16)),
            _ => buf(&values),
        };
        let max = arg_op(
            true,
            dtype,
            &[2, 3, 2],
            DataType::I32,
            1i32,
            DataType::I32,
            &[2, 2],
        );
        assert_eq!(run1::<i32>(&max, &[input]), max_expected, "{dtype:?}");
        let min = arg_op(
            false,
            dtype,
            &[2, 3, 2],
            DataType::I32,
            1i32,
            DataType::I64,
            &[2, 2],
        );
        assert_eq!(
            run1::<i64>(&min, &[input]),
            min_expected.map(i64::from),
            "{dtype:?}"
        );
    }
    // uint8 values above 127 compare as unsigned.
    let bytes: [u8; 4] = [100, 200, 255, 0];
    let max = arg_op(
        true,
        DataType::U8,
        &[4],
        DataType::I32,
        0i32,
        DataType::I32,
        &[],
    );
    assert_eq!(run1::<i32>(&max, &[buf(&bytes)]), [2]);
    let min = arg_op(
        false,
        DataType::U8,
        &[4],
        DataType::I32,
        0i32,
        DataType::I32,
        &[],
    );
    assert_eq!(run1::<i32>(&min, &[buf(&bytes)]), [3]);
    // bool.
    let flags: [u8; 6] = [0, 0, 1, 1, 0, 1];
    let max = arg_op(
        true,
        DataType::Bool,
        &[2, 3],
        DataType::I32,
        1i32,
        DataType::I32,
        &[2],
    );
    assert_eq!(run1::<i32>(&max, &[buf(&flags)]), [2, 0]);
}

#[test]
fn arg_max_refusals() {
    let ok = arg_op(
        true,
        DataType::I8,
        &[2, 3],
        DataType::I32,
        1i32,
        DataType::I32,
        &[2],
    );
    assert!(select(&ok).is_some());
    let refused = [
        // The schema default output type (FLOAT32), and a mismatched output tensor.
        arg_op(
            true,
            DataType::I8,
            &[2, 3],
            DataType::I32,
            1i32,
            DataType::F32,
            &[2],
        ),
        op(
            BuiltinOp::ARG_MAX,
            OpOptions::ArgMinMax {
                output_type: DataType::I64,
            },
            &[
                Some(t(0, DataType::I8, &[2, 3])),
                Some(konst(t(1, DataType::I32, &[1]), &[1i32])),
            ],
            &[t(2, DataType::I32, &[2])],
        ),
        // Axis out of range, wrong output shape, empty axis, unsupported type.
        arg_op(
            true,
            DataType::I8,
            &[2, 3],
            DataType::I32,
            2i32,
            DataType::I32,
            &[2],
        ),
        arg_op(
            true,
            DataType::I8,
            &[2, 3],
            DataType::I32,
            -3i32,
            DataType::I32,
            &[2],
        ),
        arg_op(
            true,
            DataType::I8,
            &[2, 3],
            DataType::I32,
            1i32,
            DataType::I32,
            &[2, 1],
        ),
        arg_op(
            true,
            DataType::I8,
            &[2, 0],
            DataType::I32,
            1i32,
            DataType::I32,
            &[2],
        ),
        arg_op(
            true,
            DataType::I64,
            &[2, 3],
            DataType::I32,
            1i32,
            DataType::I32,
            &[2],
        ),
        // A scalar input has no axis.
        arg_op(
            true,
            DataType::I8,
            &[],
            DataType::I32,
            0i32,
            DataType::I32,
            &[],
        ),
        // An axis of two elements, or of another type.
        op(
            BuiltinOp::ARG_MAX,
            OpOptions::ArgMinMax {
                output_type: DataType::I32,
            },
            &[
                Some(t(0, DataType::I8, &[2, 3])),
                Some(konst(t(1, DataType::I32, &[2]), &[1i32, 1])),
            ],
            &[t(2, DataType::I32, &[2])],
        ),
        arg_op(
            true,
            DataType::I8,
            &[2, 3],
            DataType::I16,
            1i16,
            DataType::I32,
            &[2],
        ),
        // An axis computed at run time.
        op(
            BuiltinOp::ARG_MAX,
            OpOptions::ArgMinMax {
                output_type: DataType::I32,
            },
            &[
                Some(t(0, DataType::I8, &[2, 3])),
                Some(t(1, DataType::I32, &[1])),
            ],
            &[t(2, DataType::I32, &[2])],
        ),
    ];
    for (i, op) in refused.iter().enumerate() {
        assert!(select(op).is_none(), "case {i}");
    }
}
