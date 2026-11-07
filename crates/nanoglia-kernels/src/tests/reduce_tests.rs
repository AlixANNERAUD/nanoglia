//! `SUM`, `REDUCE_MAX` and `REDUCE_MIN`: special values, overflow, kernel names and what the
//! backend refuses. Golden cases cover the results on every type.

use super::*;

/// A reduction of `input` over the constant int32 `axes` into `output`.
fn reduce(
    code: BuiltinOp,
    input: TensorDesc<'static>,
    axes: &[i32],
    output: TensorDesc<'static>,
) -> OpDesc<'static, 'static> {
    op(
        code,
        OpOptions::Reducer { keep_dims: false },
        &[
            Some(input),
            Some(konst(t(1, DataType::I32, &[axes.len() as i32]), axes)),
        ],
        &[output],
    )
}

#[test]
fn reductions_select_their_kernels() {
    let f = |code| {
        let op = reduce(
            code,
            t(0, DataType::F32, &[2, 3]),
            &[1],
            t(2, DataType::F32, &[2]),
        );
        select(&op).unwrap().name()
    };
    assert_eq!(f(BuiltinOp::MEAN), "rust/mean");
    assert_eq!(f(BuiltinOp::SUM), "rust/sum");
    assert_eq!(f(BuiltinOp::REDUCE_MAX), "rust/reduce_min_max");
    assert_eq!(f(BuiltinOp::REDUCE_MIN), "rust/reduce_min_max");
}

#[test]
fn min_max_skip_nan_and_start_from_the_finite_extremes() {
    // TensorFlow Lite Micro starts from `lowest()` / `max()` and keeps the current value unless
    // the next one compares greater (smaller): NaN never replaces it, and infinities of the
    // wrong sign leave the finite extreme.
    let x = [
        f32::NAN,
        1.0,
        -2.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::INFINITY,
        f32::INFINITY,
    ];
    let run_op = |code| {
        let op = reduce(
            code,
            t(0, DataType::F32, &[4, 2]),
            &[1],
            t(2, DataType::F32, &[4]),
        );
        run1::<f32>(&op, &[buf(&x)])
    };
    assert_bits(
        &run_op(BuiltinOp::REDUCE_MAX),
        &[1.0, -2.0, f32::MIN, f32::INFINITY],
    );
    assert_bits(
        &run_op(BuiltinOp::REDUCE_MIN),
        &[1.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::MAX],
    );
}

#[test]
fn integer_sums_wrap_around() {
    let op = reduce(
        BuiltinOp::SUM,
        t(0, DataType::I32, &[2, 2]),
        &[-1],
        t(2, DataType::I32, &[2]),
    );
    let x = [i32::MAX, 2, i32::MIN, -1];
    assert_eq!(
        run1::<i32>(&op, &[buf(&x)]),
        [i32::MIN.wrapping_add(1), i32::MAX]
    );
    let op = reduce(
        BuiltinOp::SUM,
        t(0, DataType::I64, &[3]),
        &[0],
        t(2, DataType::I64, &[]),
    );
    assert_eq!(run1::<i64>(&op, &[buf(&[i64::MAX, 1, 5])]), [i64::MIN + 5]);
}

#[test]
fn quantized_sum_requantizes_in_fixed_point() {
    // Sum of (x - zero point) = 4 + 6 = 10 at scale 0.5 is 5.0, i.e. 25 at scale 0.2, plus 3.
    let op = reduce(
        BuiltinOp::SUM,
        q(t(0, DataType::I8, &[2]), 0.5, -1),
        &[0],
        q(t(2, DataType::I8, &[]), 0.2, 3),
    );
    assert_eq!(run1::<i8>(&op, &[buf(&[3i8, 5])]), [28]);
}

#[test]
fn int16_sum_accumulates_in_32_bits_and_saturates() {
    // Scale 1 to 2: multiplier 0.5, exact on even sums. The golden cases stay in the int8 range
    // (their outputs come from an equivalent uint8 model): these rows use the whole int16 range,
    // past it in the accumulator, and saturate on both sides.
    let op = reduce(
        BuiltinOp::SUM,
        q(t(0, DataType::I16, &[3, 3]), 1.0, 0),
        &[1],
        q(t(2, DataType::I16, &[3]), 2.0, 0),
    );
    let x: [i16; 9] = [
        30000,
        30002,
        0,
        i16::MAX,
        i16::MAX,
        i16::MAX - 1,
        i16::MIN,
        i16::MIN,
        i16::MIN,
    ];
    assert_eq!(run1::<i16>(&op, &[buf(&x)]), [30001, i16::MAX, i16::MIN]);
}

#[test]
fn backend_refuses_unsupported_reductions() {
    let i8t = |i, dims: &[i32], s, z| q(t(i, DataType::I8, dims), s, z);
    // REDUCE_MAX compares without requantizing: the quantization must be the same.
    for (s, z) in [(0.5, 1), (0.25, 0)] {
        let op = reduce(
            BuiltinOp::REDUCE_MAX,
            i8t(0, &[2, 2], 0.25, 1),
            &[1],
            i8t(2, &[2], s, z),
        );
        assert!(select(&op).is_none());
    }
    // A multiplier that would shift an `i32` left by more than 30 bits.
    let op = reduce(
        BuiltinOp::SUM,
        i8t(0, &[2, 2], 1.0e10, 0),
        &[1],
        i8t(2, &[2], 1.0e-10, 0),
    );
    assert!(select(&op).is_none());
    // int16 is symmetric.
    let op = reduce(
        BuiltinOp::SUM,
        q(t(0, DataType::I16, &[2, 2]), 0.5, 1),
        &[1],
        q(t(2, DataType::I16, &[2]), 0.5, 0),
    );
    assert!(select(&op).is_none());
    // MEAN has no integer path for int32, SUM none for bool.
    let op = reduce(
        BuiltinOp::MEAN,
        t(0, DataType::I32, &[2, 2]),
        &[1],
        t(2, DataType::I32, &[2]),
    );
    assert!(select(&op).is_none());
    let op = reduce(
        BuiltinOp::SUM,
        t(0, DataType::Bool, &[2, 2]),
        &[1],
        t(2, DataType::Bool, &[2]),
    );
    assert!(select(&op).is_none());
    // The output must have the kept dimensions' element count.
    let op = reduce(
        BuiltinOp::SUM,
        t(0, DataType::F32, &[2, 3]),
        &[1],
        t(2, DataType::F32, &[3]),
    );
    assert!(select(&op).is_none());
    // Out-of-range axis.
    let op = reduce(
        BuiltinOp::REDUCE_MIN,
        t(0, DataType::F32, &[2, 3]),
        &[2],
        t(2, DataType::F32, &[2]),
    );
    assert!(select(&op).is_none());
}
