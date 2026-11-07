//! `ABS`, `NEG`, `SQRT`, `RSQRT`, `EXP` and `LOG`: kernel names, special values, the failure of
//! quantized `RSQRT` on negative values and what the backend refuses. Golden cases cover the
//! results.

use super::*;

fn unary(
    code: BuiltinOp,
    x: TensorDesc<'static>,
    y: TensorDesc<'static>,
) -> OpDesc<'static, 'static> {
    op(code, OpOptions::None, &[Some(x)], &[y])
}

#[test]
fn unary_kernels() {
    let f = |code| unary(code, t(0, DataType::F32, &[4]), t(1, DataType::F32, &[4]));
    let x = [4.0f32, -0.0, f32::INFINITY, -1.0];
    let cases: [(BuiltinOp, [f32; 4]); 6] = [
        (BuiltinOp::ABS, [4.0, 0.0, f32::INFINITY, 1.0]),
        (BuiltinOp::NEG, [-4.0, 0.0, f32::NEG_INFINITY, 1.0]),
        (BuiltinOp::SQRT, [2.0, -0.0, f32::INFINITY, f32::NAN]),
        (BuiltinOp::RSQRT, [0.5, f32::NEG_INFINITY, 0.0, f32::NAN]),
        (
            BuiltinOp::EXP,
            [54.598_15, 1.0, f32::INFINITY, 0.367_879_45],
        ),
        (
            BuiltinOp::LOG,
            [1.386_294_4, f32::NEG_INFINITY, f32::INFINITY, f32::NAN],
        ),
    ];
    for (code, expected) in cases {
        let op = f(code);
        assert_eq!(select(&op).unwrap().name(), "rust/unary_f32");
        let y = run1::<f32>(&op, &[buf(&x)]);
        for (a, e) in y.iter().zip(expected) {
            assert!(a == &e || (a.is_nan() && e.is_nan()), "{code:?}: {y:?}");
        }
    }
    let int = unary(
        BuiltinOp::NEG,
        t(0, DataType::I32, &[2]),
        t(1, DataType::I32, &[2]),
    );
    assert_eq!(select(&int).unwrap().name(), "rust/unary_int");
    assert_eq!(run1::<i32>(&int, &[buf(&[i32::MIN, 5])]), [i32::MIN, -5]);
    let q8 = unary(
        BuiltinOp::ABS,
        q(t(0, DataType::I8, &[3]), 0.5, 2),
        q(t(1, DataType::I8, &[3]), 0.5, -1),
    );
    assert_eq!(select(&q8).unwrap().name(), "rust/unary_q");
    // |x - 2| - 1, without rescaling.
    assert_eq!(run1::<i8>(&q8, &[buf(&[-128i8, 2, 127])]), [127, -1, 124]);
}

#[test]
fn quantized_rsqrt_fails_on_negative_values() {
    let op = unary(
        BuiltinOp::RSQRT,
        q(t(0, DataType::I8, &[2]), 0.25, -10),
        q(t(1, DataType::I8, &[2]), 0.0625, -128),
    );
    // x = 0.25 * (q + 10): 1.0 gives 1 / 1 = 1.0, i.e. 16 - 128; 0 gives 127.
    assert_eq!(run1::<i8>(&op, &[buf(&[-6i8, -10])]), [-112, 127]);
    let err = run(&op, &[buf(&[-6i8, -11])]).unwrap_err();
    assert_eq!(err.code, nanoglia_core::ErrorCode::InvalidArgument);
}

#[test]
fn unary_refusals() {
    let refused = |op: OpDesc<'static, 'static>| assert!(select(&op).is_none());
    // No quantized NEG, SQRT, EXP or LOG; no uint8; no int16 RSQRT.
    for code in [
        BuiltinOp::NEG,
        BuiltinOp::SQRT,
        BuiltinOp::EXP,
        BuiltinOp::LOG,
    ] {
        refused(unary(
            code,
            q(t(0, DataType::I8, &[2]), 0.5, 0),
            q(t(1, DataType::I8, &[2]), 0.5, 0),
        ));
    }
    refused(unary(
        BuiltinOp::ABS,
        q(t(0, DataType::U8, &[2]), 0.5, 0),
        q(t(1, DataType::U8, &[2]), 0.5, 0),
    ));
    refused(unary(
        BuiltinOp::RSQRT,
        q(t(0, DataType::I16, &[2]), 0.5, 0),
        q(t(1, DataType::I16, &[2]), 0.5, 0),
    ));
    // int16 is symmetric; shapes and types must match; multipliers must fit.
    refused(unary(
        BuiltinOp::ABS,
        q(t(0, DataType::I16, &[2]), 0.5, 3),
        q(t(1, DataType::I16, &[2]), 0.5, 0),
    ));
    refused(unary(
        BuiltinOp::ABS,
        t(0, DataType::F32, &[2]),
        t(1, DataType::F32, &[3]),
    ));
    refused(unary(
        BuiltinOp::SQRT,
        t(0, DataType::F32, &[2]),
        t(1, DataType::I32, &[2]),
    ));
    refused(unary(
        BuiltinOp::ABS,
        q(t(0, DataType::I8, &[2]), 1.0e10, 0),
        q(t(1, DataType::I8, &[2]), 1.0e-10, 0),
    ));
    refused(unary(
        BuiltinOp::RSQRT,
        q(t(0, DataType::I8, &[2]), 1.0e-30, 0),
        q(t(1, DataType::I8, &[2]), 1.0e-10, 0),
    ));
}
