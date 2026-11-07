//! `DIV` and `SQUARED_DIFFERENCE`: the int16 version LiteRT lacks, integer edge cases and what
//! the backend refuses. Golden cases cover the other results.

use super::*;
use nanoglia_core::Activation;

fn binary(
    code: BuiltinOp,
    options: OpOptions<'static>,
    a: TensorDesc<'static>,
    b: TensorDesc<'static>,
    y: TensorDesc<'static>,
) -> OpDesc<'static, 'static> {
    op(code, options, &[Some(a), Some(b)], &[y])
}

#[test]
fn squared_difference_int16() {
    // TensorFlow Lite Micro, left shift 0: both inputs at scale 2^-10 are halved (input
    // multipliers 0.5), and the squared difference is requantized by 2^-10 to scale 2^-8.
    let i16t = |i, s| q(t(i, DataType::I16, &[3]), s, 0);
    let op = binary(
        BuiltinOp::SQUARED_DIFFERENCE,
        OpOptions::None,
        i16t(0, 1.0 / 1024.0),
        i16t(1, 1.0 / 1024.0),
        i16t(2, 1.0 / 256.0),
    );
    assert_eq!(select(&op).unwrap().name(), "rust/binary_q");
    // (100 / 2 - 40 / 2)² = 900, and 900 / 1024 rounds to 1; (32767 / 2 + 32768 / 2)² = 2^30,
    // which saturates the output; 0.
    let y = run1::<i16>(&op, &[buf(&[100i16, 32767, -7]), buf(&[40i16, -32768, -7])]);
    assert_eq!(y, [1, 32767, 0]);
}

#[test]
fn integer_division() {
    let i32t = |i| t(i, DataType::I32, &[3]);
    let div = |activation| {
        binary(
            BuiltinOp::DIV,
            OpOptions::Elementwise { activation },
            i32t(0),
            i32t(1),
            i32t(2),
        )
    };
    let op = div(Activation::None);
    assert_eq!(select(&op).unwrap().name(), "rust/binary");
    // C++ division truncates; the most negative value divided by -1 wraps around.
    assert_eq!(
        run1::<i32>(&op, &[buf(&[7i32, -7, i32::MIN]), buf(&[2i32, 2, -1])]),
        [3, -3, i32::MIN]
    );
    let relu6 = div(Activation::Relu6);
    assert_eq!(
        run1::<i32>(&relu6, &[buf(&[70i32, -7, 9]), buf(&[2i32, 2, 3])]),
        [6, 0, 3]
    );
    // A zero denominator fails, as in TFLite.
    let err = run(&op, &[buf(&[1i32, 2, 3]), buf(&[1i32, 0, 1])]).unwrap_err();
    assert_eq!(err.code, nanoglia_core::ErrorCode::InvalidArgument);
}

#[test]
fn binary_refusals() {
    let i8t = |i| q(t(i, DataType::I8, &[2]), 0.1, 0);
    let refused = |op: OpDesc<'static, 'static>| assert!(select(&op).is_none());
    // Quantized DIV (TFLite's DivElementwise shifts by more than 31 bits on small quotients).
    refused(binary(
        BuiltinOp::DIV,
        OpOptions::None,
        i8t(0),
        i8t(1),
        i8t(2),
    ));
    // Quantized int32 DIV, int64 DIV, uint8 SQUARED_DIFFERENCE.
    refused(binary(
        BuiltinOp::DIV,
        OpOptions::None,
        q(t(0, DataType::I32, &[2]), 0.5, 0),
        t(1, DataType::I32, &[2]),
        t(2, DataType::I32, &[2]),
    ));
    refused(binary(
        BuiltinOp::DIV,
        OpOptions::None,
        t(0, DataType::I64, &[2]),
        t(1, DataType::I64, &[2]),
        t(2, DataType::I64, &[2]),
    ));
    let u8t = |i| q(t(i, DataType::U8, &[2]), 0.1, 128);
    refused(binary(
        BuiltinOp::SQUARED_DIFFERENCE,
        OpOptions::None,
        u8t(0),
        u8t(1),
        u8t(2),
    ));
    // DIV has no TANH activation; SQUARED_DIFFERENCE no activation at all.
    refused(binary(
        BuiltinOp::DIV,
        OpOptions::Elementwise {
            activation: Activation::Tanh,
        },
        t(0, DataType::F32, &[2]),
        t(1, DataType::F32, &[2]),
        t(2, DataType::F32, &[2]),
    ));
    refused(binary(
        BuiltinOp::SQUARED_DIFFERENCE,
        OpOptions::Elementwise {
            activation: Activation::Relu,
        },
        t(0, DataType::F32, &[2]),
        t(1, DataType::F32, &[2]),
        t(2, DataType::F32, &[2]),
    ));
}
