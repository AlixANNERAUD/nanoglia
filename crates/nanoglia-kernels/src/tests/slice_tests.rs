//! `TRANSPOSE`, `SLICE` and `STRIDED_SLICE`: vectors from the TFLite kernel tests (`*_test.cc`),
//! and what the backend refuses. The golden cases of `testdata/` compare the same kernels with
//! LiteRT on more shapes and types.

use super::*;
use crate::strided::StridedCopy;
use nanoglia_core::StridedSliceOptions;

pub(crate) fn iota(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

fn transpose_op(
    dtype: DataType,
    shape: &[i32],
    perm: &[i32],
    out: &[i32],
) -> OpDesc<'static, 'static> {
    op(
        BuiltinOp::TRANSPOSE,
        OpOptions::None,
        &[
            Some(t(0, dtype, shape)),
            Some(konst(t(1, DataType::I32, &[perm.len() as i32]), perm)),
        ],
        &[t(2, dtype, out)],
    )
}

#[test]
fn transpose_permutes_axes() {
    // transpose_test.cc Test3DInputDynamicTensor: perm [2, 0, 1] of a 2x3x4 iota.
    let op1 = transpose_op(DataType::I32, &[2, 3, 4], &[2, 0, 1], &[4, 2, 3]);
    assert_eq!(
        run1::<i32>(&op1, &[buf(&iota(24))]),
        [
            0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23
        ]
    );
    // Negative axes wrap; 1, 2 and 8-byte elements.
    let x: Vec<i8> = (0..6).collect();
    let op8 = transpose_op(DataType::I8, &[2, 3], &[-1, 0], &[3, 2]);
    assert_eq!(run1::<i8>(&op8, &[buf(&x)]), [0, 3, 1, 4, 2, 5]);
    let x: Vec<i16> = (0..6).collect();
    let op16 = transpose_op(DataType::I16, &[2, 3], &[1, 0], &[3, 2]);
    assert_eq!(run1::<i16>(&op16, &[buf(&x)]), [0, 3, 1, 4, 2, 5]);
    let x: Vec<i64> = (0..6).map(|v| v << 40).collect();
    let op64 = transpose_op(DataType::I64, &[2, 3], &[1, 0], &[3, 2]);
    assert_eq!(
        run1::<i64>(&op64, &[buf(&x)]),
        [0, 3, 1, 4, 2, 5].map(|v: i64| v << 40)
    );
    // The identity copies rows whole.
    let op_id = transpose_op(DataType::I32, &[2, 3, 4], &[0, 1, 2], &[2, 3, 4]);
    assert_eq!(run1::<i32>(&op_id, &[buf(&iota(24))]), iota(24));
}

#[test]
fn transpose_refusals() {
    // Not a permutation, out of range, wrong output shape, int64 or non-constant permutation,
    // int4 elements.
    assert!(select(&transpose_op(DataType::I8, &[2, 3], &[0, 0], &[2, 2])).is_none());
    assert!(select(&transpose_op(DataType::I8, &[2, 3], &[0, 2], &[2, 3])).is_none());
    assert!(select(&transpose_op(DataType::I8, &[2, 3], &[1, 0], &[2, 3])).is_none());
    assert!(select(&transpose_op(DataType::I4, &[2, 4], &[1, 0], &[4, 2])).is_none());
    let perm64 = op(
        BuiltinOp::TRANSPOSE,
        OpOptions::None,
        &[
            Some(t(0, DataType::I8, &[2, 3])),
            Some(konst(t(1, DataType::I64, &[2]), &[1i64, 0])),
        ],
        &[t(2, DataType::I8, &[3, 2])],
    );
    assert!(select(&perm64).is_none());
    let dynamic = op(
        BuiltinOp::TRANSPOSE,
        OpOptions::None,
        &[
            Some(t(0, DataType::I8, &[2, 3])),
            Some(t(1, DataType::I32, &[2])),
        ],
        &[t(2, DataType::I8, &[3, 2])],
    );
    assert!(select(&dynamic).is_none());
}

fn slice_op(
    shape: &[i32],
    begin: &[i64],
    size: &[i64],
    out: &[i32],
    int64: bool,
) -> OpDesc<'static, 'static> {
    let n = begin.len() as i32;
    let index = |i: u32, v: &[i64]| {
        if int64 {
            konst(t(i, DataType::I64, &[n]), v)
        } else {
            let v: Vec<i32> = v.iter().map(|&x| x as i32).collect();
            konst(t(i, DataType::I32, &[n]), &v)
        }
    };
    op(
        BuiltinOp::SLICE,
        OpOptions::None,
        &[
            Some(t(0, DataType::I32, shape)),
            Some(index(1, begin)),
            Some(index(2, size)),
        ],
        &[t(3, DataType::I32, out)],
    )
}

#[test]
fn slice_vectors_and_refusals() {
    // slice_test.cc In3D and SizeMinus1, with int32 and int64 indices.
    for int64 in [false, true] {
        let op1 = slice_op(&[2, 3, 2], &[1, 0, 0], &[1, -1, 2], &[1, 3, 2], int64);
        assert_eq!(run1::<i32>(&op1, &[buf(&iota(12))]), [6, 7, 8, 9, 10, 11]);
        let op2 = slice_op(
            &[3, 2, 3, 1],
            &[1, 0, 0, 0],
            &[2, 1, -1, 1],
            &[2, 1, 3, 1],
            int64,
        );
        assert_eq!(run1::<i32>(&op2, &[buf(&iota(18))]), [6, 7, 8, 12, 13, 14]);
    }
    // An empty slice at the end of an axis.
    let empty = slice_op(&[4], &[4], &[0], &[0], false);
    assert!(run1::<i32>(&empty, &[buf(&iota(4))]).is_empty());
    // Begin out of range, size below -1 or too large, wrong output shape.
    assert!(select(&slice_op(&[4], &[5], &[-1], &[0], false)).is_none());
    assert!(select(&slice_op(&[4], &[-1], &[1], &[1], false)).is_none());
    assert!(select(&slice_op(&[4], &[1], &[-2], &[3], false)).is_none());
    assert!(select(&slice_op(&[4], &[1], &[4], &[4], false)).is_none());
    assert!(select(&slice_op(&[4], &[1], &[2], &[3], false)).is_none());
}

/// A 1D strided slice: `begin`, `end`, `strides`, options and the expected output.
type Case1D = (
    &'static [i32],
    &'static [i32],
    &'static [i32],
    StridedSliceOptions,
    &'static [f32],
);

fn strided(
    shape: &[i32],
    begin: &[i32],
    end: &[i32],
    strides: &[i32],
    options: StridedSliceOptions,
    out: &[i32],
) -> OpDesc<'static, 'static> {
    let n = begin.len() as i32;
    op(
        BuiltinOp::STRIDED_SLICE,
        OpOptions::StridedSlice(options),
        &[
            Some(t(0, DataType::F32, shape)),
            Some(konst(t(1, DataType::I32, &[n]), begin)),
            Some(konst(t(2, DataType::I32, &[n]), end)),
            Some(konst(t(3, DataType::I32, &[n]), strides)),
        ],
        &[t(4, DataType::F32, out)],
    )
}

/// Runs a strided slice on `1, 2, 3...` and returns the output.
fn run_strided(op: &OpDesc<'static, 'static>, len: usize) -> Vec<f32> {
    let x: Vec<f32> = (1..=len).map(|v| v as f32).collect();
    run1::<f32>(op, &[buf(&x)])
}

#[test]
fn strided_slice_vectors() {
    let none = StridedSliceOptions::default();
    // strided_slice_test.cc, 1D cases on [1, 2, 3, 4].
    let cases: &[Case1D] = &[
        (&[1], &[3], &[1], none, &[2., 3.]),
        (&[-3], &[3], &[1], none, &[2., 3.]),
        (&[-5], &[3], &[1], none, &[1., 2., 3.]),
        (&[1], &[-2], &[1], none, &[2.]),
        (&[1], &[5], &[1], none, &[2., 3., 4.]),
        (&[-1], &[-4], &[-1], none, &[4., 3., 2.]),
        (&[3], &[0], &[-1], none, &[4., 3., 2.]),
        (&[10], &[3], &[1], none, &[]),
        (
            &[1],
            &[3],
            &[1],
            StridedSliceOptions {
                begin_mask: 1,
                ..none
            },
            &[1., 2., 3.],
        ),
        (
            &[1],
            &[3],
            &[1],
            StridedSliceOptions {
                end_mask: 1,
                ..none
            },
            &[2., 3., 4.],
        ),
        (&[0], &[4], &[3], none, &[1., 4.]),
    ];
    for &(begin, end, strides, options, expected) in cases {
        let op1 = strided(&[4], begin, end, strides, options, &[expected.len() as i32]);
        assert_eq!(
            run_strided(&op1, 4),
            expected,
            "{begin:?} {end:?} {strides:?}"
        );
    }
    // A shrunk axis: In1D_ShrinkAxisMask1 gives a scalar.
    let shrink = StridedSliceOptions {
        shrink_axis_mask: 1,
        ..none
    };
    assert_eq!(
        run_strided(&strided(&[4], &[1], &[2], &[1], shrink, &[]), 4),
        [2.]
    );
    // In3D_IdentityShrinkAxis4 and its masks on a 2x3x2 tensor.
    let shrink4 = StridedSliceOptions {
        shrink_axis_mask: 4,
        ..none
    };
    let op3 = strided(
        &[2, 3, 2],
        &[0, 0, 0],
        &[2, 3, 1],
        &[1, 1, 1],
        shrink4,
        &[2, 3],
    );
    assert_eq!(run_strided(&op3, 12), [1., 3., 5., 7., 9., 11.]);
    // Ellipsis and new axis: `x[..., None, 1]` with the last axis shrunk.
    let ellipsis = StridedSliceOptions {
        ellipsis_mask: 1,
        new_axis_mask: 2,
        shrink_axis_mask: 4,
        ..none
    };
    let op4 = strided(
        &[2, 3, 2],
        &[0, 0, 1],
        &[0, 0, 2],
        &[1, 1, 1],
        ellipsis,
        &[2, 3, 1],
    );
    assert_eq!(run_strided(&op4, 12), [2., 4., 6., 8., 10., 12.]);
    // Offset: `end` holds sizes.
    let offset = StridedSliceOptions {
        offset: true,
        ..none
    };
    let op5 = strided(
        &[2, 3, 2],
        &[1, 1, 0],
        &[1, 2, 2],
        &[1, 1, 1],
        offset,
        &[1, 2, 2],
    );
    assert_eq!(run_strided(&op5, 12), [9., 10., 11., 12.]);
}

#[test]
fn strided_slice_refusals() {
    let none = StridedSliceOptions::default();
    // Zero stride, wrong output shape, several ellipses.
    assert!(select(&strided(&[4], &[0], &[4], &[0], none, &[4])).is_none());
    assert!(select(&strided(&[4], &[0], &[4], &[1], none, &[3])).is_none());
    let two = StridedSliceOptions {
        ellipsis_mask: 3,
        ..none
    };
    assert!(select(&strided(&[2, 2], &[0, 0], &[1, 1], &[1, 1], two, &[2, 2])).is_none());
    // A shrunk axis selecting nothing: TFLite would leave the output unwritten.
    let shrink = StridedSliceOptions {
        shrink_axis_mask: 1,
        ..none
    };
    assert!(select(&strided(&[4], &[4], &[5], &[1], shrink, &[])).is_none());
    assert!(select(&strided(&[4], &[2], &[1], &[-1], shrink, &[])).is_none());
    // Offset whose sizes overrun the input: TFLite's output shape and loop disagree.
    let offset = StridedSliceOptions {
        offset: true,
        ..none
    };
    assert!(select(&strided(&[4], &[2], &[3], &[1], offset, &[3])).is_none());
    // Non-constant indices and the wrong options.
    let mut dynamic = strided(&[4], &[0], &[4], &[1], none, &[4]);
    let mut inputs = dynamic.inputs.to_vec();
    inputs[1] = Some(t(1, DataType::I32, &[1]));
    dynamic.inputs = Box::leak(inputs.into_boxed_slice());
    assert!(select(&dynamic).is_none());
    let mut no_options = strided(&[4], &[0], &[4], &[1], none, &[4]);
    no_options.options = OpOptions::None;
    assert!(select(&no_options).is_none());
}

#[test]
fn strided_copy_merges_contiguous_axes() {
    let err = Error::invalid_argument(Detail::None);
    // A row-major copy of the whole input is one run.
    let copy = StridedCopy::new(&[(2, 12), (3, 4), (4, 1)], 0, 24, 4, err).unwrap();
    let out = buf(&[0i32; 24]);
    copy.run(buf(&iota(24)), out).unwrap();
    assert_eq!(view::<i32>(out), iota(24));
    // Reads outside the input are refused when planning.
    assert!(StridedCopy::new(&[(2, 12), (3, 4), (4, 1)], 1, 24, 4, err).is_err());
    assert!(StridedCopy::new(&[(3, -1)], 1, 24, 4, err).is_err());
    assert!(StridedCopy::new(&[(3, i64::MAX)], 0, 24, 4, err).is_err());
    // Short buffers fail at evaluation.
    assert!(copy.run(&[0u8; 8], out).is_err());
}
