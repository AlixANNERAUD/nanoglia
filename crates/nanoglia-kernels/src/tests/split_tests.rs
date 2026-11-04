//! `SPLIT` and `SPLIT_V`: vectors from the TFLite kernel tests (`split_test.cc`,
//! `split_v_test.cc`), and what the backend refuses.

use super::slice_tests::iota;
use super::*;

fn split_op(
    v: bool,
    shape: &[i32],
    axis: i32,
    sizes: &[i32],
    outs: &[&[i32]],
) -> OpDesc<'static, 'static> {
    let data = Some(t(0, DataType::I32, shape));
    let axis = Some(konst(t(1, DataType::I32, &[1]), &[axis]));
    let outputs: Vec<TensorDesc<'static>> = outs
        .iter()
        .enumerate()
        .map(|(i, s)| t(10 + i as u32, DataType::I32, s))
        .collect();
    let num_splits = outs.len() as i32;
    if v {
        let sizes = Some(konst(t(2, DataType::I32, &[sizes.len() as i32]), sizes));
        op(
            BuiltinOp::SPLIT_V,
            OpOptions::Split { num_splits },
            &[data, sizes, axis],
            &outputs,
        )
    } else {
        op(
            BuiltinOp::SPLIT,
            OpOptions::Split { num_splits },
            &[axis, data],
            &outputs,
        )
    }
}

#[test]
fn split_vectors_and_refusals() {
    // split_test.cc FourDimensional, axis 1 and -3 of a 2x2x2x2 iota.
    for axis in [1, -3] {
        let op1 = split_op(
            false,
            &[2, 2, 2, 2],
            axis,
            &[],
            &[&[2, 1, 2, 2], &[2, 1, 2, 2]],
        );
        let out = run(&op1, &[buf(&iota(16))]).unwrap();
        assert_eq!(view::<i32>(out[0]), [0, 1, 2, 3, 8, 9, 10, 11]);
        assert_eq!(view::<i32>(out[1]), [4, 5, 6, 7, 12, 13, 14, 15]);
    }
    // split_v_test.cc: sizes with -1.
    let op2 = split_op(true, &[2, 5], 1, &[1, -1, 2], &[&[2, 1], &[2, 2], &[2, 2]]);
    let out = run(&op2, &[buf(&iota(10))]).unwrap();
    assert_eq!(view::<i32>(out[0]), [0, 5]);
    assert_eq!(view::<i32>(out[1]), [1, 2, 6, 7]);
    assert_eq!(view::<i32>(out[2]), [3, 4, 8, 9]);
    // An empty output.
    let op3 = split_op(true, &[2, 3], 1, &[3, 0], &[&[2, 3], &[2, 0]]);
    let out = run(&op3, &[buf(&iota(6))]).unwrap();
    assert_eq!(view::<i32>(out[0]), iota(6));

    // Uneven split, wrong output shape, axis out of range, two -1, sizes not summing to the axis
    // or negative, num_splits not matching the outputs.
    assert!(select(&split_op(false, &[2, 3], 1, &[], &[&[2, 1], &[2, 1]])).is_none());
    assert!(select(&split_op(false, &[2, 4], 1, &[], &[&[2, 2], &[2, 1]])).is_none());
    assert!(select(&split_op(false, &[2, 4], 2, &[], &[&[2, 2], &[2, 2]])).is_none());
    assert!(select(&split_op(true, &[2, 4], 1, &[-1, -1], &[&[2, 2], &[2, 2]])).is_none());
    assert!(select(&split_op(true, &[2, 4], 1, &[1, 2], &[&[2, 1], &[2, 2]])).is_none());
    assert!(select(&split_op(true, &[2, 4], 1, &[5, -1], &[&[2, 5], &[2, -1]])).is_none());
    assert!(select(&split_op(true, &[2, 4], 1, &[6, -2], &[&[2, 6], &[2, 0]])).is_none());
    let mut wrong_count = split_op(false, &[2, 4], 1, &[], &[&[2, 2], &[2, 2]]);
    wrong_count.options = OpOptions::Split { num_splits: 4 };
    assert!(select(&wrong_count).is_none());
}
