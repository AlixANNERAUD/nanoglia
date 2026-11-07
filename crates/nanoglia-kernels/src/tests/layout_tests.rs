//! `PACK`, `UNPACK`, `SPACE_TO_DEPTH`, `DEPTH_TO_SPACE`, `SPACE_TO_BATCH_ND` and
//! `BATCH_TO_SPACE_ND`: kernel names and what the backend refuses. Golden cases cover the
//! results.

use super::*;

fn pack(n: usize, values_count: i32, axis: i32, out: &[i32]) -> OpDesc<'static, 'static> {
    let inputs: Vec<_> = (0..n)
        .map(|i| Some(t(i as u32, DataType::F32, &[2, 3])))
        .collect();
    op(
        BuiltinOp::PACK,
        OpOptions::Pack { values_count, axis },
        &inputs,
        &[t(9, DataType::F32, out)],
    )
}

#[test]
fn pack_checks_count_axis_and_shapes() {
    let ok = pack(2, 2, -1, &[2, 3, 2]);
    assert_eq!(select(&ok).unwrap().name(), "rust/pack");
    let x: [f32; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y: [f32; 6] = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    assert_eq!(
        run1::<f32>(&ok, &[buf(&x), buf(&y)]),
        [
            0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0, 4.0, 14.0, 5.0, 15.0
        ]
    );
    // `values_count` must be the number of inputs, the axis in `-rank - 1..=rank`.
    assert!(select(&pack(2, 3, 0, &[2, 2, 3])).is_none());
    assert!(select(&pack(2, 2, 3, &[2, 3, 2])).is_none());
    assert!(select(&pack(2, 2, -4, &[2, 3, 2])).is_none());
    // The output shape TFLite computes.
    assert!(select(&pack(2, 2, 0, &[3, 2, 2])).is_none());
    // Inputs of different shapes or quantization.
    let mixed = op(
        BuiltinOp::PACK,
        OpOptions::Pack {
            values_count: 2,
            axis: 0,
        },
        &[
            Some(t(0, DataType::F32, &[2, 3])),
            Some(t(1, DataType::F32, &[3, 2])),
        ],
        &[t(2, DataType::F32, &[2, 2, 3])],
    );
    assert!(select(&mixed).is_none());
    let requantized = op(
        BuiltinOp::PACK,
        OpOptions::Pack {
            values_count: 2,
            axis: 0,
        },
        &[
            Some(q(t(0, DataType::I8, &[2]), 0.5, 1)),
            Some(q(t(1, DataType::I8, &[2]), 0.25, 1)),
        ],
        &[q(t(2, DataType::I8, &[2, 2]), 0.5, 1)],
    );
    assert!(select(&requantized).is_none());
}

#[test]
fn unpack_checks_num_and_shapes() {
    let unpack = |num: i32, axis: i32, input: &[i32], out: &[i32]| {
        let outputs: Vec<_> = (0..num.max(0))
            .map(|i| t(1 + i as u32, DataType::I8, out))
            .collect();
        op(
            BuiltinOp::UNPACK,
            OpOptions::Unpack { num, axis },
            &[Some(t(0, DataType::I8, input))],
            &outputs,
        )
    };
    let ok = unpack(3, -1, &[2, 3], &[2]);
    assert_eq!(select(&ok).unwrap().name(), "rust/unpack");
    let out = run(&ok, &[buf(&[1i8, 2, 3, 4, 5, 6])]).unwrap();
    assert_eq!(view::<i8>(out[0]), [1, 4]);
    assert_eq!(view::<i8>(out[2]), [3, 6]);
    // `num` must be the size of the axis and the number of outputs (at most 8).
    assert!(select(&unpack(2, 1, &[2, 3], &[2])).is_none());
    assert!(select(&unpack(9, 0, &[9], &[])).is_none());
    // Empty inputs are refused, as in TFLite.
    assert!(select(&unpack(2, 0, &[2, 0], &[0])).is_none());
    // Wrong output shape.
    assert!(select(&unpack(3, 1, &[2, 3], &[3])).is_none());
}

fn depth_space(
    code: BuiltinOp,
    block_size: i32,
    input: &[i32],
    out: &[i32],
) -> OpDesc<'static, 'static> {
    op(
        code,
        OpOptions::BlockSize { block_size },
        &[Some(t(0, DataType::I8, input))],
        &[t(1, DataType::I8, out)],
    )
}

#[test]
fn space_to_depth_and_back() {
    let s2d = depth_space(BuiltinOp::SPACE_TO_DEPTH, 2, &[1, 2, 2, 2], &[1, 1, 1, 8]);
    assert_eq!(select(&s2d).unwrap().name(), "rust/space_to_depth");
    let x: Vec<i8> = (0..8).collect();
    let y = run1::<i8>(&s2d, &[buf(&x)]);
    // Pixels (0,0), (0,1), (1,0), (1,1), each of two channels.
    assert_eq!(y, x);
    let x: Vec<i8> = (0..16).collect();
    let s2d = depth_space(BuiltinOp::SPACE_TO_DEPTH, 2, &[1, 2, 4, 2], &[1, 1, 2, 8]);
    let y = run1::<i8>(&s2d, &[buf(&x)]);
    assert_eq!(y, [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]);
    let d2s = depth_space(BuiltinOp::DEPTH_TO_SPACE, 2, &[1, 1, 2, 8], &[1, 2, 4, 2]);
    assert_eq!(select(&d2s).unwrap().name(), "rust/depth_to_space");
    assert_eq!(run1::<i8>(&d2s, &[buf(&y)]), x);
}

#[test]
fn depth_space_refusals() {
    let s2d = BuiltinOp::SPACE_TO_DEPTH;
    let d2s = BuiltinOp::DEPTH_TO_SPACE;
    // Block sizes below 1, or that do not divide the input.
    assert!(select(&depth_space(s2d, 0, &[1, 2, 2, 1], &[1, 2, 2, 1])).is_none());
    assert!(select(&depth_space(s2d, -2, &[1, 2, 2, 1], &[1, 1, 1, 4])).is_none());
    assert!(select(&depth_space(s2d, 2, &[1, 3, 2, 1], &[1, 1, 1, 4])).is_none());
    assert!(select(&depth_space(d2s, 2, &[1, 1, 1, 6], &[1, 2, 2, 1])).is_none());
    // Huge block sizes are refused without overflowing.
    let huge = i32::MAX;
    assert!(select(&depth_space(s2d, huge, &[1, 0, 0, 4], &[1, 0, 0, 0])).is_none());
    assert!(select(&depth_space(d2s, huge, &[1, 1, 1, 4], &[1, huge, huge, 0])).is_none());
    // Rank 4 only.
    assert!(select(&depth_space(s2d, 2, &[2, 2, 1], &[1, 1, 4])).is_none());
}

fn space_batch(
    code: BuiltinOp,
    dtype: DataType,
    input: &[i32],
    block: &[i32],
    pads: &[i32],
    out: &[i32],
) -> OpDesc<'static, 'static> {
    let spatial = block.len() as i32;
    let quant = |x: TensorDesc<'static>| {
        if dtype == DataType::I8 {
            q(x, 0.5, -3)
        } else {
            x
        }
    };
    op(
        code,
        OpOptions::None,
        &[
            Some(quant(t(0, dtype, input))),
            Some(konst(t(1, DataType::I32, &[spatial]), block)),
            Some(konst(t(2, DataType::I32, &[spatial, 2]), pads)),
        ],
        &[quant(t(3, dtype, out))],
    )
}

#[test]
fn space_to_batch_pads_with_the_zero_point() {
    let s2b = space_batch(
        BuiltinOp::SPACE_TO_BATCH_ND,
        DataType::I8,
        &[1, 1, 2],
        &[2],
        &[1, 0],
        &[2, 1, 2],
    );
    assert_eq!(select(&s2b).unwrap().name(), "rust/space_to_batch_nd");
    // The padded rows: [pad, pad], [5, 6]; batch 0 takes the even rows.
    assert_eq!(run1::<i8>(&s2b, &[buf(&[5i8, 6])]), [-3, -3, 5, 6]);
    let b2s = space_batch(
        BuiltinOp::BATCH_TO_SPACE_ND,
        DataType::I8,
        &[2, 1, 2],
        &[2],
        &[1, 0],
        &[1, 1, 2],
    );
    assert_eq!(select(&b2s).unwrap().name(), "rust/batch_to_space_nd");
    assert_eq!(run1::<i8>(&b2s, &[buf(&[-3i8, -3, 5, 6])]), [5, 6]);
}

#[test]
fn space_batch_refusals() {
    let (s2b, b2s, f32) = (
        BuiltinOp::SPACE_TO_BATCH_ND,
        BuiltinOp::BATCH_TO_SPACE_ND,
        DataType::F32,
    );
    let ok = |op: &OpDesc<'_, '_>| select(op).is_some();
    assert!(ok(&space_batch(
        s2b,
        f32,
        &[1, 4, 4, 1],
        &[2, 2],
        &[0, 0, 0, 0],
        &[4, 2, 2, 1]
    )));
    // Negative paddings, blocks below 1, sizes not divisible by the block.
    assert!(!ok(&space_batch(
        s2b,
        f32,
        &[1, 2, 1],
        &[2],
        &[-1, 1],
        &[2, 1, 1]
    )));
    assert!(!ok(&space_batch(
        s2b,
        f32,
        &[1, 2, 1],
        &[0],
        &[0, 0],
        &[0, 2, 1]
    )));
    assert!(!ok(&space_batch(
        s2b,
        f32,
        &[1, 3, 1],
        &[2],
        &[0, 0],
        &[2, 1, 1]
    )));
    // Padded sizes that do not fit an `int`.
    let max = i32::MAX;
    assert!(!ok(&space_batch(
        s2b,
        f32,
        &[1, 2, 1],
        &[1],
        &[max, max],
        &[1, 0, 1]
    )));
    // Crops larger than the uncropped size, batches not divisible by the blocks.
    assert!(!ok(&space_batch(
        b2s,
        f32,
        &[2, 1, 1],
        &[2],
        &[2, 1],
        &[1, 0, 1]
    )));
    assert!(!ok(&space_batch(
        b2s,
        f32,
        &[3, 1, 1],
        &[2],
        &[0, 0],
        &[1, 2, 1]
    )));
    // Rank 3 or 4 only, matching block and padding shapes.
    assert!(!ok(&space_batch(s2b, f32, &[1, 2], &[], &[], &[1, 2])));
    assert!(!ok(&space_batch(
        s2b,
        f32,
        &[1, 2, 2, 1],
        &[1],
        &[0, 0],
        &[1, 2, 2, 1]
    )));
    // Quantized inputs and outputs share their quantization.
    let mut requantized = space_batch(s2b, DataType::I8, &[1, 2, 1], &[1], &[0, 0], &[1, 2, 1]);
    let outputs = vec![q(t(3, DataType::I8, &[1, 2, 1]), 0.25, -3)];
    requantized.outputs = Box::leak(outputs.into_boxed_slice());
    assert!(!ok(&requantized));
}
