//! `RESIZE_NEAREST_NEIGHBOR` and `RESIZE_BILINEAR`: small vectors, and what the backend refuses.

use super::*;

fn resize_op(
    code: BuiltinOp,
    dtype: DataType,
    shape: &[i32],
    size: &[i32],
    (align_corners, half_pixel_centers): (bool, bool),
) -> OpDesc<'static, 'static> {
    let out = [shape[0], size[0], size[1], shape[3]];
    op(
        code,
        OpOptions::Resize {
            align_corners,
            half_pixel_centers,
        },
        &[
            Some(t(0, dtype, shape)),
            Some(konst(t(1, DataType::I32, &[2]), size)),
        ],
        &[t(2, dtype, &out)],
    )
}

const NEAREST: BuiltinOp = BuiltinOp::RESIZE_NEAREST_NEIGHBOR;
const BILINEAR: BuiltinOp = BuiltinOp::RESIZE_BILINEAR;

#[test]
fn resize_nearest_neighbor_vectors() {
    // 2x2 to 3x3: the scale is 2/3, indices floor(0, 2/3, 4/3) = 0, 0, 1.
    let x: [i8; 4] = [3, 6, 9, 12];
    let up = resize_op(
        NEAREST,
        DataType::I8,
        &[1, 2, 2, 1],
        &[3, 3],
        (false, false),
    );
    assert_eq!(run1::<i8>(&up, &[buf(&x)]), [3, 3, 6, 3, 3, 6, 9, 9, 12]);
    // Aligned corners: round(0, 0.5, 1) = 0, 1, 1.
    let aligned = resize_op(NEAREST, DataType::I8, &[1, 2, 2, 1], &[3, 3], (true, false));
    assert_eq!(
        run1::<i8>(&aligned, &[buf(&x)]),
        [3, 6, 6, 9, 12, 12, 9, 12, 12]
    );
    // Half-pixel centers: floor((0.5, 1.5, 2.5) * 2/3) = 0, 1, 1.
    let half = resize_op(NEAREST, DataType::I8, &[1, 2, 2, 1], &[3, 3], (false, true));
    assert_eq!(
        run1::<i8>(&half, &[buf(&x)]),
        [3, 6, 6, 9, 12, 12, 9, 12, 12]
    );
    // Several channels move together; f32 and i16 elements.
    let pixels: [f32; 4] = [1.0, -1.0, 2.0, -2.0];
    let wide = resize_op(
        NEAREST,
        DataType::F32,
        &[1, 1, 2, 2],
        &[1, 4],
        (false, false),
    );
    assert_bits(
        &run1::<f32>(&wide, &[buf(&pixels)]),
        &[1.0, -1.0, 1.0, -1.0, 2.0, -2.0, 2.0, -2.0],
    );
    let values: [i16; 3] = [-300, 0, 300];
    let down = resize_op(
        NEAREST,
        DataType::I16,
        &[1, 3, 1, 1],
        &[1, 1],
        (false, false),
    );
    assert_eq!(run1::<i16>(&down, &[buf(&values)]), [-300]);
}

#[test]
fn resize_bilinear_vectors() {
    // resize_bilinear_test.cc: 2x2 to 3x3.
    let x: [f32; 4] = [3.0, 6.0, 9.0, 12.0];
    let expected = [3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 9.0, 11.0, 12.0];
    let f = resize_op(
        BILINEAR,
        DataType::F32,
        &[1, 2, 2, 1],
        &[3, 3],
        (false, false),
    );
    assert_bits(&run1::<f32>(&f, &[buf(&x)]), &expected);
    // The same in fixed point (i8, i16) and in float with rounding (u8).
    let q = resize_op(
        BILINEAR,
        DataType::I8,
        &[1, 2, 2, 1],
        &[3, 3],
        (false, false),
    );
    assert_eq!(
        run1::<i8>(&q, &[buf(&x.map(|v| v as i8))]),
        expected.map(|v| v as i8)
    );
    let q = resize_op(
        BILINEAR,
        DataType::I16,
        &[1, 2, 2, 1],
        &[3, 3],
        (false, false),
    );
    assert_eq!(
        run1::<i16>(&q, &[buf(&x.map(|v| v as i16))]),
        expected.map(|v| v as i16)
    );
    let q = resize_op(
        BILINEAR,
        DataType::U8,
        &[1, 2, 2, 1],
        &[3, 3],
        (false, false),
    );
    assert_eq!(
        run1::<u8>(&q, &[buf(&x.map(|v| v as u8))]),
        expected.map(|v| v as u8)
    );
    // Fixed point rounds half away from zero: (-3 + -6) / 2 = -4.5 gives -5.
    let neg: [i8; 2] = [-3, -6];
    let q = resize_op(
        BILINEAR,
        DataType::I8,
        &[1, 1, 2, 1],
        &[1, 3],
        (true, false),
    );
    assert_eq!(run1::<i8>(&q, &[buf(&neg)]), [-3, -5, -6]);
}

#[test]
fn resize_refusals() {
    let ok = resize_op(
        BILINEAR,
        DataType::I8,
        &[1, 2, 2, 1],
        &[3, 3],
        (false, false),
    );
    assert!(select(&ok).is_some());
    let mut refused = vec![
        // TFLite refuses both options together in RESIZE_BILINEAR, and a zero size.
        resize_op(BILINEAR, DataType::I8, &[1, 2, 2, 1], &[3, 3], (true, true)),
        resize_op(
            BILINEAR,
            DataType::I8,
            &[1, 2, 2, 1],
            &[0, 3],
            (false, false),
        ),
        // Types TFLite does not resize.
        resize_op(
            BILINEAR,
            DataType::I32,
            &[1, 2, 2, 1],
            &[3, 3],
            (false, false),
        ),
        resize_op(
            NEAREST,
            DataType::I64,
            &[1, 2, 2, 1],
            &[3, 3],
            (false, false),
        ),
        // An empty input with a non-empty output.
        resize_op(
            NEAREST,
            DataType::I8,
            &[1, 0, 2, 1],
            &[3, 3],
            (false, false),
        ),
        resize_op(
            BILINEAR,
            DataType::I8,
            &[1, 2, 0, 1],
            &[3, 3],
            (false, false),
        ),
    ];
    // Wrong output shape, size computed at run time, size of another type or length, rank 3.
    let options = OpOptions::Resize {
        align_corners: false,
        half_pixel_centers: false,
    };
    let x = Some(t(0, DataType::I8, &[1, 2, 2, 1]));
    let size = Some(konst(t(1, DataType::I32, &[2]), &[3i32, 3]));
    refused.push(op(
        NEAREST,
        options,
        &[x, size],
        &[t(2, DataType::I8, &[1, 3, 4, 1])],
    ));
    refused.push(op(
        NEAREST,
        options,
        &[x, Some(t(1, DataType::I32, &[2]))],
        &[t(2, DataType::I8, &[1, 3, 3, 1])],
    ));
    refused.push(op(
        NEAREST,
        options,
        &[x, Some(konst(t(1, DataType::I64, &[2]), &[3i64, 3]))],
        &[t(2, DataType::I8, &[1, 3, 3, 1])],
    ));
    refused.push(op(
        NEAREST,
        options,
        &[x, Some(konst(t(1, DataType::I32, &[3]), &[3i32, 3, 1]))],
        &[t(2, DataType::I8, &[1, 3, 3, 1])],
    ));
    refused.push(op(
        BILINEAR,
        options,
        &[
            Some(t(0, DataType::I8, &[2, 2, 1])),
            Some(konst(t(1, DataType::I32, &[2]), &[3i32, 3])),
        ],
        &[t(2, DataType::I8, &[3, 3, 1])],
    ));
    // A mismatched output type.
    refused.push(op(
        BILINEAR,
        options,
        &[x, size],
        &[t(2, DataType::I16, &[1, 3, 3, 1])],
    ));
    for (i, op) in refused.iter().enumerate() {
        assert!(select(op).is_none(), "case {i}");
    }
    // An empty output is fine for nearest neighbor (nothing to compute).
    let empty = resize_op(
        NEAREST,
        DataType::I8,
        &[1, 2, 2, 1],
        &[0, 3],
        (false, false),
    );
    assert_eq!(run1::<i8>(&empty, &[buf(&[0i8; 4])]), []);
}
