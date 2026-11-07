//! Host tests: operators are described by hand and run through `select`, `plan`, `prepare` and
//! `eval` with test buffers.

use crate::BACKEND;
use nanoglia_core::backend::{Backend, EvalCtx, Kernel};
use nanoglia_core::le::LeSlice;
use nanoglia_core::pod::{Pod, cast_slice, cast_slice_mut};
use nanoglia_core::{
    BuiltinOp, DataType, Detail, Dims, Error, OpCode, OpDesc, OpOptions, Quantization, Result,
    TensorDesc,
};
use std::boxed::Box;
use std::vec;
use std::vec::Vec;

mod arg_min_max_tests;
mod gather_tests;
mod golden;
mod golden_tests;
mod hand_tests;
mod reduce_tests;
mod resize_tests;
mod slice_tests;
mod split_tests;
mod transpose_conv_tests;

/// Sixteen bytes aligned like the interpreter's arena.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Chunk([u8; 16]);

/// A zeroed, 16-byte aligned buffer that lives until the end of the tests.
fn zeroed(len: usize) -> &'static mut [u8] {
    let chunks: &'static mut [Chunk] =
        Box::leak(vec![Chunk([0; 16]); len.div_ceil(16).max(1)].into_boxed_slice());
    // SAFETY: `Chunk` is 16 plain bytes without padding, and the slice holds at least `len` bytes.
    unsafe { core::slice::from_raw_parts_mut(chunks.as_mut_ptr().cast::<u8>(), len) }
}

/// An aligned buffer holding `values`.
pub(crate) fn buf<T: Pod>(values: &[T]) -> &'static mut [u8] {
    let bytes = zeroed(core::mem::size_of_val(values));
    cast_slice_mut::<T>(bytes).unwrap().copy_from_slice(values);
    bytes
}

/// The contents of `bytes` as `T`s.
pub(crate) fn view<T: Pod>(bytes: &[u8]) -> Vec<T> {
    cast_slice::<T>(bytes).unwrap().to_vec()
}

/// `f32` values from their bit patterns.
pub(crate) fn floats(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

/// Every `i8` value in increasing order.
pub(crate) fn all_i8() -> Vec<i8> {
    (-128..=127).collect()
}

/// A non-quantized tensor.
pub(crate) fn t(index: u32, dtype: DataType, dims: &[i32]) -> TensorDesc<'static> {
    TensorDesc {
        index,
        dtype,
        dims: Dims::new(dims).unwrap(),
        quant: Quantization::NONE,
        data: None,
        name: "",
    }
}

/// `tensor` with per-tensor quantization.
pub(crate) fn q(tensor: TensorDesc<'static>, scale: f32, zero_point: i64) -> TensorDesc<'static> {
    qc(tensor, &[scale], &[zero_point], 0)
}

/// `tensor` with per-axis quantization.
pub(crate) fn qc(
    mut tensor: TensorDesc<'static>,
    scales: &[f32],
    zero_points: &[i64],
    axis: i32,
) -> TensorDesc<'static> {
    let scale: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let zero_point: Vec<u8> = zero_points.iter().flat_map(|z| z.to_le_bytes()).collect();
    tensor.quant = Quantization {
        scale: LeSlice::new(Box::leak(scale.into_boxed_slice())),
        zero_point: LeSlice::new(Box::leak(zero_point.into_boxed_slice())),
        axis,
    };
    tensor
}

/// `tensor` as a constant holding `values`.
pub(crate) fn konst<T: Pod>(mut tensor: TensorDesc<'static>, values: &[T]) -> TensorDesc<'static> {
    tensor.data = Some(buf(values));
    tensor
}

/// An operator description.
pub(crate) fn op(
    code: BuiltinOp,
    options: OpOptions<'static>,
    inputs: &[Option<TensorDesc<'static>>],
    outputs: &[TensorDesc<'static>],
) -> OpDesc<'static, 'static> {
    OpDesc {
        index: 7,
        code: OpCode::Builtin(code),
        options,
        inputs: Box::leak(inputs.to_vec().into_boxed_slice()),
        outputs: Box::leak(outputs.to_vec().into_boxed_slice()),
    }
}

/// The kernel the backend selects for `op`.
pub(crate) fn select(op: &OpDesc<'_, '_>) -> Option<&'static dyn Kernel> {
    BACKEND.select(op)
}

/// Runs `op`: constant inputs come from their tensors, the others from `activations` in order.
/// Returns the outputs.
pub(crate) fn run(op: &OpDesc<'_, '_>, activations: &[&[u8]]) -> Result<Vec<&'static mut [u8]>> {
    let kernel = select(op).ok_or(Error::unsupported(Detail::None))?;
    let plan = kernel.plan(op)?;
    let persist = zeroed(plan.persistent);
    kernel.prepare(op, persist)?;
    let scratch = zeroed(plan.scratch);
    // Garbage in the scratch buffer must not matter.
    scratch.fill(0xa5);

    let mut next = activations.iter();
    let inputs: Vec<&[u8]> = op
        .inputs
        .iter()
        .map(|t| match t {
            Some(TensorDesc { data: Some(d), .. }) => &**d,
            Some(_) => &*buf::<u8>(next.next().expect("missing activation")),
            None => &[][..],
        })
        .collect();
    let mut outputs: Vec<&'static mut [u8]> = op
        .outputs
        .iter()
        .map(|t| {
            let out = zeroed(t.byte_len().unwrap());
            out.fill(0x5a);
            out
        })
        .collect();
    let mut views: Vec<&mut [u8]> = outputs.iter_mut().map(|o| &mut **o).collect();
    let mut ctx = EvalCtx {
        inputs: &inputs,
        outputs: &mut views,
        persist,
        scratch,
    };
    kernel.eval(&mut ctx)?;
    Ok(outputs)
}

/// Runs a single-output operator and returns its output as `T`s.
pub(crate) fn run1<T: Pod>(op: &OpDesc<'_, '_>, activations: &[&[u8]]) -> Vec<T> {
    view(run(op, activations).unwrap()[0])
}

/// Asserts that two `f32` slices are bit-identical, reporting the first difference.
#[track_caller]
pub(crate) fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "element {i}: {a} != {e}");
    }
}

/// Asserts that two `f32` slices are equal within `tolerance`.
#[track_caller]
pub(crate) fn assert_near(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tolerance, "element {i}: {a} != {e}");
    }
}
