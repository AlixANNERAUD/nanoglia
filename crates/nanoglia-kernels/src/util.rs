//! Helpers shared by the kernels: error constructors, operator checks, typed views of the
//! evaluation buffers, padding and broadcasting arithmetic.

use nanoglia_core::backend::KernelPlan;
use nanoglia_core::pod::{self, Pod, cast_slice, cast_slice_mut};
use nanoglia_core::{
    Activation, DataType, Detail, Error, MAX_RANK, OpDesc, Padding, Result, TensorDesc,
};

/// Declares a `#[repr(C)]` parameter struct and implements [`Pod`] for it, after checking at
/// compile time that it has no padding bytes.
macro_rules! pod_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $($(#[$fmeta:meta])* $fvis:vis $field:ident : $ty:ty),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug)]
        #[repr(C)]
        $vis struct $name {
            $($(#[$fmeta])* $fvis $field: $ty),*
        }

        const _: () = assert!(
            core::mem::size_of::<$name>() == 0 $(+ core::mem::size_of::<$ty>())*
        );

        // SAFETY: `repr(C)` without padding (checked above), made of integers, floats and arrays
        // of them, which accept any bit pattern; its alignment is at most 4.
        unsafe impl nanoglia_core::pod::Pod for $name {}
    };
}
pub(crate) use pod_struct;

/// Declares a kernel whose persistent parameters are a single [`Pod`] struct computed from the
/// operator by `$params` (which also validates the operator).
macro_rules! fixed_kernel {
    ($(#[$meta:meta])* $kernel:ident, $name:literal, $params:path, $eval:path) => {
        $(#[$meta])*
        pub(crate) static $kernel: nanoglia_core::backend::FnKernel = nanoglia_core::backend::FnKernel {
            name: $name,
            plan: |op| crate::util::plan_fixed(op, $params),
            prepare: |op, persist| crate::util::prepare_fixed(op, persist, $params),
            eval: $eval,
        };
    };
}
pub(crate) use fixed_kernel;

/// The plan of a kernel whose parameters are one `P` returned by `params`.
pub(crate) fn plan_fixed<P: Pod>(
    op: &OpDesc<'_, '_>,
    params: fn(&OpDesc<'_, '_>) -> Result<P>,
) -> Result<KernelPlan> {
    params(op)?;
    Ok(KernelPlan {
        persistent: core::mem::size_of::<P>(),
        scratch: 0,
    })
}

/// Writes the parameters returned by `params` at the start of `persist`.
pub(crate) fn prepare_fixed<P: Pod>(
    op: &OpDesc<'_, '_>,
    persist: &mut [u8],
    params: fn(&OpDesc<'_, '_>) -> Result<P>,
) -> Result<()> {
    pod::write(persist, params(op)?)
}

/// The operator is not supported by this kernel.
pub(crate) fn unsupported(op: &OpDesc<'_, '_>) -> Error {
    Error::unsupported(Detail::UnsupportedOp {
        op_index: op.index,
        opcode: op.opcode_number(),
    })
}

/// The type of `tensor` is not supported by this kernel.
pub(crate) fn unsupported_type(tensor: &TensorDesc<'_>) -> Error {
    Error::unsupported(Detail::UnsupportedType {
        tensor: tensor.index,
    })
}

/// `tensor` does not have the expected shape.
pub(crate) fn shape_mismatch(tensor: &TensorDesc<'_>) -> Error {
    Error::invalid_argument(Detail::ShapeMismatch {
        tensor: tensor.index,
    })
}

/// A buffer is smaller than expected.
pub(crate) fn size_mismatch(expected: usize, actual: usize) -> Error {
    let clamp = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
    Error::invalid_argument(Detail::SizeMismatch {
        expected: clamp(expected),
        actual: clamp(actual),
    })
}

/// Input `i` of `op`, or an error if it is omitted.
pub(crate) fn input<'o, 'a>(op: &'o OpDesc<'a, '_>, i: usize) -> Result<&'o TensorDesc<'a>> {
    op.input(i).ok_or_else(|| unsupported(op))
}

/// Output `i` of `op`.
pub(crate) fn output<'o, 'a>(op: &'o OpDesc<'a, '_>, i: usize) -> Result<&'o TensorDesc<'a>> {
    op.output(i).ok_or_else(|| unsupported(op))
}

/// Fails unless `op` has exactly `inputs` inputs (counting omitted ones) and `outputs` outputs.
pub(crate) fn expect_arity(op: &OpDesc<'_, '_>, inputs: usize, outputs: usize) -> Result<()> {
    if op.inputs.len() == inputs && op.outputs.len() == outputs {
        Ok(())
    } else {
        Err(unsupported(op))
    }
}

/// Fails unless `tensor` has type `dtype`.
pub(crate) fn expect_type(tensor: &TensorDesc<'_>, dtype: DataType) -> Result<()> {
    if tensor.dtype == dtype {
        Ok(())
    } else {
        Err(unsupported_type(tensor))
    }
}

/// The number of elements of `tensor`, which must fit in an `i32` like in TFLite.
pub(crate) fn elements(tensor: &TensorDesc<'_>) -> Result<usize> {
    match tensor.dims.num_elements() {
        Some(n) if i32::try_from(n).is_ok() && tensor.byte_len().is_some() => Ok(n),
        _ => Err(shape_mismatch(tensor)),
    }
}

/// The product of `dims` as a `u32`, or a shape error for `tensor` if it does not fit (a tensor
/// with a zero dimension may have huge other ones).
pub(crate) fn dims_product(dims: &[i32], tensor: &TensorDesc<'_>) -> Result<u32> {
    dims.iter()
        .try_fold(1u32, |acc, &d| acc.checked_mul(u32::try_from(d).ok()?))
        .ok_or_else(|| shape_mismatch(tensor))
}

/// Fails unless `a` and `b` have the same number of elements; returns it.
pub(crate) fn same_elements(a: &TensorDesc<'_>, b: &TensorDesc<'_>) -> Result<usize> {
    let n = elements(a)?;
    if elements(b)? != n {
        return Err(shape_mismatch(b));
    }
    Ok(n)
}

/// Fails unless `a` and `b` have exactly the same shape; returns the number of elements.
pub(crate) fn same_shape(a: &TensorDesc<'_>, b: &TensorDesc<'_>) -> Result<usize> {
    if a.dims != b.dims {
        return Err(shape_mismatch(b));
    }
    elements(a)
}

/// The constant contents of `tensor`, or an error if it is not a constant.
pub(crate) fn constant<'a>(op: &OpDesc<'a, '_>, tensor: &TensorDesc<'a>) -> Result<&'a [u8]> {
    match tensor.data {
        Some(data) if Some(data.len()) >= tensor.byte_len() => Ok(data),
        Some(_) => Err(shape_mismatch(tensor)),
        None => Err(unsupported(op)),
    }
}

/// Fails if `tensor` is a constant whose data is not aligned for elements of `align` bytes.
pub(crate) fn check_aligned(tensor: &TensorDesc<'_>, align: usize) -> Result<()> {
    match tensor.data {
        Some(data) if data.as_ptr().align_offset(align) != 0 => {
            Err(Error::invalid_argument(Detail::Misaligned {
                tensor: tensor.index,
            }))
        }
        _ => Ok(()),
    }
}

/// Reads element `i` of a constant integer tensor (`I32` or `I64`) as `i64`.
pub(crate) fn read_index(tensor: &TensorDesc<'_>, data: &[u8], i: usize) -> Option<i64> {
    match tensor.dtype {
        DataType::I32 => nanoglia_core::le::read::<i32>(data, i * 4).map(i64::from),
        DataType::I64 => nanoglia_core::le::read::<i64>(data, i * 8),
        _ => None,
    }
}

/// The per-tensor scale and zero point of `tensor`; fails if it is not per-tensor quantized or if
/// the scale is not a positive finite number.
pub(crate) fn per_tensor(tensor: &TensorDesc<'_>) -> Result<(f32, i32)> {
    let q = &tensor.quant;
    let scale = q.scale();
    if !q.is_per_tensor()
        || q.zero_point.len() > 1
        || !(scale > 0.0 && scale.is_finite())
        || q.zero_point
            .get(0)
            .is_some_and(|z| i32::try_from(z).is_err())
    {
        return Err(unsupported_type(tensor));
    }
    Ok((scale, q.zero_point()))
}

/// The value range of a quantized integer type.
pub(crate) fn int_range(dtype: DataType) -> Option<(i32, i32)> {
    Some(match dtype {
        DataType::I8 => (i8::MIN.into(), i8::MAX.into()),
        DataType::U8 => (u8::MIN.into(), u8::MAX.into()),
        DataType::I16 => (i16::MIN.into(), i16::MAX.into()),
        _ => return None,
    })
}

/// Encodes a [`DataType`] for storage in parameters.
pub(crate) fn type_code(dtype: DataType) -> u32 {
    dtype as u32
}

/// The float clamping range of a fused activation, or an error for activations TFLite does not
/// fuse into float kernels.
pub(crate) fn float_activation(op: &OpDesc<'_, '_>, activation: Activation) -> Result<(f32, f32)> {
    match activation {
        Activation::None | Activation::Relu | Activation::Relu6 | Activation::ReluN1To1 => {
            Ok(nanoglia_core::quant::activation_range_f32(activation))
        }
        _ => Err(unsupported(op)),
    }
}

/// TFLite `CalculateActivationRangeQuantized`: the clamping range of a fused activation for an
/// `output` of integer type, quantized with `(scale, zero_point)`. Fails like TFLite when a bound
/// does not fit an `i32`, and for activations TFLite does not fuse into quantized kernels.
pub(crate) fn quantized_activation(
    op: &OpDesc<'_, '_>,
    activation: Activation,
    output: &TensorDesc<'_>,
    (scale, zero_point): (f32, i32),
) -> Result<(i32, i32)> {
    let bounds: &[f32] = match activation {
        Activation::None => &[],
        Activation::Relu => &[0.0],
        Activation::Relu6 => &[0.0, 6.0],
        Activation::ReluN1To1 => &[-1.0, 1.0],
        _ => return Err(unsupported(op)),
    };
    for &bound in bounds {
        let q = nanoglia_core::quant::round_f32(bound / scale);
        if !(q >= i32::MIN as f32 && q <= i32::MAX as f32) {
            return Err(unsupported(op));
        }
    }
    let (qmin, qmax) = int_range(output.dtype).ok_or_else(|| unsupported_type(output))?;
    Ok(nanoglia_core::quant::activation_range_quantized(
        activation, scale, zero_point, qmin, qmax,
    ))
}

/// `std::max(a, b)` of C++: `a < b ? b : a`.
#[inline(always)]
pub(crate) fn cmax(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

/// `std::min(a, b)` of C++: `b < a ? b : a`.
#[inline(always)]
pub(crate) fn cmin(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

/// TFLite `ActivationFunctionWithMinMax`: `min(max(x, lo), hi)`.
#[inline(always)]
pub(crate) fn clamp_f32(x: f32, lo: f32, hi: f32) -> f32 {
    cmin(cmax(x, lo), hi)
}

/// TFLite `ComputeOutSize`.
pub(crate) fn compute_out_size(
    padding: Padding,
    image: i32,
    filter: i32,
    stride: i32,
    dilation: i32,
) -> i64 {
    let effective = (i64::from(filter) - 1) * i64::from(dilation) + 1;
    let (image, stride) = (i64::from(image), i64::from(stride));
    if stride == 0 {
        return 0;
    }
    match padding {
        Padding::Same => (image + stride - 1) / stride,
        Padding::Valid => (image + stride - effective) / stride,
    }
}

/// TFLite `ComputePaddingWithOffset`: the padding before the image (the offset is not used by
/// the reference kernels).
pub(crate) fn compute_padding(
    stride: i32,
    dilation: i32,
    image: i32,
    filter: i32,
    out: i32,
) -> i32 {
    let effective = (i64::from(filter) - 1) * i64::from(dilation) + 1;
    let total = (i64::from(out) - 1) * i64::from(stride) + effective - i64::from(image);
    (total.max(0) / 2) as i32
}

/// Spatial geometry of a 2D window operator (convolution or pooling), as TFLite computes it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Window {
    pub pad_h: i32,
    pub pad_w: i32,
}

/// Checks the output size of a 2D window operator and computes its padding
/// (TFLite `ComputePaddingHeightWidth`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn window(
    output: &TensorDesc<'_>,
    padding: Padding,
    (in_h, in_w): (i32, i32),
    (filter_h, filter_w): (i32, i32),
    (stride_h, stride_w): (i32, i32),
    (dilation_h, dilation_w): (i32, i32),
    (out_h, out_w): (i32, i32),
) -> Result<Window> {
    if stride_h <= 0
        || stride_w <= 0
        || dilation_h <= 0
        || dilation_w <= 0
        || filter_h <= 0
        || filter_w <= 0
    {
        return Err(shape_mismatch(output));
    }
    let expected_h = compute_out_size(padding, in_h, filter_h, stride_h, dilation_h);
    let expected_w = compute_out_size(padding, in_w, filter_w, stride_w, dilation_w);
    if i64::from(out_h) != expected_h || i64::from(out_w) != expected_w {
        return Err(shape_mismatch(output));
    }
    Ok(Window {
        pad_h: compute_padding(stride_h, dilation_h, in_h, filter_h, out_h),
        pad_w: compute_padding(stride_w, dilation_w, in_w, filter_w, out_w),
    })
}

/// The dimensions of a rank-4 tensor as `i32`.
pub(crate) fn dims4(tensor: &TensorDesc<'_>) -> Result<[i32; 4]> {
    match tensor.dims.as_slice() {
        &[a, b, c, d] if a >= 0 && b >= 0 && c >= 0 && d >= 0 => {
            elements(tensor)?;
            Ok([a, b, c, d])
        }
        _ => Err(shape_mismatch(tensor)),
    }
}

pod_struct! {
    /// How two inputs broadcast to an output of up to [`MAX_RANK`] dimensions (TFLite
    /// `BroadcastBinaryFunction6DSlow` semantics).
    pub(crate) struct Broadcast {
        /// Output dimensions, left-padded with ones.
        out: [u32; MAX_RANK],
        /// Element strides of the first input along each output dimension (0 when broadcast).
        stride1: [u32; MAX_RANK],
        /// Element strides of the second input along each output dimension (0 when broadcast).
        stride2: [u32; MAX_RANK],
        /// Number of elements of the first input.
        pub len1: u32,
        /// Number of elements of the second input.
        pub len2: u32,
        /// Number of elements of the output.
        pub len: u32,
        /// Whether both inputs have the output's number of elements (no broadcasting).
        flat: u32,
    }
}

impl Broadcast {
    /// Computes how `a` and `b` broadcast to `out`; fails if `out` is not their broadcast shape.
    pub(crate) fn new(
        a: &TensorDesc<'_>,
        b: &TensorDesc<'_>,
        out: &TensorDesc<'_>,
    ) -> Result<Self> {
        let len1 = elements(a)?;
        let len2 = elements(b)?;
        let len = elements(out)?;
        let rank = out.dims.rank();
        if a.dims.rank() > rank || b.dims.rank() > rank {
            return Err(shape_mismatch(out));
        }
        let pad = |t: &TensorDesc<'_>| {
            let mut d = [1i32; MAX_RANK];
            let s = t.dims.as_slice();
            d[MAX_RANK - s.len()..].copy_from_slice(s);
            d
        };
        let (da, db, dout) = (pad(a), pad(b), pad(out));
        let mut params = Self {
            out: [1; MAX_RANK],
            stride1: [0; MAX_RANK],
            stride2: [0; MAX_RANK],
            len1: len1 as u32,
            len2: len2 as u32,
            len: len as u32,
            flat: u32::from(a.dims.as_slice() == b.dims.as_slice() && len1 == len),
        };
        let (mut s1, mut s2) = (1u32, 1u32);
        for axis in (0..MAX_RANK).rev() {
            let (x, y, o) = (da[axis], db[axis], dout[axis]);
            let ok = |d: i32| d == o || d == 1;
            if !ok(x) || !ok(y) || (o != x && o != y) {
                return Err(shape_mismatch(out));
            }
            params.out[axis] = o as u32;
            params.stride1[axis] = if x == 1 { 0 } else { s1 };
            params.stride2[axis] = if y == 1 { 0 } else { s2 };
            s1 *= x as u32;
            s2 *= y as u32;
        }
        Ok(params)
    }

    /// Calls `f(output_index, input1_index, input2_index)` for every output element, in order.
    #[inline]
    pub(crate) fn for_each(&self, mut f: impl FnMut(usize, usize, usize)) {
        if self.flat != 0 {
            for i in 0..self.len as usize {
                f(i, i, i);
            }
            return;
        }
        const OUTER: usize = MAX_RANK - 1;
        let inner = self.out[OUTER] as usize;
        let (inner1, inner2) = (self.stride1[OUTER] as usize, self.stride2[OUTER] as usize);
        let outer: usize = self.out[..OUTER].iter().map(|&d| d as usize).product();
        let mut index = [0u32; OUTER];
        let mut o = 0;
        for _ in 0..outer {
            let (mut base1, mut base2) = (0usize, 0usize);
            for ((&i, &s1), &s2) in index.iter().zip(&self.stride1).zip(&self.stride2) {
                base1 += i as usize * s1 as usize;
                base2 += i as usize * s2 as usize;
            }
            for j in 0..inner {
                f(o, base1 + j * inner1, base2 + j * inner2);
                o += 1;
            }
            for axis in (0..OUTER).rev() {
                index[axis] += 1;
                if index[axis] < self.out[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
    }
}

/// Input `i` of an evaluation as at least `len` elements of `T` (only the first `len` are
/// returned).
pub(crate) fn in_slice<'d, T: Pod>(inputs: &[&'d [u8]], i: usize, len: usize) -> Result<&'d [T]> {
    let bytes = inputs.get(i).copied().unwrap_or(&[]);
    let need = len * core::mem::size_of::<T>();
    let bytes = bytes
        .get(..need)
        .ok_or_else(|| size_mismatch(need, bytes.len()))?;
    cast_slice(bytes)
}

/// Output `i` of an evaluation as at least `len` elements of `T` (only the first `len` are
/// returned).
pub(crate) fn out_slice<'a, T: Pod>(
    outputs: &'a mut [&mut [u8]],
    i: usize,
    len: usize,
) -> Result<&'a mut [T]> {
    let need = len * core::mem::size_of::<T>();
    let bytes: &'a mut [u8] = match outputs.get_mut(i) {
        Some(out) => out,
        None => return Err(size_mismatch(need, 0)),
    };
    let actual = bytes.len();
    let bytes = bytes
        .get_mut(..need)
        .ok_or_else(|| size_mismatch(need, actual))?;
    cast_slice_mut(bytes)
}
