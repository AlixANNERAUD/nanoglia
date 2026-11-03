//! Checks and parameter computations shared by the kernels, ported from TFLite Micro's
//! `kernel_util.cc`, `padding.h` and `quantization_util.cc`.

use core::ffi::c_void;

use nanoglia_cmsis_nn_sys as sys;
use nanoglia_core::quant;
use nanoglia_core::{
    Activation, DataType, Detail, Dims, Error, OpDesc, Padding, Result, TensorDesc,
};

/// The backend name used in error details.
pub(crate) const NAME: &str = "cmsis-nn";

/// The error returned by `plan`/`prepare` for an operator that `select` would have refused.
pub(crate) fn unsupported(op: &OpDesc<'_, '_>) -> Error {
    Error::unsupported(Detail::UnsupportedOp {
        op_index: op.index,
        opcode: op.opcode_number(),
    })
}

/// Maps a CMSIS-NN status to a result.
pub(crate) fn check_status(status: sys::arm_cmsis_nn_status) -> Result<()> {
    if status == sys::ARM_CMSIS_NN_SUCCESS {
        Ok(())
    } else {
        Err(Error::backend(NAME, status))
    }
}

/// The `[qmin, qmax]` range of a quantized type.
pub(crate) fn type_range(dtype: DataType) -> Option<(i32, i32)> {
    match dtype {
        DataType::I8 => Some((i8::MIN as i32, i8::MAX as i32)),
        DataType::I16 => Some((i16::MIN as i32, i16::MAX as i32)),
        _ => None,
    }
}

/// The per-tensor quantization of an activation tensor of type `dtype`: `(scale, zero_point)`.
///
/// The scale must be finite and positive and the zero point inside the range of the type (CMSIS-NN
/// takes `-zero_point` as an input offset in `[-127, 128]`).
pub(crate) fn activation_quant(t: &TensorDesc<'_>, dtype: DataType) -> Option<(f32, i32)> {
    if t.dtype != dtype || !t.quant.is_per_tensor() {
        return None;
    }
    let scale = t.quant.scale();
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let zero_point = match t.quant.zero_point.len() {
        0 => 0,
        _ => i32::try_from(t.quant.zero_point.get(0)?).ok()?,
    };
    let (qmin, qmax) = type_range(dtype)?;
    (qmin..=qmax)
        .contains(&zero_point)
        .then_some((scale, zero_point))
}

/// Checks the quantization of a weight tensor with `channels` output channels along `axis`:
/// per-tensor, or per-channel along `axis`, with finite non-negative scales and zero zero points.
pub(crate) fn weight_quant(t: &TensorDesc<'_>, channels: usize, axis: i32) -> Option<()> {
    let scales = t.quant.scale;
    let per_channel = match scales.len() {
        1 => false,
        n if n == channels && n > 1 => true,
        _ => return None,
    };
    if per_channel && t.quant.axis != axis {
        return None;
    }
    if !scales.iter().all(|s| s.is_finite() && s >= 0.0) {
        return None;
    }
    t.quant.zero_point.iter().all(|z| z == 0).then_some(())
}

/// The scale of output channel `channel` of a weight tensor checked by [`weight_quant`].
pub(crate) fn weight_scale(t: &TensorDesc<'_>, channel: usize) -> f32 {
    let scales = t.quant.scale;
    if scales.len() == 1 {
        scales.get(0).unwrap_or(0.0)
    } else {
        scales.get(channel).unwrap_or(0.0)
    }
}

/// The dimensions of a rank-4 tensor, each in `1..=u16::MAX` (CMSIS-NN keeps some of them in
/// `uint16_t` variables).
pub(crate) fn dims4(t: &TensorDesc<'_>) -> Option<[i32; 4]> {
    let d = t.dims.as_slice();
    if d.len() != 4 || !d.iter().all(|&x| (1..=u16::MAX as i32).contains(&x)) {
        return None;
    }
    t.dims.num_elements().filter(|&n| n <= i32::MAX as usize)?;
    Some([d[0], d[1], d[2], d[3]])
}

/// The number of elements of `dims`, if all dimensions are positive and the count fits an `i32`.
pub(crate) fn element_count(dims: &Dims) -> Option<i32> {
    if dims.as_slice().iter().any(|&d| d <= 0) {
        return None;
    }
    let n = dims.num_elements()?;
    i32::try_from(n).ok()
}

/// The fused activations CMSIS-NN can apply as a clamp.
pub(crate) fn clamp_activation(activation: Activation) -> Option<Activation> {
    match activation {
        Activation::None | Activation::Relu | Activation::Relu6 | Activation::ReluN1To1 => {
            Some(activation)
        }
        _ => None,
    }
}

/// TFLite Micro `CalculateActivationRangeQuantized` for an output of type `dtype`, failing like
/// its `Quantize` helper when a bound does not fit an `i32`.
pub(crate) fn activation_range(
    activation: Activation,
    dtype: DataType,
    scale: f32,
    zero_point: i32,
) -> Option<(i32, i32)> {
    let activation = clamp_activation(activation)?;
    let bounds: &[f32] = match activation {
        Activation::Relu => &[0.0],
        Activation::Relu6 => &[0.0, 6.0],
        Activation::ReluN1To1 => &[-1.0, 1.0],
        _ => &[],
    };
    for &bound in bounds {
        let q = quant::round_f32(bound / scale);
        if !(q >= i32::MIN as f32 && q <= i32::MAX as f32) {
            return None;
        }
    }
    let (qmin, qmax) = type_range(dtype)?;
    let (min, max) = quant::activation_range_quantized(activation, scale, zero_point, qmin, qmax);
    (min <= max).then_some((min, max))
}

/// TFLite `ComputeOutSize` and `ComputePaddingWithOffset` for one spatial dimension: the output
/// size and the padding before the data (the extra padding after it, when the total is odd, is
/// implied).
pub(crate) fn output_size_and_padding(
    padding: Padding,
    in_size: i32,
    filter_size: i32,
    stride: i32,
    dilation: i32,
) -> Option<(i32, i32)> {
    if in_size < 0 || filter_size <= 0 || stride <= 0 || dilation <= 0 {
        return None;
    }
    let effective_filter = (i64::from(filter_size) - 1) * i64::from(dilation) + 1;
    let (in_size, stride) = (i64::from(in_size), i64::from(stride));
    let out = match padding {
        Padding::Same => (in_size + stride - 1) / stride,
        Padding::Valid => (in_size + stride - effective_filter) / stride,
    };
    if out < 0 {
        return None;
    }
    let total = ((out - 1) * stride + effective_filter - in_size).max(0);
    Some((i32::try_from(out).ok()?, i32::try_from(total / 2).ok()?))
}

/// Converts a CMSIS-NN buffer size to bytes (negative sizes are refused).
pub(crate) fn buffer_size(size: i32) -> Option<i32> {
    (size >= 0).then_some(size)
}

/// The bytes of input `i` (empty if omitted), with the lifetime of the buffers rather than of
/// the context.
pub(crate) fn input<'d>(inputs: &[&'d [u8]], i: usize) -> &'d [u8] {
    inputs.get(i).copied().unwrap_or(&[])
}

/// Checks that a buffer has exactly `expected` bytes.
pub(crate) fn check_len(bytes: &[u8], expected: i32) -> Result<()> {
    if bytes.len() as u64 == expected as u64 {
        Ok(())
    } else {
        Err(Error::invalid_argument(Detail::SizeMismatch {
            expected: expected as u32,
            actual: u32::try_from(bytes.len()).unwrap_or(u32::MAX),
        }))
    }
}

/// An `i32` buffer (bias) of exactly `expected` elements, or a null pointer when absent.
pub(crate) fn optional_i32(bytes: &[u8], present: bool, expected: i32) -> Result<*const i32> {
    if !present {
        return Ok(core::ptr::null());
    }
    check_len(bytes, expected.saturating_mul(4))?;
    Ok(nanoglia_core::pod::cast_slice::<i32>(bytes)?.as_ptr())
}

/// The scratch buffer as a CMSIS-NN context: checks its size and its 4-byte alignment (the kernels
/// use it as `int16_t`/`int32_t` arrays). An empty scratch gives a null buffer.
pub(crate) fn scratch_context(scratch: &mut [u8], size: i32) -> Result<sys::cmsis_nn_context> {
    if size <= 0 {
        return Ok(sys::cmsis_nn_context {
            buf: core::ptr::null_mut(),
            size: 0,
        });
    }
    if (scratch.len() as u64) < size as u64 {
        return Err(Error::invalid_argument(Detail::SizeMismatch {
            expected: size as u32,
            actual: u32::try_from(scratch.len()).unwrap_or(u32::MAX),
        }));
    }
    if scratch.as_ptr().align_offset(4) != 0 {
        return Err(Error::invalid_argument(Detail::Misaligned {
            tensor: u32::MAX,
        }));
    }
    Ok(sys::cmsis_nn_context {
        buf: scratch.as_mut_ptr().cast::<c_void>(),
        size,
    })
}

/// A `cmsis_nn_dims` from NHWC values.
pub(crate) const fn nhwc(n: i32, h: i32, w: i32, c: i32) -> sys::cmsis_nn_dims {
    sys::cmsis_nn_dims { n, h, w, c }
}

/// The number of elements described by a `cmsis_nn_dims` (all dimensions were checked positive
/// and their product to fit an `i32` before the parameters were written).
pub(crate) fn dims_len(d: &sys::cmsis_nn_dims) -> i32 {
    d.n.saturating_mul(d.h)
        .saturating_mul(d.w)
        .saturating_mul(d.c)
}
