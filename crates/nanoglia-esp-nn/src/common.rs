//! Checks and parameter computations shared by the kernels, ported from TFLite Micro's
//! `kernel_util.cc`, `padding.h` and `quantization_util.cc` (the same code as in the CMSIS-NN
//! backend: both must accept the same operators with the same parameters).

use core::ffi::c_void;

use nanoglia_core::quant;
use nanoglia_core::{
    ALIGN, Activation, DataType, Detail, Dims, Error, OpDesc, Padding, Result, TensorDesc,
};

/// The backend name used in error details.
pub(crate) const NAME: &str = "esp-nn";

/// The error returned by `plan`/`prepare` for an operator that `select` would have refused.
pub(crate) fn unsupported(op: &OpDesc<'_, '_>) -> Error {
    Error::unsupported(Detail::UnsupportedOp {
        op_index: op.index,
        opcode: op.opcode_number(),
    })
}

/// The per-tensor quantization of an int8 activation tensor: `(scale, zero_point)`, with a finite
/// positive scale and a zero point in `[-128, 127]`.
pub(crate) fn activation_quant(t: &TensorDesc<'_>) -> Option<(f32, i32)> {
    if t.dtype != DataType::I8 || !t.quant.is_per_tensor() {
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
    (i8::MIN as i32..=i8::MAX as i32)
        .contains(&zero_point)
        .then_some((scale, zero_point))
}

/// Checks the quantization of a weight tensor with `channels` output channels along `axis`:
/// per-tensor, or per-channel along `axis`, with finite non-negative scales and zero zero points.
/// Returns whether it is per-channel.
pub(crate) fn weight_quant(t: &TensorDesc<'_>, channels: usize, axis: i32) -> Option<bool> {
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
    t.quant
        .zero_point
        .iter()
        .all(|z| z == 0)
        .then_some(per_channel)
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

/// The dimensions of a rank-4 tensor, each in `1..=u16::MAX` (ESP-NN keeps them in `uint16_t`
/// variables), with an element count that fits an `i32`.
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

/// TFLite Micro `CalculateActivationRangeQuantized` for an int8 output, failing like its
/// `Quantize` helper when a bound does not fit an `i32`. Only the activations a clamp implements.
pub(crate) fn activation_range(
    activation: Activation,
    scale: f32,
    zero_point: i32,
) -> Option<(i32, i32)> {
    let bounds: &[f32] = match activation {
        Activation::None => &[],
        Activation::Relu => &[0.0],
        Activation::Relu6 => &[0.0, 6.0],
        Activation::ReluN1To1 => &[-1.0, 1.0],
        _ => return None,
    };
    for &bound in bounds {
        let q = quant::round_f32(bound / scale);
        if !(q >= i32::MIN as f32 && q <= i32::MAX as f32) {
            return None;
        }
    }
    let (min, max) = quant::activation_range_quantized(
        activation,
        scale,
        zero_point,
        i8::MIN as i32,
        i8::MAX as i32,
    );
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

/// Whether a stride, filter size or padding fits the `uint16_t` parameters of ESP-NN (and the
/// `int16_t` arithmetic some of its kernels do on them).
pub(crate) fn fits_u16(values: &[i32]) -> bool {
    values.iter().all(|&v| (0..=i16::MAX as i32).contains(&v))
}

/// Converts an ESP-NN scratch size to bytes (negative sizes are refused).
pub(crate) fn buffer_size(size: i32) -> Option<usize> {
    usize::try_from(size).ok()
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

/// The scratch buffer to hand to ESP-NN: checks its size and its [`ALIGN`]-byte alignment (the
/// optimized kernels use it for SIMD loads). An empty scratch gives a null pointer.
pub(crate) fn scratch_ptr(scratch: &mut [u8], size: i32) -> Result<*mut c_void> {
    if size <= 0 {
        return Ok(core::ptr::null_mut());
    }
    if (scratch.len() as u64) < size as u64 {
        return Err(Error::invalid_argument(Detail::SizeMismatch {
            expected: size as u32,
            actual: u32::try_from(scratch.len()).unwrap_or(u32::MAX),
        }));
    }
    if scratch.as_ptr().align_offset(ALIGN) != 0 {
        return Err(Error::invalid_argument(Detail::Misaligned {
            tensor: u32::MAX,
        }));
    }
    Ok(scratch.as_mut_ptr().cast::<c_void>())
}

/// `data_dims_t` of one batch of an NHWC tensor.
pub(crate) const fn dims(
    width: i32,
    height: i32,
    channels: i32,
    extra: i32,
) -> nanoglia_esp_nn_sys::data_dims_t {
    nanoglia_esp_nn_sys::data_dims_t {
        width,
        height,
        channels,
        extra,
    }
}

/// The number of elements of one batch described by a `data_dims_t` (all dimensions were checked
/// positive and their product to fit an `i32` before the parameters were written).
pub(crate) fn dims_len(d: &nanoglia_esp_nn_sys::data_dims_t) -> i32 {
    d.width.saturating_mul(d.height).saturating_mul(d.channels)
}

/// The element counts of one batch of an input and of an output, both positive (else the
/// persistent buffer is inconsistent).
pub(crate) fn batch_lens(
    input: &nanoglia_esp_nn_sys::data_dims_t,
    output: &nanoglia_esp_nn_sys::data_dims_t,
) -> Result<(usize, usize)> {
    let input = usize::try_from(dims_len(input)).unwrap_or(0);
    let output = usize::try_from(dims_len(output)).unwrap_or(0);
    if input == 0 || output == 0 {
        return Err(crate::layout::inconsistent());
    }
    Ok((input, output))
}
