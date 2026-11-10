//! Checks and parameter computations shared by the kernels, ported from TFLite Micro's
//! `kernel_util.cc`, `padding.h` and `quantization_util.cc`.

use core::ffi::c_void;

use nanoglia_cmsis_nn_sys as sys;
use nanoglia_core::le::LeSlice;
use nanoglia_core::quant::{self, Multiplier};
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

/// The largest magnitude of an int16 activation with zero point 0 (`-32768`).
const S16_MAX_INPUT: u128 = 1 << 15;

/// The largest magnitude of an int8 weight (`-128`), assumed for weights computed at run time.
const S8_MAX_WEIGHT: u128 = 1 << 7;

/// Whether CMSIS-NN's `arm_nn_requantize_s64` computes TFLite's 64-bit
/// `MultiplyByQuantizedMultiplier` for every accumulator of magnitude at most `max_acc`.
///
/// Workaround for CMSIS-NN v8.0.0, whose `arm_nn_requantize_s64` keeps
/// `(acc * reduced_multiplier) >> (14 - shift)` in an `int32_t` before its last rounding shift,
/// where TFLite keeps 64 bits: that intermediate value is twice the result, so it wraps once the
/// result reaches `2^30` in magnitude and the output saturates to the wrong side
/// (<https://github.com/AlixANNERAUD/nanoglia/issues/38>, fixed upstream by
/// <https://github.com/ARM-software/CMSIS-NN/pull/267>). Remove this bound, and the 16x8 kernels'
/// calls to [`s16_accumulators_exact`], once a CMSIS-NN release with the fix is vendored.
///
/// The product with the 16-bit multiplier must fit 62 bits (and the accumulator an `int64_t`), and
/// its shift by `14 - shift` an `int32_t` to which CMSIS-NN can add 1. Shifts of 8 or more are
/// refused (TFLite's 64-bit `MultiplyByQuantizedMultiplier` expects less).
pub(crate) fn s16_requantize_exact(max_acc: u128, m: Multiplier) -> bool {
    if m.shift >= 8 || max_acc >= 1 << 62 {
        return false;
    }
    // `REDUCE_MULTIPLIER`: the Q31 multiplier rounded to Q15.
    let reduced = if m.multiplier < 0x7fff_0000 {
        (i64::from(m.multiplier) + (1 << 15)) >> 16
    } else {
        0x7fff
    };
    let Ok(reduced) = u128::try_from(reduced) else {
        return false;
    };
    // CMSIS-NN computes `result = (int32_t)(product >> (14 - shift))`, then `(result + 1) >> 1`
    // in `int32_t`: both stay in range when `|product| < 2^(45 - shift) - 2^(14 - shift)`, so
    // that `product >> (14 - shift)` is at most `2^31 - 2`. `45 - shift` is at least 38 and
    // `14 - shift` at least 7 (shift < 8); the product must also fit 62 bits.
    let bits = (45 - i64::from(m.shift)).min(62) as u32;
    let low = (14 - i64::from(m.shift)).min(63) as u32;
    let limit = (1u128 << bits).saturating_sub(1u128 << low);
    max_acc
        .checked_mul(reduced)
        .is_some_and(|product| product < limit)
}

/// Whether CMSIS-NN requantizes every accumulator of a 16x8 layer like TFLite (see
/// [`s16_requantize_exact`]), from a bound on the accumulator of each output channel:
/// `2^15 × Σ|weight| + |bias|`, the input values being in `-32768..=32767` (zero point 0;
/// padding contributes 0).
///
/// Output channel `c` (of `channels`) has `taps` weights, at byte `weight_index(c, t)` of the
/// filter for `t` in `0..taps`, the bias (if any) `c` and the multiplier `multiplier(c)`. Weights
/// computed at run time count as `-128` each; a bias computed at run time cannot be bounded and
/// is refused.
pub(crate) fn s16_accumulators_exact(
    filter: &TensorDesc<'_>,
    bias: Option<&TensorDesc<'_>>,
    channels: usize,
    taps: usize,
    weight_index: impl Fn(usize, usize) -> usize,
    multiplier: impl Fn(usize) -> Multiplier,
) -> Option<()> {
    let bias = match bias {
        Some(b) => Some(LeSlice::<i64>::new(b.data?)),
        None => None,
    };
    let weights = filter.data;
    if let Some(weights) = weights
        && weights.len() != channels.checked_mul(taps)?
    {
        return None;
    }
    for c in 0..channels {
        let weight_sum = match weights {
            Some(weights) => (0..taps).try_fold(0u128, |sum, t| {
                let w = *weights.get(weight_index(c, t))? as i8;
                Some(sum + u128::from(w.unsigned_abs()))
            })?,
            None => u128::try_from(taps).ok()? * S8_MAX_WEIGHT,
        };
        let bias = match bias {
            Some(values) => values.get(c)?.unsigned_abs(),
            None => 0,
        };
        let max_acc = weight_sum * S16_MAX_INPUT + u128::from(bias);
        if !s16_requantize_exact(max_acc, multiplier(c)) {
            return None;
        }
    }
    Some(())
}
