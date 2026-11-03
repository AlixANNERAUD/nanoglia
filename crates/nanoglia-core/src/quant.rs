//! These functions reproduce TensorFlow Lite's reference implementation bit for bit
//! (`tensorflow/lite/kernels/internal/quantization_util.cc` and `common.h`).
//!
//! TFLite has two requantization modes: *double rounding* (gemmlowp's
//! `SaturatingRoundingDoublingHighMul` then `RoundingDivideByPOT`), the default of TensorFlow
//! Lite Micro, CMSIS-NN and the LiteRT reference kernels, and *single rounding*
//! (`TFLITE_SINGLE_ROUNDING`, one rounding of the 64-bit product, slightly faster). They differ
//! by one unit on rare ties. nanoglia uses double rounding; the `single-rounding` feature switches
//! every crate to the other mode.

use crate::Activation;

/// Splits `x` into a mantissa in `[0.5, 1)` and a power of two, like C's `frexp`.
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent == 0 {
        // Subnormal: scale into the normal range first.
        let (m, e) = frexp(x * f64::from_bits(0x4350_0000_0000_0000)); // 2^54
        return (m, e - 54);
    }
    let mantissa = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (mantissa, exponent - 1022)
}

/// Rounds half away from zero, like C's `round` (`TfLiteRound`).
pub fn round_f64(x: f64) -> f64 {
    if !x.is_finite() || x.abs() >= 4_503_599_627_370_496.0 {
        return x; // already an integer (|x| >= 2^52)
    }
    let truncated = x as i64 as f64;
    let fraction = x - truncated;
    if fraction >= 0.5 {
        truncated + 1.0
    } else if fraction <= -0.5 {
        truncated - 1.0
    } else {
        truncated
    }
}

/// Rounds half away from zero, like C's `roundf`.
pub fn round_f32(x: f32) -> f32 {
    if !x.is_finite() || x.abs() >= 8_388_608.0 {
        return x; // already an integer (|x| >= 2^23)
    }
    let truncated = x as i32 as f32;
    let fraction = x - truncated;
    if fraction >= 0.5 {
        truncated + 1.0
    } else if fraction <= -0.5 {
        truncated - 1.0
    } else {
        truncated
    }
}

/// A real multiplier encoded as a Q31 fixed-point significand and a power-of-two shift
/// (positive: left shift), as produced by TFLite's `QuantizeMultiplier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Multiplier {
    /// Q31 significand.
    pub multiplier: i32,
    /// Power of two; positive values shift left.
    pub shift: i32,
}

// SAFETY: two `i32`, no padding.
unsafe impl crate::pod::Pod for Multiplier {}

/// TFLite `QuantizeMultiplier`.
pub fn quantize_multiplier(real: f64) -> Multiplier {
    if real == 0.0 {
        return Multiplier::default();
    }
    let (q, mut shift) = frexp(real);
    let mut q_fixed = round_f64(q * (1i64 << 31) as f64) as i64;
    if q_fixed == 1i64 << 31 {
        q_fixed /= 2;
        shift += 1;
    }
    if shift < -31 {
        shift = 0;
        q_fixed = 0;
    }
    Multiplier {
        multiplier: q_fixed as i32,
        shift,
    }
}

/// TFLite `QuantizeMultiplierSmallerThanOneExp`: `real` must be in `(0, 1)`, else `None` (where
/// TFLite aborts).
pub fn quantize_multiplier_smaller_than_one(real: f64) -> Option<Multiplier> {
    if !(real > 0.0 && real < 1.0) {
        return None;
    }
    let m = quantize_multiplier(real);
    (m.shift <= 0).then_some(m)
}

/// TFLite `QuantizeMultiplierGreaterThanOne`: `real` must be a finite number greater than 1,
/// else `None` (where TFLite aborts).
pub fn quantize_multiplier_greater_than_one(real: f64) -> Option<Multiplier> {
    if !(real > 1.0 && real.is_finite()) {
        return None;
    }
    let m = quantize_multiplier(real);
    (m.shift >= 0).then_some(m)
}

/// Whether `scale` is a power of two for TFLite's `CheckedLog2`, or close to one.
///
/// `CheckedLog2` accepts scales whose base-2 logarithm, computed in `f32`, is within `1e-3` of an
/// integer. This check accepts a margin of `2e-3` so that it never answers `false` where
/// `CheckedLog2` would answer `true`, whatever the rounding of the C library's `log`: kernels use
/// it to refuse the power-of-two special cases of TFLite they do not implement.
pub fn is_near_power_of_two(scale: f32) -> bool {
    // 2^-0.002 and 2^-0.998: bounds of the mantissas in [0.5, 1) whose log2 is within 0.002 of
    // an integer.
    const UPPER: f64 = 0.998_614_666_101_028_9;
    const LOWER: f64 = 0.500_693_627_855_667_3;
    let (mantissa, _) = frexp(f64::from(scale).abs());
    scale != 0.0 && scale.is_finite() && (mantissa >= UPPER || mantissa <= LOWER)
}

/// TFLite `SaturatingRoundingDoublingHighMul` (gemmlowp).
#[inline]
pub fn saturating_rounding_doubling_high_mul(a: i32, b: i32) -> i32 {
    if a == b && a == i32::MIN {
        return i32::MAX;
    }
    let ab = a as i64 * b as i64;
    let nudge = if ab >= 0 {
        1i64 << 30
    } else {
        1 - (1i64 << 30)
    };
    ((ab + nudge) / (1i64 << 31)) as i32
}

/// TFLite `RoundingDivideByPOT` (gemmlowp): `x / 2^exponent` rounded half away from zero.
#[inline]
pub fn rounding_divide_by_pot(x: i32, exponent: i32) -> i32 {
    debug_assert!((0..=31).contains(&exponent));
    let mask = ((1i64 << exponent) - 1) as i32;
    let remainder = x & mask;
    let threshold = (mask >> 1) + i32::from(x < 0);
    (x >> exponent) + i32::from(remainder > threshold)
}

/// Whether requantization uses single rounding (feature `single-rounding`) rather than double
/// rounding (the default).
pub const SINGLE_ROUNDING: bool = cfg!(feature = "single-rounding");

/// TFLite `MultiplyByQuantizedMultiplier`, in the mode selected by [`SINGLE_ROUNDING`].
#[inline]
pub fn multiply_by_quantized_multiplier(x: i32, m: Multiplier) -> i32 {
    if SINGLE_ROUNDING {
        multiply_by_quantized_multiplier_single(x, m)
    } else {
        multiply_by_quantized_multiplier_double(x, m)
    }
}

/// TFLite `MultiplyByQuantizedMultiplier` with `TFLITE_SINGLE_ROUNDING`: one rounding of the
/// 64-bit product.
#[inline]
pub fn multiply_by_quantized_multiplier_single(x: i32, m: Multiplier) -> i32 {
    let total_shift = 31 - m.shift.clamp(-31, 30);
    let round = 1i64 << (total_shift - 1);
    ((x as i64 * m.multiplier as i64 + round) >> total_shift) as i32
}

/// TFLite `MultiplyByQuantizedMultiplier` without `TFLITE_SINGLE_ROUNDING` (gemmlowp double
/// rounding), the TensorFlow Lite Micro and CMSIS-NN default.
#[inline]
pub fn multiply_by_quantized_multiplier_double(x: i32, m: Multiplier) -> i32 {
    let left_shift = m.shift.max(0);
    let right_shift = (-m.shift).max(0);
    rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(x.wrapping_mul(1i32 << left_shift), m.multiplier),
        right_shift,
    )
}

/// TFLite `MultiplyByQuantizedMultiplier` for 64-bit accumulators (16x8 convolutions and
/// fully-connected layers), the same in both rounding modes: the multiplier is first reduced to
/// 16 bits, then the product is rounded once (half up). CMSIS-NN's `arm_nn_requantize_s64`
/// computes the same value.
///
/// TFLite expects `shift` in `-31..8` and `x` in `-2^47..2^47`; out of these ranges this function
/// wraps around instead of panicking.
#[inline]
pub fn multiply_by_quantized_multiplier_i64(x: i64, m: Multiplier) -> i32 {
    let reduced = if m.multiplier < 0x7fff_0000 {
        (m.multiplier.wrapping_add(1 << 15)) >> 16
    } else {
        0x7fff
    };
    let total_shift = (15 - m.shift.clamp(-47, 14)) as u32;
    let round = 1i64 << (total_shift - 1);
    (x.wrapping_mul(i64::from(reduced)).wrapping_add(round) >> total_shift) as i32
}

/// Per-channel effective scale of convolutions, as computed by TFLite's
/// `PopulateConvolutionQuantizationParams`: `double(input) * double(filter) / double(output)`.
pub fn conv_effective_scale(input_scale: f32, filter_scale: f32, output_scale: f32) -> f64 {
    input_scale as f64 * filter_scale as f64 / output_scale as f64
}

/// Effective scale of per-tensor fully-connected layers, as computed by TFLite's
/// `GetQuantizedConvolutionMultipler`: `double(input) * double(filter) / double(output)`, the
/// same formula as [`conv_effective_scale`].
pub fn fully_connected_effective_scale(
    input_scale: f32,
    filter_scale: f32,
    output_scale: f32,
) -> f64 {
    conv_effective_scale(input_scale, filter_scale, output_scale)
}

/// Quantizes `value` with TFLite's rule: `zero_point + round(value / scale)` computed in `f32`.
pub fn quantize_f32(value: f32, scale: f32, zero_point: i32) -> i32 {
    zero_point.saturating_add(round_f32(value / scale) as i32)
}

/// TFLite `CalculateActivationRangeQuantized`: the clamping range of a fused activation in the
/// quantized domain of the output, intersected with `[qmin, qmax]`.
pub fn activation_range_quantized(
    activation: Activation,
    output_scale: f32,
    output_zero_point: i32,
    qmin: i32,
    qmax: i32,
) -> (i32, i32) {
    let q = |v: f32| quantize_f32(v, output_scale, output_zero_point);
    match activation {
        Activation::Relu => (qmin.max(q(0.0)), qmax),
        Activation::Relu6 => (qmin.max(q(0.0)), qmax.min(q(6.0))),
        Activation::ReluN1To1 => (qmin.max(q(-1.0)), qmax.min(q(1.0))),
        _ => (qmin, qmax),
    }
}

/// TFLite `CalculateActivationRange` for `f32`.
pub fn activation_range_f32(activation: Activation) -> (f32, f32) {
    match activation {
        Activation::Relu => (0.0, f32::MAX),
        Activation::Relu6 => (0.0, 6.0),
        Activation::ReluN1To1 => (-1.0, 1.0),
        _ => (f32::MIN, f32::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frexp_matches_c() {
        assert_eq!(frexp(1.0), (0.5, 1));
        assert_eq!(frexp(0.75), (0.75, 0));
        assert_eq!(frexp(3.0), (0.75, 2));
        assert_eq!(frexp(-0.1), (-0.8, -3));
        let (m, e) = frexp(f64::MIN_POSITIVE / 4.0);
        assert_eq!((m, e), (0.5, -1023));
    }

    #[test]
    fn rounding() {
        assert_eq!(round_f64(2.5), 3.0);
        assert_eq!(round_f64(-2.5), -3.0);
        assert_eq!(round_f64(2.4999), 2.0);
        assert_eq!(round_f32(0.5), 1.0);
        assert_eq!(round_f32(-0.5), -1.0);
    }

    #[test]
    fn quantize_multiplier_matches_tflite() {
        // Values checked against tflite::QuantizeMultiplier.
        assert_eq!(
            quantize_multiplier(0.5),
            Multiplier {
                multiplier: 1 << 30,
                shift: 0
            }
        );
        assert_eq!(
            quantize_multiplier(1.0),
            Multiplier {
                multiplier: 1 << 30,
                shift: 1
            }
        );
        assert_eq!(
            quantize_multiplier(0.25),
            Multiplier {
                multiplier: 1 << 30,
                shift: -1
            }
        );
        assert_eq!(quantize_multiplier(0.0), Multiplier::default());
        assert_eq!(
            quantize_multiplier(1e-12),
            Multiplier {
                multiplier: 0,
                shift: 0
            }
        );
    }

    #[test]
    fn requantize() {
        let m = quantize_multiplier(0.5);
        assert_eq!(multiply_by_quantized_multiplier_double(10, m), 5);
        assert_eq!(multiply_by_quantized_multiplier_double(11, m), 6); // 5.5 rounds away from zero
        assert_eq!(multiply_by_quantized_multiplier_double(-11, m), -5); // gemmlowp nudge then C++ truncation
        assert_eq!(multiply_by_quantized_multiplier_single(11, m), 6);
        assert_eq!(multiply_by_quantized_multiplier_single(-11, m), -5); // rounds half up
        let m = quantize_multiplier(3.0);
        assert_eq!(multiply_by_quantized_multiplier_double(7, m), 21);
        assert_eq!(multiply_by_quantized_multiplier_single(7, m), 21);
        assert_eq!(rounding_divide_by_pot(-5, 1), -3);
        assert_eq!(rounding_divide_by_pot(5, 1), 3);
        assert_eq!(
            saturating_rounding_doubling_high_mul(i32::MIN, i32::MIN),
            i32::MAX
        );
    }

    #[test]
    fn constrained_multipliers() {
        assert_eq!(
            quantize_multiplier_smaller_than_one(0.25),
            Some(quantize_multiplier(0.25))
        );
        assert_eq!(quantize_multiplier_smaller_than_one(1.0), None);
        assert_eq!(quantize_multiplier_smaller_than_one(0.0), None);
        assert_eq!(
            quantize_multiplier_greater_than_one(3.0),
            Some(quantize_multiplier(3.0))
        );
        assert_eq!(quantize_multiplier_greater_than_one(1.0), None);
        assert_eq!(quantize_multiplier_greater_than_one(f64::INFINITY), None);
    }

    #[test]
    fn powers_of_two() {
        for scale in [1.0, 0.5, 2.0, 1.0 / 32768.0, 0.125 * 1.0005, 4.0 * 0.9995] {
            assert!(is_near_power_of_two(scale), "{scale}");
        }
        for scale in [0.75, 0.3, 9.112_61e-5, 0.125 * 1.003, 0.0, f32::NAN] {
            assert!(!is_near_power_of_two(scale), "{scale}");
        }
    }

    #[test]
    fn requantize_i64() {
        // Reference: tflite::MultiplyByQuantizedMultiplier(int64_t, int32_t, int).
        let m = quantize_multiplier(0.5);
        assert_eq!(multiply_by_quantized_multiplier_i64(11, m), 6);
        assert_eq!(multiply_by_quantized_multiplier_i64(-11, m), -5); // rounds half up
        let small = quantize_multiplier(1.0 / (1u64 << 20) as f64);
        assert_eq!(
            multiply_by_quantized_multiplier_i64(1 << 46, small),
            1 << 26
        );
        let m = Multiplier {
            multiplier: 0x7fff_8000,
            shift: -1,
        };
        // The reduced multiplier saturates at 0x7fff.
        assert_eq!(multiply_by_quantized_multiplier_i64(1 << 16, m), 0x7fff);
        let m = quantize_multiplier(3.0);
        assert_eq!(multiply_by_quantized_multiplier_i64(7, m), 21);
        // Out-of-range values wrap around without panicking.
        let _ = multiply_by_quantized_multiplier_i64(i64::MAX, quantize_multiplier(1e30));
    }

    #[test]
    fn activation_ranges() {
        assert_eq!(
            activation_range_quantized(Activation::Relu6, 0.1, -128, -128, 127),
            (-128, -68)
        );
        assert_eq!(
            activation_range_quantized(Activation::None, 0.1, 3, -128, 127),
            (-128, 127)
        );
    }
}
