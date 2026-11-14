// This file is a port to Rust of parts of Eigen, a lightweight C++ template library for linear
// algebra (Eigen/src/Core/MathFunctionsImpl.h, Eigen/src/Core/arch/Default/
// GenericPacketMathFunctions.h and Eigen/src/Core/arch/NEON/PacketMath.h).
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)
// Copyright (C) 2009-2019 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2018-2025 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The `f32` functions of Eigen that LiteRT's float `GELU` calls, reproduced bit for bit.
//!
//! LiteRT's reference `Gelu` evaluates its formula with Eigen array expressions
//! (`internal/reference/gelu.h`), so `tanh` and `erfc` are Eigen's own approximations, not the
//! C library's. These are ports of Eigen at the commit LiteRT 2.3.0 builds with
//! (`ea13a98decd497a8c5588fb5de71b57bcf10d864`, MPL-2.0), as compiled for AArch64: Eigen then
//! evaluates `pmadd` with fused multiply-adds, in its NEON packets of four floats and in scalar
//! code alike (`EIGEN_VECTORIZE_FMA`, `EIGEN_SCALAR_MADD_USE_FMA`), and with `EIGEN_FAST_MATH`.
//!
//! Eigen evaluates an expression over a whole buffer four elements at a time, then finishes the
//! last `len % 4` elements with scalar code (TFLite's buffers are aligned, so no element comes
//! before the first packet). `tanh` is the same rational approximation either way. `erfc` is
//! not: its packet version takes the reciprocal with the NEON estimate (`FRECPE`) refined by two
//! Newton steps (`FRECPS`) and computes `exp` with Eigen's polynomial, where the scalar version
//! divides and calls the C library's `expf` (glibc's, [`crate::math::expf`]). [`erfcf`] takes
//! which one to follow.

use crate::math::expf;

/// `a * b + c` rounded once, as Eigen's `pmadd` on AArch64.
#[inline(always)]
fn madd(a: f32, b: f32, c: f32) -> f32 {
    crate::math::mul_add(a, b, c)
}

/// Eigen's `ppolevl`: the polynomial of coefficients `c` (highest degree first) at `x`, by
/// Horner's rule with fused multiply-adds.
#[inline(always)]
fn polevl(x: f32, c: &[u32]) -> f32 {
    let mut iter = c.iter().map(|&bits| f32::from_bits(bits));
    let first = iter.next().unwrap_or(0.0);
    iter.fold(first, |acc, coeff| madd(acc, x, coeff))
}

/// `min(a, b)` of AArch64 `FMIN` (NaN if either is NaN), as Eigen's `pmin` on NEON.
#[inline(always)]
fn fmin(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a < b {
        a
    } else {
        b
    }
}

/// `max(a, b)` of AArch64 `FMAX` (NaN if either is NaN), as Eigen's `pmax` on NEON.
#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a > b {
        a
    } else {
        b
    }
}

/// Eigen `ptanh_float` (`EIGEN_FAST_MATH`, with FMA): a [9/8] rational approximation on inputs
/// clamped to `[-c, c]`, where `c` is the smallest float at which it is exactly 1.
pub(crate) fn tanhf(x: f32) -> f32 {
    // 8.01773357391357422.
    const CLAMP: f32 = f32::from_bits(0x4100_48a3);
    // Numerator (odd) and denominator (even) coefficients, highest degree first.
    const ALPHA: [u32; 4] = [0x326f_951e, 0x37b0_63dc, 0x3b66_bc7f, 0x3e09_3cf6];
    const BETA: [u32; 5] = [
        0x3557_2c00,
        0x39ae_6d96,
        0x3cd4_c460,
        0x3eef_4922,
        0x3f80_0000,
    ];
    let x = fmax(fmin(x, CLAMP), -CLAMP);
    let x2 = x * x;
    let x3 = x2 * x;
    let p = polevl(x2, &ALPHA);
    let q = polevl(x2, &BETA);
    madd(x3, p, x) / q
}

/// Eigen `pexp_float` (`IsFinite = false`) on one lane: `2^m * exp(r)`, with `exp(r)` a
/// polynomial.
fn expf_packet(x_in: f32) -> f32 {
    // 88.723, log2(e), the two parts of -ln(2) and the polynomial (degree 2 to 6).
    const EXP_HI: f32 = f32::from_bits(0x42b1_722d);
    const LOG2EF: f32 = f32::from_bits(0x3fb8_aa3b);
    const C1: f32 = f32::from_bits(0xbf31_8000);
    const C2: f32 = f32::from_bits(0x395e_8083);
    const P2: f32 = f32::from_bits(0x3eff_fffc);
    const P3: f32 = f32::from_bits(0x3e2a_aa47);
    const P4: f32 = f32::from_bits(0x3d2a_adcc);
    const P5: f32 = f32::from_bits(0x3c09_1de6);
    const P6: f32 = f32::from_bits(0x3ab4_2872);
    let x = fmin(x_in, EXP_HI);
    let m = libm::floorf(madd(x, LOG2EF, 0.5));
    let r = madd(m, C1, x);
    let r = madd(m, C2, r);
    let r2 = r * r;
    let p_even = madd(r2, P6, P4);
    let p_odd = madd(r2, P5, P3);
    let p_even = madd(r2, p_even, P2);
    let p_low = r + 1.0;
    let y = madd(r, p_odd, p_even);
    let y = madd(r2, y, p_low);
    if x_in < -104.0 {
        return 0.0;
    }
    // Eigen takes `pldexp_fast` when no lane exceeds 87 in magnitude and `pldexp_generic`
    // otherwise: both are the exactly rounded `y * 2^m` in the range of `m` that reaches here.
    fmax(ldexp(y, m), x_in)
}

/// Eigen `pldexp_generic`: `a * 2^exponent` in four exact scalings.
fn ldexp(a: f32, exponent: f32) -> f32 {
    // `exponent` is an integer; the cast saturates NaN to 0, as `FCVTZS` does.
    let e = exponent.clamp(-278.0, 278.0) as i32;
    let b = e >> 2;
    let pow2 = |k: i32| f32::from_bits(((k + 127) as u32) << 23);
    let c = pow2(b);
    let out = a * c * c * c;
    out * pow2(e - 3 * b)
}

/// The AArch64 reciprocal estimate `FRECPE` on `f32` (Arm ARM `FPRecipEstimate`, with `FPCR.FZ`
/// and `FPCR.AH` clear as Linux leaves them): 8 bits of `1 / a`.
pub(crate) fn frecpe(a: f32) -> f32 {
    const FRACTION: u64 = (1 << 52) - 1;
    let bits = a.to_bits();
    let sign = bits & 0x8000_0000;
    if a.is_nan() {
        return a + a;
    }
    if a.is_infinite() {
        return f32::from_bits(sign);
    }
    if a.abs() < f32::from_bits(0x0020_0000) {
        // Zero, or a reciprocal too large to represent: infinity (round to nearest).
        return f32::from_bits(sign | 0x7f80_0000);
    }
    // The significand as a 52-bit fraction, and the exponent.
    let mut fraction = u64::from(bits & 0x007f_ffff) << 29;
    let mut exp = ((bits >> 23) & 0xff) as i32;
    if exp == 0 {
        if fraction >> 51 == 0 {
            exp = -1;
            fraction = (fraction << 2) & FRACTION;
        } else {
            fraction = (fraction << 1) & FRACTION;
        }
    }
    // `RecipEstimate` of the 9-bit fixed-point significand in [0.5, 1).
    let scaled = 256 + ((fraction >> 44) & 0xff) as u32;
    let estimate = ((1u32 << 19) / (scaled * 2 + 1)).div_ceil(2);
    let mut result_exp = 253 - exp;
    let mut fraction = u64::from(estimate & 0xff) << 44;
    if result_exp == 0 {
        fraction = (1 << 51) | (fraction >> 1);
    } else if result_exp == -1 {
        fraction = (1 << 50) | (fraction >> 2);
        result_exp = 0;
    }
    f32::from_bits(sign | ((result_exp as u32 & 0xff) << 23) | (fraction >> 29) as u32)
}

/// Eigen `preciprocal` on NEON: the `FRECPE` estimate refined by two `FRECPS` Newton steps
/// (`FRECPS` computes `2 - a * r` rounded once).
fn reciprocal_packet(a: f32) -> f32 {
    let mut r = frecpe(a);
    r *= madd(-a, r, 2.0);
    r *= madd(-a, r, 2.0);
    r
}

/// Eigen `generic_fast_erfc<float>`: `1 + x * S(x^2)` for `|x| <= 1`, and
/// `exp(-x^2) / x * P(1/x^2) / Q(1/x^2)` (plus 2 for negative `x`) beyond, on inputs clamped to
/// `[-11, 11]`. `packet` selects the NEON packet version, else the scalar one (see the module).
pub(crate) fn erfcf(x: f32, packet: bool) -> f32 {
    const CLAMP: f32 = 11.0;
    // S, P and Q, highest degree first.
    const ALPHA: [u32; 6] = [
        0x3a13_45ea,
        0xbba1_0415,
        0x3cda_c9b8,
        0xbde7_03be,
        0x3ec0_9330,
        0xbf90_6eba,
    ];
    const GAMMA: [u32; 4] = [0x3dd1_0fee, 0x3edb_c0fa, 0x3ea5_c7e8, 0x3d5d_11b8];
    const DELTA: [u32; 5] = [
        0x3c8d_5364,
        0x3ec8_61dd,
        0x3f80_0000,
        0x3f1f_29db,
        0x3dc3_eae4,
    ];
    let x = fmin(fmax(x, -CLAMP), CLAMP);
    let x2 = x * x;
    let erfc_small = madd(x, polevl(x2, &ALPHA), 1.0);
    // Eigen selects the second form where `1 < x2`: NaN keeps the first.
    if x2 <= 1.0 || x2.is_nan() {
        return erfc_small;
    }
    // x^2 = x2 + x2_lo exactly; exp(-x^2) ~= exp(-x2) * (1 - x2_lo).
    let x2_lo = madd(x, x, -x2);
    let exp2_hi = if packet { expf_packet(-x2) } else { expf(-x2) };
    let z = madd(-exp2_hi, x2_lo, exp2_hi);
    let q2 = if packet {
        reciprocal_packet(x2)
    } else {
        1.0 / x2
    };
    let num = polevl(q2, &GAMMA);
    let denom = x * polevl(q2, &DELTA);
    let r = num / denom;
    let maybe_two = if x < 0.0 { 2.0 } else { 0.0 };
    madd(z, r, maybe_two)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `frecpe` against the instruction itself, on every exponent and many significands.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn frecpe_matches_the_instruction() {
        use core::arch::aarch64::vrecpes_f32;
        let mut bits = 0u32;
        for _ in 0..2_000_000 {
            bits = bits.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
            let x = f32::from_bits(bits);
            // SAFETY: NEON is part of the AArch64 baseline.
            let expected = unsafe { vrecpes_f32(x) };
            let actual = frecpe(x);
            assert!(
                actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
                "frecpe({x:e}): {actual:e} != {expected:e}"
            );
        }
    }

    #[test]
    fn special_values() {
        assert_eq!(tanhf(0.0), 0.0);
        assert_eq!(tanhf(20.0), 1.0);
        assert_eq!(tanhf(f32::NEG_INFINITY), -1.0);
        assert!(tanhf(f32::NAN).is_nan());
        assert_eq!(frecpe(f32::INFINITY), 0.0);
        assert_eq!(frecpe(-0.0), f32::NEG_INFINITY);
        // 1 / 2 to 8 bits: 511 / 1024.
        assert_eq!(frecpe(2.0), 511.0 / 1024.0);
        for packet in [false, true] {
            assert_eq!(erfcf(0.0, packet), 1.0);
            assert_eq!(erfcf(20.0, packet), 0.0);
            assert_eq!(erfcf(-20.0, packet), 2.0);
            assert!(erfcf(f32::NAN, packet).is_nan());
        }
        assert_eq!(expf_packet(0.0), 1.0);
        assert_eq!(expf_packet(-200.0), 0.0);
    }
}
