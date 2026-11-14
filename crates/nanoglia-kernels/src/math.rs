//! The floating-point functions of the reference kernels, reproduced bit for bit.
//!
//! TFLite calls the C library: `std::exp` and `std::tanh` on `float`. The reference outputs are
//! produced on Linux, so these are ports of the GNU C library 2.41 implementations, which in turn
//! come from Arm's optimized-routines (`expf`, MIT) and from the CORE-MATH project (`tanhf`,
//! correctly rounded, MIT). Both were checked against glibc on every one of the 2^32 inputs.
//! They compute in `f64`, so they cost more than `libm`'s `f32` versions on single-precision
//! FPUs; the int8 kernels only call them while preparing their lookup tables.
//!
//! The dot products of the reference convolutions (`total += input * filter`) are compiled into
//! fused multiply-adds by Clang on AArch64 (the LiteRT wheels) and by GCC on Arm (TFLite Micro),
//! so [`mul_add`] rounds once too. Where Clang vectorizes such a loop instead, [`dot_by_eight`]
//! reproduces the order in which the vectorized code rounds.

/// `2^(i/32)` as `f64` bits, minus `i << 47` (`__exp2f_data.tab`).
const EXP2_TABLE: [u64; 32] = [
    0x3ff0_0000_0000_0000,
    0x3fef_d9b0_d315_8574,
    0x3fef_b558_6cf9_890f,
    0x3fef_9301_d012_5b51,
    0x3fef_72b8_3c7d_517b,
    0x3fef_5487_3168_b9aa,
    0x3fef_387a_6e75_6238,
    0x3fef_1e9d_f51f_dee1,
    0x3fef_06fe_0a31_b715,
    0x3fee_f1a7_373a_a9cb,
    0x3fee_dea6_4c12_3422,
    0x3fee_ce08_6061_892d,
    0x3fee_bfda_d536_2a27,
    0x3fee_b42b_569d_4f82,
    0x3fee_ab07_dd48_5429,
    0x3fee_a47e_b03a_5585,
    0x3fee_a09e_667f_3bcd,
    0x3fee_9f75_e8ec_5f74,
    0x3fee_a114_73eb_0187,
    0x3fee_a589_994c_ce13,
    0x3fee_ace5_422a_a0db,
    0x3fee_b737_b0cd_c5e5,
    0x3fee_c491_82a3_f090,
    0x3fee_d503_b23e_255d,
    0x3fee_e89f_995a_d3ad,
    0x3fee_ff76_f2fb_5e47,
    0x3fef_199b_dd85_529c,
    0x3fef_3720_dcef_9069,
    0x3fef_5818_dcfb_a487,
    0x3fef_7c97_337b_9b5f,
    0x3fef_a4af_a2a4_90da,
    0x3fef_d076_5b6e_4540,
];

/// `e^x` on `f32`, bit-exact with glibc's `expf`.
pub(crate) fn expf(x: f32) -> f32 {
    // 32 / ln(2), and the polynomial approximating 2^(r/32) - 1, scaled by powers of 1/32.
    let inv_ln2_n = f64::from_bits(0x3ff7_1547_652b_82fe) * 32.0;
    let c0 = f64::from_bits(0x3fac_6af8_4b91_2394) / 32.0 / 32.0 / 32.0;
    let c1 = f64::from_bits(0x3fce_bfce_50fa_c4f3) / 32.0 / 32.0;
    let c2 = f64::from_bits(0x3fe6_2e42_ff0c_52d6) / 32.0;
    // 1.5 * 2^52: adding it rounds to an integer (ties to even).
    let shift = f64::from_bits(0x4338_0000_0000_0000);

    let abstop = (x.to_bits() >> 20) & 0x7ff;
    if abstop >= (88.0f32.to_bits() >> 20) {
        // |x| >= 88 or NaN.
        if x == f32::NEG_INFINITY {
            return 0.0;
        }
        if abstop >= (f32::INFINITY.to_bits() >> 20) {
            return x + x;
        }
        if x > f32::from_bits(0x42b1_7217) {
            return f32::INFINITY; // x > ln(2^128)
        }
        if x < f32::from_bits(0xc2cf_f1b4) {
            return 0.0; // x < ln(2^-150)
        }
    }
    // x * 32 / ln(2) = k + r, with r in [-1/2, 1/2] and integer k.
    let z = inv_ln2_n * f64::from(x);
    let kd = z + shift;
    let ki = kd.to_bits();
    let kd = kd - shift;
    let r = z - kd;
    // e^x = 2^(k/32) * 2^(r/32) ~= s * (c0 r^3 + c1 r^2 + c2 r + 1).
    let t = EXP2_TABLE[(ki % 32) as usize].wrapping_add(ki << 47);
    let s = f64::from_bits(t);
    let p = c0 * r + c1;
    let r2 = r * r;
    let y = c2 * r + 1.0;
    let y = p * r2 + y;
    (y * s) as f32
}

/// `tanh(x)` on `f32`, correctly rounded (bit-exact with glibc's `tanhf`).
pub(crate) fn tanhf(x: f32) -> f32 {
    const CN: [u64; 8] = [
        0x3ff0_0000_0000_0000,
        0x3fc3_0877_b8b7_2d33,
        0x3f76_94aa_09ae_9e5e,
        0x3f14_1013_77ab_b729,
        0x3e9e_0392_b1db_0018,
        0x3e12_5337_56e5_46f7,
        0x3d6d_62e5_abe6_ae8a,
        0x3c9b_06be_5341_82de,
    ];
    const CD: [u64; 8] = [
        0x3ff0_0000_0000_0000,
        0x3fde_d991_31b0_ebea,
        0x3fa0_d27e_d6c9_5a69,
        0x3f47_cbda_ca0e_9fcc,
        0x3edb_4e60_b892_578e,
        0x3e5a_6f70_7c5c_71ab,
        0x3dc3_5a8b_6e2c_d94c,
        0x3d0c_a823_0677_aa01,
    ];
    let ux = x.to_bits();
    let e = (ux >> 23) & 0xff;
    if e == 0xff {
        if ux << 9 != 0 {
            return x + x; // NaN
        }
        return if ux >> 31 != 0 { -1.0 } else { 1.0 };
    }
    if e < 115 {
        // |x| < 2^-12: tanh(x) ~= x - x^3 / 3.
        if e < 102 {
            if ux << 1 == 0 {
                return x;
            }
            return libm::fmaf(-x, x.abs(), x);
        }
        let x2 = x * x;
        return libm::fmaf(x, f32::from_bits(0xbeaa_aaab) * x2, x);
    }
    if (ux << 1) > (0x4110_2cb3u32 << 1) {
        // |x| > ~9.01: tanh(x) rounds to +-(1 - 2^-25).
        return 1.0f32.copysign(x) - f32::from_bits(0x3300_0000).copysign(x);
    }
    let z = f64::from(x);
    let (z2, cn, cd) = (z * z, CN.map(f64::from_bits), CD.map(f64::from_bits));
    let z4 = z2 * z2;
    let z8 = z4 * z4;
    let mut n0 = cn[0] + z2 * cn[1];
    let n2 = cn[2] + z2 * cn[3];
    let mut n4 = cn[4] + z2 * cn[5];
    let n6 = cn[6] + z2 * cn[7];
    n0 += z4 * n2;
    n4 += z4 * n6;
    n0 += z8 * n4;
    let mut d0 = cd[0] + z2 * cd[1];
    let d2 = cd[2] + z2 * cd[3];
    let mut d4 = cd[4] + z2 * cd[5];
    let d6 = cd[6] + z2 * cd[7];
    d0 += z4 * d2;
    d4 += z4 * d6;
    d0 += z8 * d4;
    (z * n0 / d0) as f32
}

/// `a * b + c` rounded once.
#[inline(always)]
pub(crate) fn mul_add(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(nanoglia_arm_vfma)]
    {
        let mut acc = c.to_bits();
        // SAFETY: the instructions only use the three general registers given and `s0`-`s2`,
        // declared as clobbered; every Arm M-profile hard-float target (VFPv4 and later) has
        // `vfma.f32` (see `build.rs`).
        unsafe {
            core::arch::asm!(
                "vmov s0, {a}",
                "vmov s1, {b}",
                "vmov s2, {acc}",
                "vfma.f32 s2, s0, s1",
                "vmov {acc}, s2",
                acc = inout(reg) acc,
                a = in(reg) a.to_bits(),
                b = in(reg) b.to_bits(),
                out("s0") _,
                out("s1") _,
                out("s2") _,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        f32::from_bits(acc)
    }
    #[cfg(not(nanoglia_arm_vfma))]
    {
        libm::fmaf(a, b, c)
    }
}

/// The sum of `term(k).0 * term(k).1` for `k` in `0..len`, as LiteRT's AArch64 build computes the
/// reduction loops of its reference `BATCH_MATMUL` and `L2_NORMALIZATION` (`total += a * b`, both
/// operands contiguous): Clang vectorizes them eight terms at a time, rounding each product and
/// adding them in order (a strict, in-order reduction), and fuses the multiply-add of the
/// `len % 8` terms left, which its scalar loop runs. Measured against LiteRT 2.3.0 for every
/// length from 1 to 40.
#[inline(always)]
pub(crate) fn dot_by_eight(len: usize, term: impl Fn(usize) -> (f32, f32)) -> f32 {
    let vectorized = len - len % 8;
    let mut total = 0.0f32;
    for k in 0..vectorized {
        let (a, b) = term(k);
        total += a * b;
    }
    for k in vectorized..len {
        let (a, b) = term(k);
        total = mul_add(a, b, total);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe extern "C" {
        #[link_name = "expf"]
        fn c_expf(x: f32) -> f32;
        #[link_name = "tanhf"]
        fn c_tanhf(x: f32) -> f32;
    }

    /// Spot-checks against the host C library (the full check covers all 2^32 inputs).
    #[test]
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn matches_glibc() {
        let mut bits = 0u32;
        for _ in 0..2_000_000 {
            bits = bits.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
            let x = f32::from_bits(bits);
            // SAFETY: plain C math functions.
            let (e, t) = unsafe { (c_expf(x), c_tanhf(x)) };
            assert!(
                expf(x).to_bits() == e.to_bits() || e.is_nan(),
                "expf({x:e})"
            );
            assert!(
                tanhf(x).to_bits() == t.to_bits() || t.is_nan(),
                "tanhf({x:e})"
            );
        }
    }

    #[test]
    fn special_values() {
        assert_eq!(expf(0.0), 1.0);
        assert_eq!(expf(f32::NEG_INFINITY), 0.0);
        assert_eq!(expf(100.0), f32::INFINITY);
        assert_eq!(expf(-120.0), 0.0);
        assert!(expf(f32::NAN).is_nan());
        assert_eq!(tanhf(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(tanhf(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(tanhf(f32::INFINITY), 1.0);
        assert_eq!(tanhf(-20.0), -1.0 + f32::from_bits(0x3300_0000));
        assert!(tanhf(f32::NAN).is_nan());
        assert_eq!(mul_add(2.0, 3.0, 1.0), 7.0);
        // 1 + 2^-23 squared needs the fused rounding to keep the 2^-46 term.
        let a = 1.0 + f32::EPSILON;
        assert_eq!(
            mul_add(a, a, -1.0),
            2.0 * f32::EPSILON + f32::EPSILON * f32::EPSILON
        );
    }

    #[test]
    fn dot_by_eight_fuses_the_remainder_only() {
        // (1 + 2^-12)^2 - 1: the fused multiply-add keeps the 2^-24 term, the rounded product
        // loses it.
        let a = 1.0 + 1.0 / 4096.0;
        let fused = 1.0 / 2048.0 + 1.0 / 16_777_216.0;
        let rounded = 1.0 / 2048.0;
        // -1, zeros, then a * a: the last term decides.
        let dot = |n: usize| {
            dot_by_eight(n, |k| match k {
                0 => (-1.0, 1.0),
                k if k + 1 == n => (a, a),
                _ => (0.0, 0.0),
            })
        };
        assert_eq!(dot(4), fused);
        assert_eq!(dot(8), rounded);
        assert_eq!(dot(9), fused);
        assert_eq!(dot(16), rounded);
        assert_eq!(dot(17), fused);
    }
}
