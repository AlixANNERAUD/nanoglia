//! Straightforward ports of the TFLite reference integer kernels
//! (`tensorflow/lite/kernels/internal/reference/integer_ops/{conv,depthwise_conv,
//! fully_connected,pooling,add,mul}.h` and the quantized path of `reference/softmax.h`), with the
//! parameter computations of the matching LiteRT `Prepare` functions.

use nanonn_core::Activation;
use nanonn_core::quant::{
    self, Multiplier, multiply_by_quantized_multiplier, rounding_divide_by_pot,
    saturating_rounding_doubling_high_mul,
};

/// `RuntimeShape::Offset` for NHWC shapes.
fn offset(dims: [i32; 4], b: i32, y: i32, x: i32, c: i32) -> usize {
    (((b * dims[1] + y) * dims[2] + x) * dims[3] + c) as usize
}

/// LiteRT `ComputePaddingHeightWidth` for one dimension: `(output size, padding)`.
pub fn padding(same: bool, input: i32, filter: i32, stride: i32, dilation: i32) -> (i32, i32) {
    let effective = (filter - 1) * dilation + 1;
    let out = if same {
        (input + stride - 1) / stride
    } else {
        (input + stride - effective) / stride
    };
    let total = ((out - 1) * stride + effective - input).max(0);
    (out, total / 2)
}

/// `CalculateActivationRangeQuantized` for int8.
pub fn activation_range(activation: Activation, scale: f32, zero_point: i32) -> (i32, i32) {
    quant::activation_range_quantized(activation, scale, zero_point, -128, 127)
}

/// `PopulateConvolutionQuantizationParams`: one multiplier per output channel.
pub fn channel_multipliers(
    input_scale: f32,
    filter_scales: &[f32],
    output_scale: f32,
    channels: usize,
) -> Vec<Multiplier> {
    (0..channels)
        .map(|c| {
            let filter_scale = if filter_scales.len() == 1 {
                filter_scales[0]
            } else {
                filter_scales[c]
            };
            let real = input_scale as f64 * filter_scale as f64 / output_scale as f64;
            quant::quantize_multiplier(real)
        })
        .collect()
}

pub struct ConvParams {
    pub input_offset: i32,
    pub output_offset: i32,
    pub stride: (i32, i32),
    pub dilation: (i32, i32),
    pub padding: (i32, i32),
    pub activation: (i32, i32),
}

/// `reference_integer_ops::ConvPerChannel` (int8), with `(h, w)` pairs in `ConvParams`.
#[allow(clippy::too_many_arguments)]
pub fn conv_per_channel(
    p: &ConvParams,
    multipliers: &[Multiplier],
    input_dims: [i32; 4],
    input: &[i8],
    filter_dims: [i32; 4],
    filter: &[i8],
    bias: Option<&[i32]>,
    output_dims: [i32; 4],
) -> Vec<i8> {
    let mut output = vec![0i8; output_dims.iter().product::<i32>() as usize];
    let [batches, in_h, in_w, in_c] = input_dims;
    let [out_c, filter_h, filter_w, filter_c] = filter_dims;
    let [_, out_h, out_w, _] = output_dims;
    let groups = in_c / filter_c;
    let filters_per_group = out_c / groups;
    for b in 0..batches {
        for oy in 0..out_h {
            let in_y0 = oy * p.stride.0 - p.padding.0;
            for ox in 0..out_w {
                let in_x0 = ox * p.stride.1 - p.padding.1;
                for oc in 0..out_c {
                    let group = oc / filters_per_group;
                    let mut acc: i32 = 0;
                    for fy in 0..filter_h {
                        let in_y = in_y0 + p.dilation.0 * fy;
                        for fx in 0..filter_w {
                            let in_x = in_x0 + p.dilation.1 * fx;
                            if in_x < 0 || in_x >= in_w || in_y < 0 || in_y >= in_h {
                                continue;
                            }
                            for ic in 0..filter_c {
                                let iv = input
                                    [offset(input_dims, b, in_y, in_x, ic + group * filter_c)]
                                    as i32;
                                let fv = filter[offset(filter_dims, oc, fy, fx, ic)] as i32;
                                acc += fv * (iv + p.input_offset);
                            }
                        }
                    }
                    if let Some(bias) = bias {
                        acc += bias[oc as usize];
                    }
                    acc = multiply_by_quantized_multiplier(acc, multipliers[oc as usize]);
                    acc += p.output_offset;
                    acc = acc.clamp(p.activation.0, p.activation.1);
                    output[offset(output_dims, b, oy, ox, oc)] = acc as i8;
                }
            }
        }
    }
    output
}

/// `reference_integer_ops::DepthwiseConvPerChannel` (int8).
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv_per_channel(
    p: &ConvParams,
    depth_multiplier: i32,
    multipliers: &[Multiplier],
    input_dims: [i32; 4],
    input: &[i8],
    filter_dims: [i32; 4],
    filter: &[i8],
    bias: Option<&[i32]>,
    output_dims: [i32; 4],
) -> Vec<i8> {
    let mut output = vec![0i8; output_dims.iter().product::<i32>() as usize];
    let [batches, in_h, in_w, in_c] = input_dims;
    let [_, filter_h, filter_w, _] = filter_dims;
    let [_, out_h, out_w, _] = output_dims;
    for b in 0..batches {
        for oy in 0..out_h {
            for ox in 0..out_w {
                for ic in 0..in_c {
                    for m in 0..depth_multiplier {
                        let oc = m + ic * depth_multiplier;
                        let in_x0 = ox * p.stride.1 - p.padding.1;
                        let in_y0 = oy * p.stride.0 - p.padding.0;
                        let mut acc: i32 = 0;
                        for fy in 0..filter_h {
                            for fx in 0..filter_w {
                                let in_x = in_x0 + p.dilation.1 * fx;
                                let in_y = in_y0 + p.dilation.0 * fy;
                                if in_x >= 0 && in_x < in_w && in_y >= 0 && in_y < in_h {
                                    let iv = input[offset(input_dims, b, in_y, in_x, ic)] as i32;
                                    let fv = filter[offset(filter_dims, 0, fy, fx, oc)] as i32;
                                    acc += fv * (iv + p.input_offset);
                                }
                            }
                        }
                        if let Some(bias) = bias {
                            acc += bias[oc as usize];
                        }
                        acc = multiply_by_quantized_multiplier(acc, multipliers[oc as usize]);
                        acc += p.output_offset;
                        acc = acc.clamp(p.activation.0, p.activation.1);
                        output[offset(output_dims, b, oy, ox, oc)] = acc as i8;
                    }
                }
            }
        }
    }
    output
}

/// `reference_integer_ops::FullyConnected` / `FullyConnectedPerChannel` (int8, weight zero
/// point 0): `input` is `[batches, accum_depth]`, `filter` `[output_depth, accum_depth]`.
#[allow(clippy::too_many_arguments)]
pub fn fully_connected(
    input_offset: i32,
    output_offset: i32,
    activation: (i32, i32),
    multipliers: &[Multiplier],
    batches: usize,
    accum_depth: usize,
    output_depth: usize,
    input: &[i8],
    filter: &[i8],
    bias: Option<&[i32]>,
) -> Vec<i8> {
    let mut output = vec![0i8; batches * output_depth];
    for b in 0..batches {
        for oc in 0..output_depth {
            let mut acc: i32 = 0;
            for d in 0..accum_depth {
                let iv = input[b * accum_depth + d] as i32;
                let fv = filter[oc * accum_depth + d] as i32;
                acc += fv * (iv + input_offset);
            }
            if let Some(bias) = bias {
                acc += bias[oc];
            }
            let mut v = multiply_by_quantized_multiplier(acc, multipliers[oc]);
            v += output_offset;
            output[oc + output_depth * b] = v.clamp(activation.0, activation.1) as i8;
        }
    }
    output
}

pub struct PoolParams {
    pub stride: (i32, i32),
    pub filter: (i32, i32),
    pub padding: (i32, i32),
    pub activation: (i32, i32),
}

/// `reference_integer_ops::AveragePool` (int8).
pub fn average_pool(
    p: &PoolParams,
    input_dims: [i32; 4],
    input: &[i8],
    output_dims: [i32; 4],
) -> Vec<i8> {
    let mut output = vec![0i8; output_dims.iter().product::<i32>() as usize];
    let [batches, in_h, in_w, depth] = input_dims;
    let [_, out_h, out_w, _] = output_dims;
    for b in 0..batches {
        for oy in 0..out_h {
            for ox in 0..out_w {
                for c in 0..depth {
                    let in_x0 = ox * p.stride.1 - p.padding.1;
                    let in_y0 = oy * p.stride.0 - p.padding.0;
                    let fx_start = 0.max(-in_x0);
                    let fx_end = p.filter.1.min(in_w - in_x0);
                    let fy_start = 0.max(-in_y0);
                    let fy_end = p.filter.0.min(in_h - in_y0);
                    let mut acc: i32 = 0;
                    let mut count: i32 = 0;
                    for fy in fy_start..fy_end {
                        for fx in fx_start..fx_end {
                            acc += input[offset(input_dims, b, in_y0 + fy, in_x0 + fx, c)] as i32;
                            count += 1;
                        }
                    }
                    assert!(count > 0, "empty pooling window");
                    acc = if acc > 0 {
                        (acc + count / 2) / count
                    } else {
                        (acc - count / 2) / count
                    };
                    output[offset(output_dims, b, oy, ox, c)] =
                        acc.clamp(p.activation.0, p.activation.1) as i8;
                }
            }
        }
    }
    output
}

/// `reference_integer_ops::MaxPool` (int8).
pub fn max_pool(
    p: &PoolParams,
    input_dims: [i32; 4],
    input: &[i8],
    output_dims: [i32; 4],
) -> Vec<i8> {
    let mut output = vec![0i8; output_dims.iter().product::<i32>() as usize];
    let [batches, in_h, in_w, depth] = input_dims;
    let [_, out_h, out_w, _] = output_dims;
    for b in 0..batches {
        for oy in 0..out_h {
            for ox in 0..out_w {
                for c in 0..depth {
                    let in_x0 = ox * p.stride.1 - p.padding.1;
                    let in_y0 = oy * p.stride.0 - p.padding.0;
                    let fx_start = 0.max(-in_x0);
                    let fx_end = p.filter.1.min(in_w - in_x0);
                    let fy_start = 0.max(-in_y0);
                    let fy_end = p.filter.0.min(in_h - in_y0);
                    let mut max = i8::MIN as i32;
                    for fy in fy_start..fy_end {
                        for fx in fx_start..fx_end {
                            max = max.max(
                                input[offset(input_dims, b, in_y0 + fy, in_x0 + fx, c)] as i32,
                            );
                        }
                    }
                    output[offset(output_dims, b, oy, ox, c)] =
                        max.clamp(p.activation.0, p.activation.1) as i8;
                }
            }
        }
    }
    output
}

pub struct AddParams {
    pub left_shift: i32,
    pub input1_offset: i32,
    pub input1: Multiplier,
    pub input2_offset: i32,
    pub input2: Multiplier,
    pub output_offset: i32,
    pub output: Multiplier,
    pub activation: (i32, i32),
}

/// LiteRT `ADD` Prepare (8-bit general path).
pub fn add_params(
    s1: f32,
    z1: i32,
    s2: f32,
    z2: i32,
    so: f32,
    zo: i32,
    activation: Activation,
) -> AddParams {
    let left_shift = 20;
    let twice_max = 2.0 * s1.max(s2) as f64;
    AddParams {
        left_shift,
        input1_offset: -z1,
        input1: quant::quantize_multiplier(s1 as f64 / twice_max),
        input2_offset: -z2,
        input2: quant::quantize_multiplier(s2 as f64 / twice_max),
        output_offset: zo,
        output: quant::quantize_multiplier(twice_max / ((1i64 << left_shift) as f64 * so as f64)),
        activation: activation_range(activation, so, zo),
    }
}

/// `reference_integer_ops::Add` (int8, same shapes).
pub fn add(p: &AddParams, input1: &[i8], input2: &[i8]) -> Vec<i8> {
    // MultiplyByQuantizedMultiplierSmallerThanOneExp.
    let smaller = |x: i32, m: Multiplier| {
        rounding_divide_by_pot(
            saturating_rounding_doubling_high_mul(x, m.multiplier),
            -m.shift,
        )
    };
    input1
        .iter()
        .zip(input2)
        .map(|(&a, &b)| {
            let v1 = (p.input1_offset + a as i32) * (1 << p.left_shift);
            let v2 = (p.input2_offset + b as i32) * (1 << p.left_shift);
            let sum = smaller(v1, p.input1) + smaller(v2, p.input2);
            let out = smaller(sum, p.output) + p.output_offset;
            out.clamp(p.activation.0, p.activation.1) as i8
        })
        .collect()
}

/// `reference_integer_ops::Mul` (int8, same shapes), with LiteRT's Prepare.
#[allow(clippy::too_many_arguments)]
pub fn mul(
    s1: f32,
    z1: i32,
    s2: f32,
    z2: i32,
    so: f32,
    zo: i32,
    activation: Activation,
    input1: &[i8],
    input2: &[i8],
) -> Vec<i8> {
    let m = quant::quantize_multiplier(s1 as f64 * s2 as f64 / so as f64);
    let (lo, hi) = activation_range(activation, so, zo);
    input1
        .iter()
        .zip(input2)
        .map(|(&a, &b)| {
            let v = (a as i32 - z1) * (b as i32 - z2);
            (zo + multiply_by_quantized_multiplier(v, m)).clamp(lo, hi) as i8
        })
        .collect()
}

// ------------------------------------------------------------------ Softmax (gemmlowp)

/// `gemmlowp::SaturatingRoundingMultiplyByPOT<exponent>`.
fn saturating_rounding_multiply_by_pot(x: i32, exponent: i32) -> i32 {
    if exponent > 0 {
        let threshold = (1i32 << (31 - exponent)) - 1;
        if x > threshold {
            i32::MAX
        } else if x < -threshold {
            i32::MIN
        } else {
            x.wrapping_shl(exponent as u32)
        }
    } else if exponent < 0 {
        rounding_divide_by_pot(x, -exponent)
    } else {
        x
    }
}

/// `gemmlowp::Rescale` of a raw value from `src` to `dst` integer bits.
fn rescale(x: i32, src: i32, dst: i32) -> i32 {
    saturating_rounding_multiply_by_pot(x, src - dst)
}

/// Fixed-point multiplication (integer bits add up).
fn fmul(a: i32, b: i32) -> i32 {
    saturating_rounding_doubling_high_mul(a, b)
}

/// `gemmlowp::exp_on_interval_between_negative_one_quarter_and_0_excl` (Q0.31).
fn exp_on_interval_between_negative_one_quarter_and_0_excl(a: i32) -> i32 {
    let constant_term: i32 = 1_895_147_668;
    let constant_1_over_3: i32 = 715_827_883;
    let x = a.wrapping_add(1 << 28);
    let x2 = fmul(x, x);
    let x3 = fmul(x2, x);
    let x4 = fmul(x2, x2);
    let x4_over_4 = saturating_rounding_multiply_by_pot(x4, -2);
    let poly = saturating_rounding_multiply_by_pot(
        fmul(x4_over_4.wrapping_add(x3), constant_1_over_3).wrapping_add(x2),
        -1,
    );
    constant_term.wrapping_add(fmul(constant_term, x.wrapping_add(poly)))
}

/// `gemmlowp::exp_on_negative_values` for an input with `integer_bits` integer bits (< 6).
fn exp_on_negative_values(a: i32, integer_bits: i32) -> i32 {
    let fractional_bits = 31 - integer_bits;
    let one_quarter = 1i32 << (fractional_bits - 2);
    let mask = one_quarter - 1;
    let a_mod_quarter_minus_one_quarter = (a & mask).wrapping_sub(one_quarter);
    let mut result = exp_on_interval_between_negative_one_quarter_and_0_excl(rescale(
        a_mod_quarter_minus_one_quarter,
        integer_bits,
        0,
    ));
    let remainder = a_mod_quarter_minus_one_quarter.wrapping_sub(a);
    let barrel = [
        (-2, 1_672_461_947),
        (-1, 1_302_514_674),
        (0, 790_015_084),
        (1, 290_630_308),
        (2, 39_332_535),
        (3, 720_401),
        (4, 242),
    ];
    for (exponent, multiplier) in barrel {
        if integer_bits > exponent {
            let shift = fractional_bits + exponent;
            if remainder & (1 << shift) != 0 {
                result = fmul(result, multiplier);
            }
        }
    }
    assert!(integer_bits <= 5);
    if a == 0 { i32::MAX } else { result }
}

/// `gemmlowp::one_over_one_plus_x_for_x_in_0_1` (Q0.31 in and out).
fn one_over_one_plus_x_for_x_in_0_1(a: i32) -> i32 {
    // RoundingHalfSum(a, One()).
    let sum = a as i64 + i32::MAX as i64;
    let half_denominator = ((sum + if sum >= 0 { 1 } else { -1 }) / 2) as i32;
    let constant_48_over_17 = 1_515_870_810;
    let constant_neg_32_over_17 = -1_010_580_540;
    let one_f2 = 1i32 << 29;
    let mut x = constant_48_over_17 + fmul(half_denominator, constant_neg_32_over_17);
    for _ in 0..3 {
        let half_denominator_times_x = fmul(half_denominator, x);
        let one_minus = one_f2.wrapping_sub(half_denominator_times_x);
        x = x.wrapping_add(rescale(fmul(x, one_minus), 4, 2));
    }
    // ExactMulByPot<-1> turns Q2 into Q1 without touching the raw value.
    rescale(x, 1, 0)
}

/// `GetReciprocal`.
fn get_reciprocal(x: i32, x_integer_digits: i32) -> (i32, i32) {
    let headroom_plus_one = (x as u32).leading_zeros() as i32;
    let num_bits_over_unit = x_integer_digits - headroom_plus_one;
    let shifted_sum_minus_one =
        ((x as u32).wrapping_shl(headroom_plus_one as u32)).wrapping_sub(1u32 << 31) as i32;
    (
        one_over_one_plus_x_for_x_in_0_1(shifted_sum_minus_one),
        num_bits_over_unit,
    )
}

/// LiteRT `SOFTMAX` Prepare for int8 (`kScaledDiffIntegerBits = 5`): `(multiplier, left shift,
/// diff_min)`.
pub fn softmax_params(beta: f32, input_scale: f32) -> (i32, i32, i32) {
    let real = (beta as f64 * input_scale as f64 * (1i64 << (31 - 5)) as f64)
        .min(((1i64 << 31) - 1) as f64);
    let m = quant::quantize_multiplier(real);
    assert!(m.shift >= 0);
    let radius = (((1i64 << 5) - 1) as f64 * (1i64 << (31 - 5)) as f64 / (1i64 << m.shift) as f64)
        .floor() as i32;
    (m.multiplier, m.shift, -radius)
}

/// `reference_ops::Softmax` (int8 input and output).
pub fn softmax(beta: f32, input_scale: f32, input: &[i8], depth: usize) -> Vec<i8> {
    let (multiplier, left_shift, diff_min) = softmax_params(beta, input_scale);
    let scaled_diff_integer_bits = 5;
    let accumulation_integer_bits = 12;
    let rescaled = |diff: i32| {
        saturating_rounding_doubling_high_mul(diff.wrapping_mul(1 << left_shift), multiplier)
    };
    let mut output = vec![0i8; input.len()];
    for (row, out) in input.chunks(depth).zip(output.chunks_mut(depth)) {
        let max = row.iter().copied().max().unwrap() as i32;
        let mut sum_of_exps: i32 = 0;
        for &v in row {
            let diff = v as i32 - max;
            if diff >= diff_min {
                let exp = exp_on_negative_values(rescaled(diff), scaled_diff_integer_bits);
                sum_of_exps = sum_of_exps.wrapping_add(rescale(exp, 0, accumulation_integer_bits));
            }
        }
        let (shifted_scale, num_bits_over_unit) =
            get_reciprocal(sum_of_exps, accumulation_integer_bits);
        let exponent = num_bits_over_unit + 31 - 8;
        assert!((0..=31).contains(&exponent));
        for (&v, o) in row.iter().zip(out.iter_mut()) {
            let diff = v as i32 - max;
            *o = if diff >= diff_min {
                let exp = exp_on_negative_values(rescaled(diff), scaled_diff_integer_bits);
                let unsat = rounding_divide_by_pot(fmul(shifted_scale, exp), exponent);
                (unsat - 128).clamp(-128, 127) as i8
            } else {
                -128
            };
        }
    }
    output
}
