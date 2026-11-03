//! `CONV_2D` and `DEPTHWISE_CONV_2D` on `f32` (TFLite reference `Conv` and `DepthwiseConv`, same
//! loop and accumulation order).

use crate::math::mul_add;
use crate::util::{
    check_aligned, clamp_f32, dims4, elements, expect_type, fixed_kernel, float_activation,
    in_slice, input, out_slice, output, pod_struct, shape_mismatch, unsupported, window,
};
use nanoglia_core::backend::EvalCtx;
use nanoglia_core::pod;
use nanoglia_core::{DataType, OpDesc, OpOptions, Result};

pod_struct! {
    /// Geometry and activation of a float convolution.
    pub(crate) struct ConvParams {
        batches: i32,
        in_h: i32,
        in_w: i32,
        in_c: i32,
        filter_h: i32,
        filter_w: i32,
        /// Input channels per filter (`CONV_2D`) or depth multiplier (`DEPTHWISE_CONV_2D`).
        filter_c: i32,
        out_h: i32,
        out_w: i32,
        out_c: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        pad_h: i32,
        pad_w: i32,
        has_bias: u32,
        min: f32,
        max: f32,
    }
}

impl ConvParams {
    fn input_len(&self) -> usize {
        (self.batches * self.in_h * self.in_w * self.in_c) as usize
    }

    fn output_len(&self) -> usize {
        (self.batches * self.out_h * self.out_w * self.out_c) as usize
    }
}

/// Input, filter and output shapes of a convolution, and whether it has a bias.
type ConvShapes = ([i32; 4], [i32; 4], [i32; 4], bool);

/// Validates the tensors common to both convolutions.
fn conv_tensors(op: &OpDesc<'_, '_>) -> Result<ConvShapes> {
    if !(2..=3).contains(&op.inputs.len()) || op.outputs.len() != 1 {
        return Err(unsupported(op));
    }
    let (input, filter, output) = (input(op, 0)?, input(op, 1)?, output(op, 0)?);
    for t in [input, filter, output] {
        expect_type(t, DataType::F32)?;
    }
    check_aligned(filter, 4)?;
    let in_dims = dims4(input)?;
    let filter_dims = dims4(filter)?;
    let out_dims = dims4(output)?;
    if in_dims[0] != out_dims[0] {
        return Err(shape_mismatch(output));
    }
    let has_bias = match op.input(2) {
        Some(bias) => {
            expect_type(bias, DataType::F32)?;
            check_aligned(bias, 4)?;
            if elements(bias)? != out_dims[3] as usize {
                return Err(shape_mismatch(bias));
            }
            true
        }
        None => false,
    };
    Ok((in_dims, filter_dims, out_dims, has_bias))
}

fn conv_params(op: &OpDesc<'_, '_>) -> Result<ConvParams> {
    let OpOptions::Conv2D(o) = op.options else {
        return Err(unsupported(op));
    };
    let (i, f, out, has_bias) = conv_tensors(op)?;
    // Grouped convolutions: the filter covers `in_c / groups` input channels.
    if f[0] != out[3] || f[3] <= 0 || i[3] <= 0 || i[3] % f[3] != 0 || out[3] % (i[3] / f[3]) != 0 {
        return Err(shape_mismatch(input(op, 1)?));
    }
    let output = output(op, 0)?;
    let w = window(
        output,
        o.padding,
        (i[1], i[2]),
        (f[1], f[2]),
        (o.stride_h, o.stride_w),
        (o.dilation_h, o.dilation_w),
        (out[1], out[2]),
    )?;
    let (min, max) = float_activation(op, o.activation)?;
    Ok(ConvParams {
        batches: i[0],
        in_h: i[1],
        in_w: i[2],
        in_c: i[3],
        filter_h: f[1],
        filter_w: f[2],
        filter_c: f[3],
        out_h: out[1],
        out_w: out[2],
        out_c: out[3],
        stride_h: o.stride_h,
        stride_w: o.stride_w,
        dilation_h: o.dilation_h,
        dilation_w: o.dilation_w,
        pad_h: w.pad_h,
        pad_w: w.pad_w,
        has_bias: u32::from(has_bias),
        min,
        max,
    })
}

fn conv_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: ConvParams = *pod::read(ctx.persist)?;
    let input = in_slice::<f32>(ctx.inputs, 0, p.input_len())?;
    let filter_len = (p.out_c * p.filter_h * p.filter_w * p.filter_c) as usize;
    let filter = in_slice::<f32>(ctx.inputs, 1, filter_len)?;
    let bias = if p.has_bias != 0 {
        Some(in_slice::<f32>(ctx.inputs, 2, p.out_c as usize)?)
    } else {
        None
    };
    let output = out_slice::<f32>(ctx.outputs, 0, p.output_len())?;

    let filters_per_group = p.out_c / (p.in_c / p.filter_c);
    let mut o = 0;
    for b in 0..p.batches {
        for out_y in 0..p.out_h {
            let in_y_origin = out_y * p.stride_h - p.pad_h;
            for out_x in 0..p.out_w {
                let in_x_origin = out_x * p.stride_w - p.pad_w;
                for oc in 0..p.out_c {
                    let group = oc / filters_per_group;
                    let mut total = 0.0f32;
                    for fy in 0..p.filter_h {
                        let in_y = in_y_origin + p.dilation_h * fy;
                        for fx in 0..p.filter_w {
                            let in_x = in_x_origin + p.dilation_w * fx;
                            if in_x < 0 || in_x >= p.in_w || in_y < 0 || in_y >= p.in_h {
                                continue;
                            }
                            let in_base = (((b * p.in_h + in_y) * p.in_w + in_x) * p.in_c
                                + group * p.filter_c)
                                as usize;
                            let f_base =
                                (((oc * p.filter_h + fy) * p.filter_w + fx) * p.filter_c) as usize;
                            let n = p.filter_c as usize;
                            for (x, w) in input[in_base..in_base + n]
                                .iter()
                                .zip(&filter[f_base..f_base + n])
                            {
                                total = mul_add(*x, *w, total);
                            }
                        }
                    }
                    let bias = bias.map_or(0.0, |b| b[oc as usize]);
                    output[o] = clamp_f32(total + bias, p.min, p.max);
                    o += 1;
                }
            }
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `CONV_2D` on `f32` (including grouped convolutions).
    CONV_2D_F32,
    "rust/conv_2d_f32",
    conv_params,
    conv_eval
);

fn depthwise_params(op: &OpDesc<'_, '_>) -> Result<ConvParams> {
    let OpOptions::DepthwiseConv2D(o) = op.options else {
        return Err(unsupported(op));
    };
    let (i, f, out, has_bias) = conv_tensors(op)?;
    // TFLite derives the depth multiplier from the shapes.
    if f[0] != 1 || f[3] != out[3] || i[3] <= 0 || out[3] % i[3] != 0 {
        return Err(shape_mismatch(input(op, 1)?));
    }
    let output = output(op, 0)?;
    let w = window(
        output,
        o.padding,
        (i[1], i[2]),
        (f[1], f[2]),
        (o.stride_h, o.stride_w),
        (o.dilation_h, o.dilation_w),
        (out[1], out[2]),
    )?;
    let (min, max) = float_activation(op, o.activation)?;
    Ok(ConvParams {
        batches: i[0],
        in_h: i[1],
        in_w: i[2],
        in_c: i[3],
        filter_h: f[1],
        filter_w: f[2],
        filter_c: out[3] / i[3],
        out_h: out[1],
        out_w: out[2],
        out_c: out[3],
        stride_h: o.stride_h,
        stride_w: o.stride_w,
        dilation_h: o.dilation_h,
        dilation_w: o.dilation_w,
        pad_h: w.pad_h,
        pad_w: w.pad_w,
        has_bias: u32::from(has_bias),
        min,
        max,
    })
}

fn depthwise_eval(ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
    let p: ConvParams = *pod::read(ctx.persist)?;
    let input = in_slice::<f32>(ctx.inputs, 0, p.input_len())?;
    let filter = in_slice::<f32>(ctx.inputs, 1, (p.filter_h * p.filter_w * p.out_c) as usize)?;
    let bias = if p.has_bias != 0 {
        Some(in_slice::<f32>(ctx.inputs, 2, p.out_c as usize)?)
    } else {
        None
    };
    let output = out_slice::<f32>(ctx.outputs, 0, p.output_len())?;
    let depth_multiplier = p.filter_c;
    for b in 0..p.batches {
        for out_y in 0..p.out_h {
            for out_x in 0..p.out_w {
                for ic in 0..p.in_c {
                    for m in 0..depth_multiplier {
                        let oc = m + ic * depth_multiplier;
                        let in_x_origin = out_x * p.stride_w - p.pad_w;
                        let in_y_origin = out_y * p.stride_h - p.pad_h;
                        let mut total = 0.0f32;
                        for fy in 0..p.filter_h {
                            for fx in 0..p.filter_w {
                                let in_x = in_x_origin + p.dilation_w * fx;
                                let in_y = in_y_origin + p.dilation_h * fy;
                                if in_x >= 0 && in_x < p.in_w && in_y >= 0 && in_y < p.in_h {
                                    let x = input[(((b * p.in_h + in_y) * p.in_w + in_x) * p.in_c
                                        + ic)
                                        as usize];
                                    let w =
                                        filter[((fy * p.filter_w + fx) * p.out_c + oc) as usize];
                                    total = mul_add(x, w, total);
                                }
                            }
                        }
                        let bias = bias.map_or(0.0, |b| b[oc as usize]);
                        let o = (((b * p.out_h + out_y) * p.out_w + out_x) * p.out_c + oc) as usize;
                        output[o] = clamp_f32(total + bias, p.min, p.max);
                    }
                }
            }
        }
    }
    Ok(())
}

fixed_kernel!(
    /// `DEPTHWISE_CONV_2D` on `f32`.
    DEPTHWISE_CONV_2D_F32,
    "rust/depthwise_conv_2d_f32",
    depthwise_params,
    depthwise_eval
);
