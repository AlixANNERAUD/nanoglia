#![no_std]
#![warn(missing_docs)]
#![allow(non_camel_case_types)]
#![doc = include_str!("../README.md")]

use core::ffi::{c_int, c_void};

mod lock;

pub use lock::{LOCK_AVAILABLE, ScratchLock, try_lock};

/// Whether the dispatch functions (`nanoglia_esp_nn_*`) run ESP-NN's ESP32-S3 optimized kernels
/// (`true`) or its ANSI C reference kernels (`false`).
pub const OPTIMIZED_ESP32S3: bool = cfg!(esp_nn_esp32s3);

/// Width, height, channels and an extra dimension (batches, or unused) of a tensor.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct data_dims_t {
    /// Width.
    pub width: i32,
    /// Height.
    pub height: i32,
    /// Channels.
    pub channels: i32,
    /// Batches or any other parameter.
    pub extra: i32,
}

/// Width and height of strides, paddings and dilations.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct data_2d_t {
    /// Width.
    pub width: i32,
    /// Height.
    pub height: i32,
}

/// Clamping range of a fused activation, in the output's quantized domain.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct act_params_t {
    /// Lower bound.
    pub min: i32,
    /// Upper bound.
    pub max: i32,
}

/// Per-channel requantization parameters: one shift and one multiplier per output channel.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct quant_data_t {
    /// Shifts (positive: left).
    pub shift: *mut i32,
    /// Q31 multipliers.
    pub mult: *mut i32,
}

/// Convolution parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct conv_params_t {
    /// Negated input zero point.
    pub in_offset: i32,
    /// Output zero point.
    pub out_offset: i32,
    /// Strides.
    pub stride: data_2d_t,
    /// Paddings (left/top).
    pub padding: data_2d_t,
    /// Dilations (ignored: ESP-NN only runs undilated convolutions).
    pub dilation: data_2d_t,
    /// Fused activation range.
    pub activation: act_params_t,
}

/// Depthwise convolution parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct dw_conv_params_t {
    /// Negated input zero point.
    pub in_offset: i32,
    /// Output zero point.
    pub out_offset: i32,
    /// Channel multiplier: `in_ch * ch_mult = out_ch`.
    pub ch_mult: i32,
    /// Strides.
    pub stride: data_2d_t,
    /// Paddings (left/top).
    pub padding: data_2d_t,
    /// Dilations (ignored: ESP-NN only runs undilated convolutions).
    pub dilation: data_2d_t,
    /// Fused activation range.
    pub activation: act_params_t,
}

// The functions selected by the build: ESP32-S3 optimized kernels or ANSI C reference kernels
// (`src/dispatch.c`).
//
// The scratch buffers of ESP-NN are global variables: a `set_*_scratch_buf` call and the kernel
// call that uses it must not interleave with another context running ESP-NN. Hold a
// [`ScratchLock`] around them.
unsafe extern "C" {
    /// Scratch bytes `nanoglia_esp_nn_conv_s8` needs for these dimensions and parameters.
    pub fn nanoglia_esp_nn_get_conv_scratch_size(
        input_dims: *const data_dims_t,
        filter_dims: *const data_dims_t,
        output_dims: *const data_dims_t,
        conv_params: *const conv_params_t,
    ) -> c_int;
    /// Sets the (global) scratch buffer of `nanoglia_esp_nn_conv_s8`; 16-byte aligned.
    pub fn nanoglia_esp_nn_set_conv_scratch_buf(buf: *mut c_void);
    /// int8 convolution of one batch, per-channel requantization, `NHWC` input and output,
    /// `OHWI` filter. A filter with fewer channels than the input runs a grouped convolution.
    pub fn nanoglia_esp_nn_conv_s8(
        input_dims: *const data_dims_t,
        input_data: *const i8,
        filter_dims: *const data_dims_t,
        filter_data: *const i8,
        bias: *const i32,
        output_dims: *const data_dims_t,
        out_data: *mut i8,
        conv_params: *const conv_params_t,
        quant_data: *const quant_data_t,
    );

    /// Scratch bytes `nanoglia_esp_nn_depthwise_conv_s8` needs.
    pub fn nanoglia_esp_nn_get_depthwise_conv_scratch_size(
        input_dims: *const data_dims_t,
        filter_dims: *const data_dims_t,
        output_dims: *const data_dims_t,
        conv_params: *const dw_conv_params_t,
    ) -> c_int;
    /// Sets the (global) scratch buffer of `nanoglia_esp_nn_depthwise_conv_s8`; 16-byte aligned.
    pub fn nanoglia_esp_nn_set_depthwise_conv_scratch_buf(buf: *mut c_void);
    /// int8 depthwise convolution of one batch, per-channel requantization, `1HWO` filter.
    pub fn nanoglia_esp_nn_depthwise_conv_s8(
        input_dims: *const data_dims_t,
        input_data: *const i8,
        filter_dims: *const data_dims_t,
        filter_data: *const i8,
        bias: *const i32,
        output_dims: *const data_dims_t,
        out_data: *mut i8,
        conv_params: *const dw_conv_params_t,
        quant_data: *const quant_data_t,
    );

    /// int8 fully-connected layer on one row, per-tensor requantization.
    pub fn nanoglia_esp_nn_fully_connected_s8(
        input_data: *const i8,
        input_offset: i32,
        row_len: u16,
        filter_data: *const i8,
        filter_offset: i32,
        bias: *const i32,
        out_data: *mut i8,
        out_channels: u16,
        out_offset: i32,
        out_shift: i32,
        out_mult: i32,
        activation_min: i32,
        activation_max: i32,
    );
    /// int8 fully-connected layer on `batches` rows, per-channel requantization.
    pub fn nanoglia_esp_nn_fully_connected_per_ch_s8_batch(
        input_data: *const i8,
        input_offset: i32,
        row_len: u16,
        filter_data: *const i8,
        filter_offset: i32,
        bias: *const i32,
        out_data: *mut i8,
        out_channels: u16,
        out_offset: i32,
        out_shift: *const i32,
        out_mult: *const i32,
        activation_min: i32,
        activation_max: i32,
        batches: i32,
    );

    /// int8 average pooling of one batch. The ESP32-S3 kernel needs a multiple of 4 channels.
    pub fn nanoglia_esp_nn_avg_pool_s8(
        input: *const i8,
        input_wd: u16,
        input_ht: u16,
        output: *mut i8,
        output_wd: u16,
        output_ht: u16,
        stride_wd: u16,
        stride_ht: u16,
        filter_wd: u16,
        filter_ht: u16,
        pad_wd: u16,
        pad_ht: u16,
        activation_min: i32,
        activation_max: i32,
        channels: u16,
    );
    /// int8 max pooling of one batch. The ESP32-S3 kernel needs a multiple of 4 channels.
    pub fn nanoglia_esp_nn_max_pool_s8(
        input: *const i8,
        input_wd: u16,
        input_ht: u16,
        output: *mut i8,
        output_wd: u16,
        output_ht: u16,
        stride_wd: u16,
        stride_ht: u16,
        filter_wd: u16,
        filter_ht: u16,
        pad_wd: u16,
        pad_ht: u16,
        activation_min: i32,
        activation_max: i32,
        channels: u16,
    );

    /// Scratch bytes `nanoglia_esp_nn_softmax_s8` needs for `height` rows of `width` values.
    pub fn nanoglia_esp_nn_get_softmax_scratch_size(width: i32, height: i32) -> i32;
    /// Sets the (global) scratch buffer of `nanoglia_esp_nn_softmax_s8`; 16-byte aligned.
    pub fn nanoglia_esp_nn_set_softmax_scratch_buf(buffer: *mut c_void);
    /// int8 softmax over rows of `width` values, output quantized with scale 1/256 and zero point
    /// -128.
    pub fn nanoglia_esp_nn_softmax_s8(
        input_data: *const i8,
        height: i32,
        width: i32,
        mult: i32,
        shift: i32,
        diff_min: i32,
        output_data: *mut i8,
    );

    /// int8 elementwise addition (TFLite `ADD` arithmetic, shifts `<= 0`).
    pub fn nanoglia_esp_nn_add_elementwise_s8(
        input1_data: *const i8,
        input2_data: *const i8,
        input1_offset: i32,
        input2_offset: i32,
        input1_mult: i32,
        input2_mult: i32,
        input1_shift: i32,
        input2_shift: i32,
        left_shift: i32,
        output: *mut i8,
        out_offset: i32,
        out_mult: i32,
        out_shift: i32,
        activation_min: i32,
        activation_max: i32,
        size: i32,
    );
    /// int8 elementwise multiplication.
    pub fn nanoglia_esp_nn_mul_elementwise_s8(
        input1_data: *const i8,
        input2_data: *const i8,
        input1_offset: i32,
        input2_offset: i32,
        output: *mut i8,
        out_offset: i32,
        out_mult: i32,
        out_shift: i32,
        activation_min: i32,
        activation_max: i32,
        size: i32,
    );
}

// The ANSI C reference kernels, built on every target. They do not use the global scratch
// buffers.
unsafe extern "C" {
    /// Reference implementation of [`nanoglia_esp_nn_conv_s8`].
    pub fn esp_nn_conv_s8_ansi(
        input_dims: *const data_dims_t,
        input_data: *const i8,
        filter_dims: *const data_dims_t,
        filter_data: *const i8,
        bias: *const i32,
        output_dims: *const data_dims_t,
        out_data: *mut i8,
        conv_params: *const conv_params_t,
        quant_data: *const quant_data_t,
    );
    /// Reference implementation of [`nanoglia_esp_nn_depthwise_conv_s8`].
    pub fn esp_nn_depthwise_conv_s8_ansi(
        input_dims: *const data_dims_t,
        input_data: *const i8,
        filter_dims: *const data_dims_t,
        filter_data: *const i8,
        bias: *const i32,
        output_dims: *const data_dims_t,
        out_data: *mut i8,
        conv_params: *const dw_conv_params_t,
        quant_data: *const quant_data_t,
    );
    /// Reference implementation of [`nanoglia_esp_nn_fully_connected_s8`].
    pub fn esp_nn_fully_connected_s8_ansi(
        input_data: *const i8,
        input_offset: i32,
        row_len: u16,
        filter_data: *const i8,
        filter_offset: i32,
        bias: *const i32,
        out_data: *mut i8,
        out_channels: u16,
        out_offset: i32,
        out_shift: i32,
        out_mult: i32,
        activation_min: i32,
        activation_max: i32,
    );
    /// Reference implementation of [`nanoglia_esp_nn_fully_connected_per_ch_s8_batch`] for one
    /// row.
    pub fn esp_nn_fully_connected_per_ch_s8_ansi(
        input_data: *const i8,
        input_offset: i32,
        row_len: u16,
        filter_data: *const i8,
        filter_offset: i32,
        bias: *const i32,
        out_data: *mut i8,
        out_channels: u16,
        out_offset: i32,
        out_shift: *const i32,
        out_mult: *const i32,
        activation_min: i32,
        activation_max: i32,
    );
    /// Reference implementation of [`nanoglia_esp_nn_avg_pool_s8`], for any channel count.
    pub fn esp_nn_avg_pool_s8_ansi(
        input: *const i8,
        input_wd: u16,
        input_ht: u16,
        output: *mut i8,
        output_wd: u16,
        output_ht: u16,
        stride_wd: u16,
        stride_ht: u16,
        filter_wd: u16,
        filter_ht: u16,
        pad_wd: u16,
        pad_ht: u16,
        activation_min: i32,
        activation_max: i32,
        channels: u16,
    );
    /// Reference implementation of [`nanoglia_esp_nn_max_pool_s8`], for any channel count.
    pub fn esp_nn_max_pool_s8_ansi(
        input: *const i8,
        input_wd: u16,
        input_ht: u16,
        output: *mut i8,
        output_wd: u16,
        output_ht: u16,
        stride_wd: u16,
        stride_ht: u16,
        filter_wd: u16,
        filter_ht: u16,
        pad_wd: u16,
        pad_ht: u16,
        activation_min: i32,
        activation_max: i32,
        channels: u16,
    );
    /// Reference implementation of [`nanoglia_esp_nn_softmax_s8`].
    pub fn esp_nn_softmax_s8_ansi(
        input_data: *const i8,
        height: i32,
        width: i32,
        mult: i32,
        shift: i32,
        diff_min: i32,
        output_data: *mut i8,
    );
    /// Reference implementation of [`nanoglia_esp_nn_add_elementwise_s8`].
    pub fn esp_nn_add_elementwise_s8_ansi(
        input1_data: *const i8,
        input2_data: *const i8,
        input1_offset: i32,
        input2_offset: i32,
        input1_mult: i32,
        input2_mult: i32,
        input1_shift: i32,
        input2_shift: i32,
        left_shift: i32,
        output: *mut i8,
        out_offset: i32,
        out_mult: i32,
        out_shift: i32,
        activation_min: i32,
        activation_max: i32,
        size: i32,
    );
    /// Reference implementation of [`nanoglia_esp_nn_mul_elementwise_s8`].
    pub fn esp_nn_mul_elementwise_s8_ansi(
        input1_data: *const i8,
        input2_data: *const i8,
        input1_offset: i32,
        input2_offset: i32,
        output: *mut i8,
        out_offset: i32,
        out_mult: i32,
        out_shift: i32,
        activation_min: i32,
        activation_max: i32,
        size: i32,
    );

    /// Entry `index` of the size/alignment/offset table of the C structs (`src/layout.c`),
    /// `u32::MAX` past the end. Used by the layout tests.
    #[doc(hidden)]
    pub fn nanoglia_esp_nn_layout(index: u32) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    #[test]
    fn struct_layouts_match_c() {
        let expected: &[usize] = &[
            size_of::<data_dims_t>(),
            align_of::<data_dims_t>(),
            offset_of!(data_dims_t, height),
            offset_of!(data_dims_t, channels),
            offset_of!(data_dims_t, extra),
            size_of::<data_2d_t>(),
            align_of::<data_2d_t>(),
            offset_of!(data_2d_t, height),
            size_of::<act_params_t>(),
            align_of::<act_params_t>(),
            offset_of!(act_params_t, max),
            size_of::<quant_data_t>(),
            align_of::<quant_data_t>(),
            offset_of!(quant_data_t, mult),
            size_of::<conv_params_t>(),
            align_of::<conv_params_t>(),
            offset_of!(conv_params_t, out_offset),
            offset_of!(conv_params_t, stride),
            offset_of!(conv_params_t, padding),
            offset_of!(conv_params_t, dilation),
            offset_of!(conv_params_t, activation),
            size_of::<dw_conv_params_t>(),
            align_of::<dw_conv_params_t>(),
            offset_of!(dw_conv_params_t, ch_mult),
            offset_of!(dw_conv_params_t, stride),
            offset_of!(dw_conv_params_t, padding),
            offset_of!(dw_conv_params_t, dilation),
            offset_of!(dw_conv_params_t, activation),
        ];
        for (index, &want) in expected.iter().enumerate() {
            // SAFETY: pure function reading a constant table.
            let got = unsafe { nanoglia_esp_nn_layout(index as u32) };
            assert_eq!(got as usize, want, "layout entry {index}");
        }
        // SAFETY: as above.
        let end = unsafe { nanoglia_esp_nn_layout(expected.len() as u32) };
        assert_eq!(end, u32::MAX, "the C table has more entries than the test");
    }

    #[test]
    fn dispatch_matches_reference() {
        // A 1x1 fully-connected layer through both entry points.
        let input = [1i8, -2, 3, -4];
        let filter = [5i8, 6, -7, 8, -9, 10, 11, -12];
        let bias = [100i32, -100];
        let (mut a, mut b) = ([0i8; 2], [0i8; 2]);
        // SAFETY: the buffers have the sizes the arguments describe.
        unsafe {
            nanoglia_esp_nn_fully_connected_s8(
                input.as_ptr(),
                3,
                4,
                filter.as_ptr(),
                0,
                bias.as_ptr(),
                a.as_mut_ptr(),
                2,
                -5,
                -6,
                1 << 30,
                -128,
                127,
            );
            esp_nn_fully_connected_s8_ansi(
                input.as_ptr(),
                3,
                4,
                filter.as_ptr(),
                0,
                bias.as_ptr(),
                b.as_mut_ptr(),
                2,
                -5,
                -6,
                1 << 30,
                -128,
                127,
            );
        }
        assert_eq!(a, b);
        assert_ne!(a, [0, 0]);
    }

    #[test]
    fn lock_is_exclusive() {
        let first = try_lock().expect("free");
        assert!(try_lock().is_none());
        drop(first);
        assert!(try_lock().is_some());
    }
}
