#![no_std]
#![warn(missing_docs)]
#![allow(non_camel_case_types)]
#![doc = include_str!("../README.md")]

use core::ffi::{c_int, c_void};

/// Status code returned by the CMSIS-NN functions (`arm_cmsis_nn_status`).
pub type arm_cmsis_nn_status = i32;
/// No error.
pub const ARM_CMSIS_NN_SUCCESS: arm_cmsis_nn_status = 0;
/// One or more arguments are incorrect.
pub const ARM_CMSIS_NN_ARG_ERROR: arm_cmsis_nn_status = -1;
/// No implementation available.
pub const ARM_CMSIS_NN_NO_IMPL_ERROR: arm_cmsis_nn_status = -2;
/// Logical error.
pub const ARM_CMSIS_NN_FAILURE: arm_cmsis_nn_status = -3;

/// Width and height of a tile (strides, paddings, dilations).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_tile {
    /// Width.
    pub w: i32,
    /// Height.
    pub h: i32,
}

/// Function context: the scratch buffer of a kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cmsis_nn_context {
    /// Scratch buffer, as large as the matching `*_get_buffer_size` function asks.
    pub buf: *mut c_void,
    /// Size of the buffer in bytes (informative; the kernels do not read it).
    pub size: i32,
}

/// Bias of the int16 kernels: either `int32_t` or `int64_t` values.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cmsis_nn_bias_data {
    /// Bias values, or null.
    pub data: *const c_void,
    /// `true` for `int32_t` values, `false` for `int64_t`.
    pub is_int32_bias: bool,
}

/// Tensor dimensions, in NHWC order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_dims {
    /// Batches, or output channels for some filters.
    pub n: i32,
    /// Height.
    pub h: i32,
    /// Width.
    pub w: i32,
    /// Channels.
    pub c: i32,
}

/// Per-channel requantization parameters: one multiplier and shift per output channel.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cmsis_nn_per_channel_quant_params {
    /// Q31 multipliers.
    pub multiplier: *mut i32,
    /// Shifts (positive: left).
    pub shift: *mut i32,
}

/// Per-tensor requantization parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_per_tensor_quant_params {
    /// Q31 multiplier.
    pub multiplier: i32,
    /// Shift (positive: left).
    pub shift: i32,
}

/// Per-tensor or per-channel requantization parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cmsis_nn_quant_params {
    /// Q31 multipliers (one, or one per output channel).
    pub multiplier: *mut i32,
    /// Shifts (one, or one per output channel).
    pub shift: *mut i32,
    /// Non-zero for per-channel parameters.
    pub is_per_channel: i32,
}

/// Clamping range of a fused activation, in the output's quantized domain.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_activation {
    /// Lower bound.
    pub min: i32,
    /// Upper bound.
    pub max: i32,
}

/// Convolution parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_conv_params {
    /// Negated input zero point.
    pub input_offset: i32,
    /// Output zero point.
    pub output_offset: i32,
    /// Strides.
    pub stride: cmsis_nn_tile,
    /// Paddings (top/left).
    pub padding: cmsis_nn_tile,
    /// Dilations.
    pub dilation: cmsis_nn_tile,
    /// Fused activation range.
    pub activation: cmsis_nn_activation,
}

/// Transposed convolution parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_transpose_conv_params {
    /// Negated input zero point.
    pub input_offset: i32,
    /// Output zero point.
    pub output_offset: i32,
    /// Strides.
    pub stride: cmsis_nn_tile,
    /// Paddings.
    pub padding: cmsis_nn_tile,
    /// Padding offsets.
    pub padding_offsets: cmsis_nn_tile,
    /// Dilations.
    pub dilation: cmsis_nn_tile,
    /// Fused activation range.
    pub activation: cmsis_nn_activation,
}

/// Depthwise convolution parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_dw_conv_params {
    /// Negated input zero point.
    pub input_offset: i32,
    /// Output zero point.
    pub output_offset: i32,
    /// Channel multiplier: `ch_mult * in_ch = out_ch`.
    pub ch_mult: i32,
    /// Strides.
    pub stride: cmsis_nn_tile,
    /// Paddings (top/left).
    pub padding: cmsis_nn_tile,
    /// Dilations.
    pub dilation: cmsis_nn_tile,
    /// Fused activation range.
    pub activation: cmsis_nn_activation,
}

/// Pooling parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_pool_params {
    /// Strides.
    pub stride: cmsis_nn_tile,
    /// Paddings (top/left).
    pub padding: cmsis_nn_tile,
    /// Fused activation range.
    pub activation: cmsis_nn_activation,
}

/// Fully-connected parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_fc_params {
    /// Negated input zero point.
    pub input_offset: i32,
    /// Negated filter zero point.
    pub filter_offset: i32,
    /// Output zero point.
    pub output_offset: i32,
    /// Fused activation range.
    pub activation: cmsis_nn_activation,
}

/// Batch matrix multiplication parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_bmm_params {
    /// Whether the left operand is transposed.
    pub adj_x: bool,
    /// Whether the right operand is transposed.
    pub adj_y: bool,
    /// Offsets and activation.
    pub fc_params: cmsis_nn_fc_params,
}

/// Transpose parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cmsis_nn_transpose_params {
    /// Number of dimensions (up to 4).
    pub num_dims: i32,
    /// Permutation applied to the input dimensions.
    pub permutations: *const u32,
}

/// SVDF parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_svdf_params {
    /// Rank.
    pub rank: i32,
    /// Negated input zero point.
    pub input_offset: i32,
    /// Output zero point.
    pub output_offset: i32,
    /// Activation range of the input stage.
    pub input_activation: cmsis_nn_activation,
    /// Activation range of the output stage.
    pub output_activation: cmsis_nn_activation,
}

/// Lookup tables of the int16 softmax.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cmsis_nn_softmax_lut_s16 {
    /// `exp(x)` for `x` in `[-10, 0]`, 513 entries.
    pub exp_lut: *const i16,
    /// `1 / (1 + x)` for `x` in `[0, 1]`, 513 entries.
    pub one_by_one_lut: *const i16,
}

/// A multiplier and shift pair.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cmsis_nn_scaling {
    /// Q31 multiplier.
    pub multiplier: i32,
    /// Shift (positive: left).
    pub shift: i32,
}

unsafe extern "C" {
    // ---------------------------------------------------------------- Convolution

    /// Convolution, int8 activations and weights, int32 bias; picks the best CMSIS-NN kernel.
    pub fn arm_convolve_wrapper_s8(
        ctx: *const cmsis_nn_context,
        conv_params: *const cmsis_nn_conv_params,
        quant_params: *const cmsis_nn_per_channel_quant_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i8,
        filter_dims: *const cmsis_nn_dims,
        filter_data: *const i8,
        bias_dims: *const cmsis_nn_dims,
        bias_data: *const i32,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i8,
    ) -> arm_cmsis_nn_status;

    /// Scratch bytes needed by [`arm_convolve_wrapper_s8`].
    pub fn arm_convolve_wrapper_s8_get_buffer_size(
        conv_params: *const cmsis_nn_conv_params,
        input_dims: *const cmsis_nn_dims,
        filter_dims: *const cmsis_nn_dims,
        output_dims: *const cmsis_nn_dims,
    ) -> i32;

    /// Convolution, int16 activations, int8 weights, int32 or int64 bias.
    pub fn arm_convolve_wrapper_s16(
        ctx: *const cmsis_nn_context,
        conv_params: *const cmsis_nn_conv_params,
        quant_params: *const cmsis_nn_per_channel_quant_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i16,
        filter_dims: *const cmsis_nn_dims,
        filter_data: *const i8,
        bias_dims: *const cmsis_nn_dims,
        bias_data: *const cmsis_nn_bias_data,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i16,
    ) -> arm_cmsis_nn_status;

    /// Scratch bytes needed by [`arm_convolve_wrapper_s16`].
    pub fn arm_convolve_wrapper_s16_get_buffer_size(
        conv_params: *const cmsis_nn_conv_params,
        input_dims: *const cmsis_nn_dims,
        filter_dims: *const cmsis_nn_dims,
        output_dims: *const cmsis_nn_dims,
    ) -> i32;

    /// Transpose convolution, int8 activations and weights, int32 bias: a convolution of the
    /// upsampled input with the reversed filter for strides up to 2 and more than 16 input
    /// channels, `arm_transpose_conv_s8` otherwise.
    pub fn arm_transpose_conv_wrapper_s8(
        ctx: *const cmsis_nn_context,
        reverse_conv_ctx: *const cmsis_nn_context,
        transpose_conv_params: *const cmsis_nn_transpose_conv_params,
        quant_params: *const cmsis_nn_per_channel_quant_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i8,
        filter_dims: *const cmsis_nn_dims,
        filter_data: *const i8,
        bias_dims: *const cmsis_nn_dims,
        bias_data: *const i32,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i8,
    ) -> arm_cmsis_nn_status;

    /// Bytes of the `ctx` buffer of [`arm_transpose_conv_wrapper_s8`].
    pub fn arm_transpose_conv_s8_get_buffer_size(
        transpose_conv_params: *const cmsis_nn_transpose_conv_params,
        input_dims: *const cmsis_nn_dims,
        filter_dims: *const cmsis_nn_dims,
        out_dims: *const cmsis_nn_dims,
    ) -> i32;

    /// Bytes of the `reverse_conv_ctx` buffer of [`arm_transpose_conv_wrapper_s8`] (the reversed
    /// filter), 0 when it is not used.
    pub fn arm_transpose_conv_s8_get_reverse_conv_buffer_size(
        transpose_conv_params: *const cmsis_nn_transpose_conv_params,
        input_dims: *const cmsis_nn_dims,
        filter_dims: *const cmsis_nn_dims,
    ) -> i32;

    /// 1x1 convolution with unit strides and no padding, int8 (also a batched fully-connected).
    pub fn arm_convolve_1x1_s8_fast(
        ctx: *const cmsis_nn_context,
        conv_params: *const cmsis_nn_conv_params,
        quant_params: *const cmsis_nn_per_channel_quant_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i8,
        filter_dims: *const cmsis_nn_dims,
        filter_data: *const i8,
        bias_dims: *const cmsis_nn_dims,
        bias_data: *const i32,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i8,
    ) -> arm_cmsis_nn_status;

    /// Scratch bytes needed by [`arm_convolve_1x1_s8_fast`].
    pub fn arm_convolve_1x1_s8_fast_get_buffer_size(input_dims: *const cmsis_nn_dims) -> i32;

    // ---------------------------------------------------------------- Depthwise convolution

    /// Depthwise convolution, int8 activations and weights, int32 bias.
    pub fn arm_depthwise_conv_wrapper_s8(
        ctx: *const cmsis_nn_context,
        dw_conv_params: *const cmsis_nn_dw_conv_params,
        quant_params: *const cmsis_nn_per_channel_quant_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i8,
        filter_dims: *const cmsis_nn_dims,
        filter_data: *const i8,
        bias_dims: *const cmsis_nn_dims,
        bias_data: *const i32,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i8,
    ) -> arm_cmsis_nn_status;

    /// Scratch bytes needed by [`arm_depthwise_conv_wrapper_s8`].
    pub fn arm_depthwise_conv_wrapper_s8_get_buffer_size(
        dw_conv_params: *const cmsis_nn_dw_conv_params,
        input_dims: *const cmsis_nn_dims,
        filter_dims: *const cmsis_nn_dims,
        output_dims: *const cmsis_nn_dims,
    ) -> i32;

    /// Depthwise convolution, int16 activations, int8 weights, int64 bias.
    pub fn arm_depthwise_conv_wrapper_s16(
        ctx: *const cmsis_nn_context,
        dw_conv_params: *const cmsis_nn_dw_conv_params,
        quant_params: *const cmsis_nn_per_channel_quant_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i16,
        filter_dims: *const cmsis_nn_dims,
        filter_data: *const i8,
        bias_dims: *const cmsis_nn_dims,
        bias_data: *const i64,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i16,
    ) -> arm_cmsis_nn_status;

    /// Scratch bytes needed by [`arm_depthwise_conv_wrapper_s16`].
    pub fn arm_depthwise_conv_wrapper_s16_get_buffer_size(
        dw_conv_params: *const cmsis_nn_dw_conv_params,
        input_dims: *const cmsis_nn_dims,
        filter_dims: *const cmsis_nn_dims,
        output_dims: *const cmsis_nn_dims,
    ) -> i32;

    // ---------------------------------------------------------------- Fully connected

    /// Fully connected, int8, per-tensor requantization.
    ///
    /// `ctx.buf` holds the kernel sums computed by [`arm_vector_sum_s8`] when
    /// [`arm_fully_connected_s8_get_buffer_size`] is non-zero (Helium), and may be null otherwise.
    pub fn arm_fully_connected_s8(
        ctx: *const cmsis_nn_context,
        fc_params: *const cmsis_nn_fc_params,
        quant_params: *const cmsis_nn_per_tensor_quant_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i8,
        filter_dims: *const cmsis_nn_dims,
        filter_data: *const i8,
        bias_dims: *const cmsis_nn_dims,
        bias_data: *const i32,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i8,
    ) -> arm_cmsis_nn_status;

    /// Fully connected, int8, per-channel requantization. Same buffer rules as
    /// [`arm_fully_connected_s8`].
    pub fn arm_fully_connected_per_channel_s8(
        ctx: *const cmsis_nn_context,
        fc_params: *const cmsis_nn_fc_params,
        quant_params: *const cmsis_nn_per_channel_quant_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i8,
        filter_dims: *const cmsis_nn_dims,
        filter_data: *const i8,
        bias_dims: *const cmsis_nn_dims,
        bias_data: *const i32,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i8,
    ) -> arm_cmsis_nn_status;

    /// Fully connected, int8, per-tensor or per-channel requantization.
    pub fn arm_fully_connected_wrapper_s8(
        ctx: *const cmsis_nn_context,
        fc_params: *const cmsis_nn_fc_params,
        quant_params: *const cmsis_nn_quant_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i8,
        filter_dims: *const cmsis_nn_dims,
        filter_data: *const i8,
        bias_dims: *const cmsis_nn_dims,
        bias_data: *const i32,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i8,
    ) -> arm_cmsis_nn_status;

    /// Kernel sums of a fully-connected layer: for each of the `vector_rows` rows,
    /// `lhs_offset * sum(row + rhs_offset)`, plus `bias_data[row]` if `bias_data` is not null.
    pub fn arm_vector_sum_s8(
        vector_sum_buf: *mut i32,
        vector_cols: i32,
        vector_rows: i32,
        vector_data: *const i8,
        lhs_offset: i32,
        rhs_offset: i32,
        bias_data: *const i32,
    ) -> arm_cmsis_nn_status;

    /// Bytes of kernel sums needed by [`arm_fully_connected_s8`] (`filter_dims.c` is the output
    /// depth).
    pub fn arm_fully_connected_s8_get_buffer_size(filter_dims: *const cmsis_nn_dims) -> i32;

    /// Fully connected, int16 activations, int8 weights, int64 bias.
    pub fn arm_fully_connected_s16(
        ctx: *const cmsis_nn_context,
        fc_params: *const cmsis_nn_fc_params,
        quant_params: *const cmsis_nn_per_tensor_quant_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i16,
        filter_dims: *const cmsis_nn_dims,
        filter_data: *const i8,
        bias_dims: *const cmsis_nn_dims,
        bias_data: *const i64,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i16,
    ) -> arm_cmsis_nn_status;

    /// Scratch bytes needed by [`arm_fully_connected_s16`].
    pub fn arm_fully_connected_s16_get_buffer_size(filter_dims: *const cmsis_nn_dims) -> i32;

    // ---------------------------------------------------------------- Elementwise

    /// Elementwise addition of two int8 vectors of `block_size` elements.
    pub fn arm_elementwise_add_s8(
        input_1_vect: *const i8,
        input_2_vect: *const i8,
        input_1_offset: i32,
        input_1_mult: i32,
        input_1_shift: i32,
        input_2_offset: i32,
        input_2_mult: i32,
        input_2_shift: i32,
        left_shift: i32,
        output: *mut i8,
        out_offset: i32,
        out_mult: i32,
        out_shift: i32,
        out_activation_min: i32,
        out_activation_max: i32,
        block_size: i32,
    ) -> arm_cmsis_nn_status;

    /// Elementwise addition of two int16 vectors of `block_size` elements.
    pub fn arm_elementwise_add_s16(
        input_1_vect: *const i16,
        input_2_vect: *const i16,
        input_1_offset: i32,
        input_1_mult: i32,
        input_1_shift: i32,
        input_2_offset: i32,
        input_2_mult: i32,
        input_2_shift: i32,
        left_shift: i32,
        output: *mut i16,
        out_offset: i32,
        out_mult: i32,
        out_shift: i32,
        out_activation_min: i32,
        out_activation_max: i32,
        block_size: i32,
    ) -> arm_cmsis_nn_status;

    /// Elementwise multiplication of two int8 vectors of `block_size` elements.
    pub fn arm_elementwise_mul_s8(
        input_1_vect: *const i8,
        input_2_vect: *const i8,
        input_1_offset: i32,
        input_2_offset: i32,
        output: *mut i8,
        out_offset: i32,
        out_mult: i32,
        out_shift: i32,
        out_activation_min: i32,
        out_activation_max: i32,
        block_size: i32,
    ) -> arm_cmsis_nn_status;

    /// Elementwise multiplication of two int16 vectors of `block_size` elements.
    pub fn arm_elementwise_mul_s16(
        input_1_vect: *const i16,
        input_2_vect: *const i16,
        input_1_offset: i32,
        input_2_offset: i32,
        output: *mut i16,
        out_offset: i32,
        out_mult: i32,
        out_shift: i32,
        out_activation_min: i32,
        out_activation_max: i32,
        block_size: i32,
    ) -> arm_cmsis_nn_status;

    /// Elementwise minimum of two int8 tensors, with broadcasting.
    pub fn arm_minimum_s8(
        ctx: *const cmsis_nn_context,
        input_1_data: *const i8,
        input_1_dims: *const cmsis_nn_dims,
        input_2_data: *const i8,
        input_2_dims: *const cmsis_nn_dims,
        output_data: *mut i8,
        output_dims: *const cmsis_nn_dims,
    ) -> arm_cmsis_nn_status;

    /// Elementwise maximum of two int8 tensors, with broadcasting.
    pub fn arm_maximum_s8(
        ctx: *const cmsis_nn_context,
        input_1_data: *const i8,
        input_1_dims: *const cmsis_nn_dims,
        input_2_data: *const i8,
        input_2_dims: *const cmsis_nn_dims,
        output_data: *mut i8,
        output_dims: *const cmsis_nn_dims,
    ) -> arm_cmsis_nn_status;

    // ---------------------------------------------------------------- Activations

    /// In-place ReLU on raw int8 values.
    pub fn arm_relu_q7(data: *mut i8, size: u16);

    /// In-place clamp of raw int8 values to `[0, 6]`.
    pub fn arm_relu6_s8(data: *mut i8, size: u16);

    // ---------------------------------------------------------------- Pooling

    /// Average pooling, int8.
    pub fn arm_avgpool_s8(
        ctx: *const cmsis_nn_context,
        pool_params: *const cmsis_nn_pool_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i8,
        filter_dims: *const cmsis_nn_dims,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i8,
    ) -> arm_cmsis_nn_status;

    /// Scratch bytes needed by [`arm_avgpool_s8`].
    pub fn arm_avgpool_s8_get_buffer_size(dim_dst_width: c_int, ch_src: c_int) -> i32;

    /// Average pooling, int16.
    pub fn arm_avgpool_s16(
        ctx: *const cmsis_nn_context,
        pool_params: *const cmsis_nn_pool_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i16,
        filter_dims: *const cmsis_nn_dims,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i16,
    ) -> arm_cmsis_nn_status;

    /// Scratch bytes needed by [`arm_avgpool_s16`].
    pub fn arm_avgpool_s16_get_buffer_size(dim_dst_width: c_int, ch_src: c_int) -> i32;

    /// Max pooling, int8. Input and output must not overlap.
    pub fn arm_max_pool_s8(
        ctx: *const cmsis_nn_context,
        pool_params: *const cmsis_nn_pool_params,
        input_dims: *const cmsis_nn_dims,
        input_data: *const i8,
        filter_dims: *const cmsis_nn_dims,
        output_dims: *const cmsis_nn_dims,
        output_data: *mut i8,
    ) -> arm_cmsis_nn_status;

    /// Max pooling, int16. Input and output must not overlap.
    pub fn arm_max_pool_s16(
        ctx: *const cmsis_nn_context,
        pool_params: *const cmsis_nn_pool_params,
        input_dims: *const cmsis_nn_dims,
        src: *const i16,
        filter_dims: *const cmsis_nn_dims,
        output_dims: *const cmsis_nn_dims,
        dst: *mut i16,
    ) -> arm_cmsis_nn_status;

    // ---------------------------------------------------------------- Softmax

    /// Softmax over the rows of an int8 tensor, int8 output (scale 1/256, zero point -128).
    pub fn arm_softmax_s8(
        input: *const i8,
        num_rows: i32,
        row_size: i32,
        mult: i32,
        shift: i32,
        diff_min: i32,
        output: *mut i8,
    );

    /// Softmax over the rows of an int8 tensor, int16 output.
    pub fn arm_softmax_s8_s16(
        input: *const i8,
        num_rows: i32,
        row_size: i32,
        mult: i32,
        shift: i32,
        diff_min: i32,
        output: *mut i16,
    );

    /// Softmax over the rows of an int16 tensor, with lookup tables.
    pub fn arm_softmax_s16(
        input: *const i16,
        num_rows: i32,
        row_size: i32,
        mult: i32,
        shift: i32,
        softmax_params: *const cmsis_nn_softmax_lut_s16,
        output: *mut i16,
    ) -> arm_cmsis_nn_status;

    // ---------------------------------------------------------------- Data movement

    /// Copies `total_size` bytes (non-overlapping).
    pub fn arm_reshape_s8(input: *const i8, output: *mut i8, total_size: u32);

    /// Transposes an int8 tensor of up to 4 dimensions.
    pub fn arm_transpose_s8(
        input_data: *const i8,
        output_data: *mut i8,
        input_dims: *const cmsis_nn_dims,
        output_dims: *const cmsis_nn_dims,
        transpose_params: *const cmsis_nn_transpose_params,
    ) -> arm_cmsis_nn_status;

    /// Pads an int8 tensor with a constant value.
    pub fn arm_pad_s8(
        input: *const i8,
        output: *mut i8,
        pad_value: i8,
        input_size: *const cmsis_nn_dims,
        pre_pad: *const cmsis_nn_dims,
        post_pad: *const cmsis_nn_dims,
    ) -> arm_cmsis_nn_status;

    /// Copies one input of a concatenation along the X (width) axis into the output.
    pub fn arm_concatenation_s8_x(
        input: *const i8,
        input_x: u16,
        input_y: u16,
        input_z: u16,
        input_w: u16,
        output: *mut i8,
        output_x: u16,
        offset_x: u32,
    );

    /// Copies one input of a concatenation along the Y (height) axis into the output.
    pub fn arm_concatenation_s8_y(
        input: *const i8,
        input_x: u16,
        input_y: u16,
        input_z: u16,
        input_w: u16,
        output: *mut i8,
        output_y: u16,
        offset_y: u32,
    );

    /// Copies one input of a concatenation along the Z (channel) axis into the output.
    pub fn arm_concatenation_s8_z(
        input: *const i8,
        input_x: u16,
        input_y: u16,
        input_z: u16,
        input_w: u16,
        output: *mut i8,
        output_z: u16,
        offset_z: u32,
    );

    /// Copies one input of a concatenation along the W (batch) axis into the output.
    pub fn arm_concatenation_s8_w(
        input: *const i8,
        input_x: u16,
        input_y: u16,
        input_z: u16,
        input_w: u16,
        output: *mut i8,
        offset_w: u32,
    );

    /// Size, alignment or field offset `index` of the CMSIS-NN structs as compiled (see
    /// `src/layout.c`), `u32::MAX` past the end. Used by the layout tests.
    #[doc(hidden)]
    pub fn nanoglia_cmsis_nn_layout(index: u32) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    macro_rules! entry {
        ($t:ty) => {
            [size_of::<$t>(), align_of::<$t>()]
        };
    }

    #[test]
    fn struct_layouts_match_c() {
        let expected: &[usize] = &[
            entry!(cmsis_nn_tile)[0],
            entry!(cmsis_nn_tile)[1],
            entry!(cmsis_nn_context)[0],
            entry!(cmsis_nn_context)[1],
            entry!(cmsis_nn_bias_data)[0],
            entry!(cmsis_nn_bias_data)[1],
            offset_of!(cmsis_nn_bias_data, is_int32_bias),
            entry!(cmsis_nn_dims)[0],
            entry!(cmsis_nn_dims)[1],
            offset_of!(cmsis_nn_dims, c),
            entry!(cmsis_nn_per_channel_quant_params)[0],
            entry!(cmsis_nn_per_channel_quant_params)[1],
            entry!(cmsis_nn_per_tensor_quant_params)[0],
            entry!(cmsis_nn_per_tensor_quant_params)[1],
            entry!(cmsis_nn_quant_params)[0],
            entry!(cmsis_nn_quant_params)[1],
            offset_of!(cmsis_nn_quant_params, is_per_channel),
            entry!(cmsis_nn_activation)[0],
            entry!(cmsis_nn_activation)[1],
            entry!(cmsis_nn_conv_params)[0],
            entry!(cmsis_nn_conv_params)[1],
            offset_of!(cmsis_nn_conv_params, stride),
            offset_of!(cmsis_nn_conv_params, padding),
            offset_of!(cmsis_nn_conv_params, dilation),
            offset_of!(cmsis_nn_conv_params, activation),
            entry!(cmsis_nn_transpose_conv_params)[0],
            entry!(cmsis_nn_transpose_conv_params)[1],
            offset_of!(cmsis_nn_transpose_conv_params, padding_offsets),
            offset_of!(cmsis_nn_transpose_conv_params, activation),
            entry!(cmsis_nn_dw_conv_params)[0],
            entry!(cmsis_nn_dw_conv_params)[1],
            offset_of!(cmsis_nn_dw_conv_params, ch_mult),
            offset_of!(cmsis_nn_dw_conv_params, stride),
            offset_of!(cmsis_nn_dw_conv_params, activation),
            entry!(cmsis_nn_pool_params)[0],
            entry!(cmsis_nn_pool_params)[1],
            offset_of!(cmsis_nn_pool_params, activation),
            entry!(cmsis_nn_fc_params)[0],
            entry!(cmsis_nn_fc_params)[1],
            offset_of!(cmsis_nn_fc_params, output_offset),
            offset_of!(cmsis_nn_fc_params, activation),
            entry!(cmsis_nn_bmm_params)[0],
            entry!(cmsis_nn_bmm_params)[1],
            offset_of!(cmsis_nn_bmm_params, fc_params),
            entry!(cmsis_nn_transpose_params)[0],
            entry!(cmsis_nn_transpose_params)[1],
            offset_of!(cmsis_nn_transpose_params, permutations),
            entry!(cmsis_nn_svdf_params)[0],
            entry!(cmsis_nn_svdf_params)[1],
            offset_of!(cmsis_nn_svdf_params, output_activation),
            entry!(cmsis_nn_softmax_lut_s16)[0],
            entry!(cmsis_nn_softmax_lut_s16)[1],
            offset_of!(cmsis_nn_softmax_lut_s16, one_by_one_lut),
            entry!(cmsis_nn_scaling)[0],
            entry!(cmsis_nn_scaling)[1],
        ];
        for (index, &want) in expected.iter().enumerate() {
            // SAFETY: pure function reading a constant table.
            let got = unsafe { nanoglia_cmsis_nn_layout(index as u32) };
            assert_eq!(got as usize, want, "layout entry {index}");
        }
        // SAFETY: as above.
        let end = unsafe { nanoglia_cmsis_nn_layout(expected.len() as u32) };
        assert_eq!(end, u32::MAX, "the C table has more entries than the test");
    }

    #[test]
    fn buffer_size_and_kernel_link() {
        let input = cmsis_nn_dims {
            n: 1,
            h: 4,
            w: 4,
            c: 3,
        };
        let filter = cmsis_nn_dims {
            n: 2,
            h: 3,
            w: 3,
            c: 3,
        };
        let output = cmsis_nn_dims {
            n: 1,
            h: 2,
            w: 2,
            c: 2,
        };
        let params = cmsis_nn_conv_params {
            stride: cmsis_nn_tile { w: 1, h: 1 },
            dilation: cmsis_nn_tile { w: 1, h: 1 },
            activation: cmsis_nn_activation {
                min: -128,
                max: 127,
            },
            ..Default::default()
        };
        // SAFETY: all pointers are to live locals.
        let size =
            unsafe { arm_convolve_wrapper_s8_get_buffer_size(&params, &input, &filter, &output) };
        assert!(size > 0);
    }
}
