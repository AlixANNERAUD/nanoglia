/*
 * Size, alignment and field offsets of the CMSIS-NN structs, as the C compiler sees them.
 * The Rust tests compare them with the hand-written `#[repr(C)]` definitions. The table is
 * tiny and dropped by the linker when nothing references it.
 *
 * SPDX-License-Identifier: MIT
 */

#include <stddef.h>
#include <stdint.h>

#include "arm_nn_types.h"

#define ENTRY(type) sizeof(type), _Alignof(type)
#define FIELD(type, field) offsetof(type, field)

static const uint32_t nanonn_cmsis_nn_layout_table[] = {
    ENTRY(cmsis_nn_tile),
    ENTRY(cmsis_nn_context),
    ENTRY(cmsis_nn_bias_data),
    FIELD(cmsis_nn_bias_data, is_int32_bias),
    ENTRY(cmsis_nn_dims),
    FIELD(cmsis_nn_dims, c),
    ENTRY(cmsis_nn_per_channel_quant_params),
    ENTRY(cmsis_nn_per_tensor_quant_params),
    ENTRY(cmsis_nn_quant_params),
    FIELD(cmsis_nn_quant_params, is_per_channel),
    ENTRY(cmsis_nn_activation),
    ENTRY(cmsis_nn_conv_params),
    FIELD(cmsis_nn_conv_params, stride),
    FIELD(cmsis_nn_conv_params, padding),
    FIELD(cmsis_nn_conv_params, dilation),
    FIELD(cmsis_nn_conv_params, activation),
    ENTRY(cmsis_nn_transpose_conv_params),
    FIELD(cmsis_nn_transpose_conv_params, padding_offsets),
    FIELD(cmsis_nn_transpose_conv_params, activation),
    ENTRY(cmsis_nn_dw_conv_params),
    FIELD(cmsis_nn_dw_conv_params, ch_mult),
    FIELD(cmsis_nn_dw_conv_params, stride),
    FIELD(cmsis_nn_dw_conv_params, activation),
    ENTRY(cmsis_nn_pool_params),
    FIELD(cmsis_nn_pool_params, activation),
    ENTRY(cmsis_nn_fc_params),
    FIELD(cmsis_nn_fc_params, output_offset),
    FIELD(cmsis_nn_fc_params, activation),
    ENTRY(cmsis_nn_bmm_params),
    FIELD(cmsis_nn_bmm_params, fc_params),
    ENTRY(cmsis_nn_transpose_params),
    FIELD(cmsis_nn_transpose_params, permutations),
    ENTRY(cmsis_nn_svdf_params),
    FIELD(cmsis_nn_svdf_params, output_activation),
    ENTRY(cmsis_nn_softmax_lut_s16),
    FIELD(cmsis_nn_softmax_lut_s16, one_by_one_lut),
    ENTRY(cmsis_nn_scaling),
};

/* Returns entry `index` of the layout table, or UINT32_MAX past its end. */
uint32_t nanonn_cmsis_nn_layout(uint32_t index)
{
    if (index >= sizeof(nanonn_cmsis_nn_layout_table) / sizeof(nanonn_cmsis_nn_layout_table[0]))
    {
        return UINT32_MAX;
    }
    return nanonn_cmsis_nn_layout_table[index];
}
