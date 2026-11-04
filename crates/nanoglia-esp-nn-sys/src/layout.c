/*
 * Size, alignment and field offsets of the ESP-NN structs, as the C compiler sees them. The Rust
 * tests compare them with the hand-written `#[repr(C)]` definitions. The table is tiny and
 * dropped by the linker when nothing references it.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stddef.h>
#include <stdint.h>

#include "esp_nn_defs.h"

#define ENTRY(type) sizeof(type), _Alignof(type)
#define FIELD(type, field) offsetof(type, field)

static const uint32_t nanoglia_esp_nn_layout_table[] = {
    ENTRY(data_dims_t),
    FIELD(data_dims_t, height),
    FIELD(data_dims_t, channels),
    FIELD(data_dims_t, extra),
    ENTRY(data_2d_t),
    FIELD(data_2d_t, height),
    ENTRY(act_params_t),
    FIELD(act_params_t, max),
    ENTRY(quant_data_t),
    FIELD(quant_data_t, mult),
    ENTRY(conv_params_t),
    FIELD(conv_params_t, out_offset),
    FIELD(conv_params_t, stride),
    FIELD(conv_params_t, padding),
    FIELD(conv_params_t, dilation),
    FIELD(conv_params_t, activation),
    ENTRY(dw_conv_params_t),
    FIELD(dw_conv_params_t, ch_mult),
    FIELD(dw_conv_params_t, stride),
    FIELD(dw_conv_params_t, padding),
    FIELD(dw_conv_params_t, dilation),
    FIELD(dw_conv_params_t, activation),
};

/* Returns entry `index` of the layout table, or UINT32_MAX past its end. */
uint32_t nanoglia_esp_nn_layout(uint32_t index)
{
    if (index >= sizeof(nanoglia_esp_nn_layout_table) / sizeof(nanoglia_esp_nn_layout_table[0])) {
        return UINT32_MAX;
    }
    return nanoglia_esp_nn_layout_table[index];
}
