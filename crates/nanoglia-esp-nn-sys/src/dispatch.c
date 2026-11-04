/*
 * ESP-NN picks the implementation of each public function with macros of `esp_nn.h`
 * (`esp_nn_conv_s8` is `esp_nn_conv_s8_esp32s3` on ESP32-S3, `esp_nn_conv_s8_ansi` elsewhere).
 * Rust cannot see these macros: the `nanoglia_esp_nn_*` functions below forward to whatever
 * implementation the build selected, so the Rust bindings stay the same on every target.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>

#include "esp_nn.h"

/* Convolution. */

int nanoglia_esp_nn_get_conv_scratch_size(const data_dims_t *input_dims,
                                          const data_dims_t *filter_dims,
                                          const data_dims_t *output_dims,
                                          const conv_params_t *conv_params)
{
    return esp_nn_get_conv_scratch_size(input_dims, filter_dims, output_dims, conv_params);
}

void nanoglia_esp_nn_set_conv_scratch_buf(void *buf)
{
    esp_nn_set_conv_scratch_buf(buf);
}

void nanoglia_esp_nn_conv_s8(const data_dims_t *input_dims, const int8_t *input_data,
                             const data_dims_t *filter_dims, const int8_t *filter_data,
                             const int32_t *bias, const data_dims_t *output_dims,
                             int8_t *out_data, const conv_params_t *conv_params,
                             const quant_data_t *quant_data)
{
    esp_nn_conv_s8(input_dims, input_data, filter_dims, filter_data, bias, output_dims, out_data,
                   conv_params, quant_data);
}

/* Depthwise convolution. */

int nanoglia_esp_nn_get_depthwise_conv_scratch_size(const data_dims_t *input_dims,
                                                    const data_dims_t *filter_dims,
                                                    const data_dims_t *output_dims,
                                                    const dw_conv_params_t *conv_params)
{
    return esp_nn_get_depthwise_conv_scratch_size(input_dims, filter_dims, output_dims,
                                                  conv_params);
}

void nanoglia_esp_nn_set_depthwise_conv_scratch_buf(void *buf)
{
    esp_nn_set_depthwise_conv_scratch_buf(buf);
}

void nanoglia_esp_nn_depthwise_conv_s8(const data_dims_t *input_dims, const int8_t *input_data,
                                       const data_dims_t *filter_dims,
                                       const int8_t *filter_data, const int32_t *bias,
                                       const data_dims_t *output_dims, int8_t *out_data,
                                       const dw_conv_params_t *conv_params,
                                       const quant_data_t *quant_data)
{
    esp_nn_depthwise_conv_s8(input_dims, input_data, filter_dims, filter_data, bias, output_dims,
                             out_data, conv_params, quant_data);
}

/* Fully connected. */

void nanoglia_esp_nn_fully_connected_s8(const int8_t *input_data, int32_t input_offset,
                                        uint16_t row_len, const int8_t *filter_data,
                                        int32_t filter_offset, const int32_t *bias,
                                        int8_t *out_data, uint16_t out_channels,
                                        int32_t out_offset, int32_t out_shift, int32_t out_mult,
                                        int32_t activation_min, int32_t activation_max)
{
    esp_nn_fully_connected_s8(input_data, input_offset, row_len, filter_data, filter_offset,
                              bias, out_data, out_channels, out_offset, out_shift, out_mult,
                              activation_min, activation_max);
}

void nanoglia_esp_nn_fully_connected_per_ch_s8_batch(
    const int8_t *input_data, int32_t input_offset, uint16_t row_len, const int8_t *filter_data,
    int32_t filter_offset, const int32_t *bias, int8_t *out_data, uint16_t out_channels,
    int32_t out_offset, const int32_t *out_shift, const int32_t *out_mult,
    int32_t activation_min, int32_t activation_max, int32_t batches)
{
    esp_nn_fully_connected_per_ch_s8_batch(input_data, input_offset, row_len, filter_data,
                                           filter_offset, bias, out_data, out_channels,
                                           out_offset, out_shift, out_mult, activation_min,
                                           activation_max, batches);
}

/* Pooling. */

void nanoglia_esp_nn_avg_pool_s8(const int8_t *input, uint16_t input_wd, uint16_t input_ht,
                                 int8_t *output, uint16_t output_wd, uint16_t output_ht,
                                 uint16_t stride_wd, uint16_t stride_ht, uint16_t filter_wd,
                                 uint16_t filter_ht, uint16_t pad_wd, uint16_t pad_ht,
                                 int32_t activation_min, int32_t activation_max,
                                 uint16_t channels)
{
    esp_nn_avg_pool_s8(input, input_wd, input_ht, output, output_wd, output_ht, stride_wd,
                       stride_ht, filter_wd, filter_ht, pad_wd, pad_ht, activation_min,
                       activation_max, channels);
}

void nanoglia_esp_nn_max_pool_s8(const int8_t *input, uint16_t input_wd, uint16_t input_ht,
                                 int8_t *output, uint16_t output_wd, uint16_t output_ht,
                                 uint16_t stride_wd, uint16_t stride_ht, uint16_t filter_wd,
                                 uint16_t filter_ht, uint16_t pad_wd, uint16_t pad_ht,
                                 int32_t activation_min, int32_t activation_max,
                                 uint16_t channels)
{
    esp_nn_max_pool_s8(input, input_wd, input_ht, output, output_wd, output_ht, stride_wd,
                       stride_ht, filter_wd, filter_ht, pad_wd, pad_ht, activation_min,
                       activation_max, channels);
}

/* Softmax. */

int32_t nanoglia_esp_nn_get_softmax_scratch_size(int32_t width, int32_t height)
{
    return esp_nn_get_softmax_scratch_size(width, height);
}

void nanoglia_esp_nn_set_softmax_scratch_buf(void *buffer)
{
    esp_nn_set_softmax_scratch_buf(buffer);
}

void nanoglia_esp_nn_softmax_s8(const int8_t *input_data, int32_t height, int32_t width,
                                int32_t mult, int32_t shift, int32_t diff_min,
                                int8_t *output_data)
{
    esp_nn_softmax_s8(input_data, height, width, mult, shift, diff_min, output_data);
}

/* Elementwise arithmetic. */

void nanoglia_esp_nn_add_elementwise_s8(const int8_t *input1_data, const int8_t *input2_data,
                                        int32_t input1_offset, int32_t input2_offset,
                                        int32_t input1_mult, int32_t input2_mult,
                                        int32_t input1_shift, int32_t input2_shift,
                                        int32_t left_shift, int8_t *output, int32_t out_offset,
                                        int32_t out_mult, int32_t out_shift,
                                        int32_t activation_min, int32_t activation_max,
                                        int32_t size)
{
    esp_nn_add_elementwise_s8(input1_data, input2_data, input1_offset, input2_offset, input1_mult,
                              input2_mult, input1_shift, input2_shift, left_shift, output,
                              out_offset, out_mult, out_shift, activation_min, activation_max,
                              size);
}

void nanoglia_esp_nn_mul_elementwise_s8(const int8_t *input1_data, const int8_t *input2_data,
                                        int32_t input1_offset, int32_t input2_offset,
                                        int8_t *output, int32_t out_offset, int32_t out_mult,
                                        int32_t out_shift, int32_t activation_min,
                                        int32_t activation_max, int32_t size)
{
    esp_nn_mul_elementwise_s8(input1_data, input2_data, input1_offset, input2_offset, output,
                              out_offset, out_mult, out_shift, activation_min, activation_max,
                              size);
}
