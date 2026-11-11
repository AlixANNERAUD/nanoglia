# Supported operators

Every TFLite builtin operator nanoglia runs, for which tensor types, on
which backend, and under which conditions. The conditions come from the
`select` and `plan` functions of the backends
(`crates/nanoglia-esp-nn/src/*.rs`, `crates/nanoglia-cmsis-nn/src/*.rs`
and `crates/nanoglia-kernels/src/*.rs`): an operator that does not meet them
is not claimed, and loading fails with `Detail::UnsupportedOp`
([Errors](errors.md)).

To know what a given model gets, operator by operator, run `nanoglia
check` on it
([Check whether a model runs](../how-to-guides/check-whether-a-model-runs.md)).

## How a kernel is chosen

`Graph::load_in` asks the backends of the registry in order, operator by
operator; the first backend whose `select` returns a kernel wins. The
default registry (`Registry::default()`) holds, in this order:

1. `esp-nn` (feature `esp-nn`, off by default): ESP-NN v1.4.2, with its
   optimized kernels on the ESP32-S3 and its ANSI C kernels elsewhere
   (see [ESP-NN](#esp-nn) below).
2. `cmsis-nn` (feature `cmsis-nn`): CMSIS-NN v8.0.0, with the DSP or
   Helium code paths on Arm Cortex-M and its portable C code on every
   other target.
3. `rust` (feature `kernels`): pure Rust kernels.

Inside the `rust` backend, a few operators have several kernels, tried
in the order of the tables below. `Graph::kernel_name(index)` returns the
name of the kernel chosen for an operator (the names are in the tables).

Which kernels exist also depends on the Cargo features
([Cargo features](features.md)):

- every operator has a feature of its own, its name in lower case with
  dashes after `op-`: `op-conv-2d` for `CONV_2D`, `op-padv2` for
  `PADV2` (`BuiltinOp::feature()`; the list is in
  [Operator features](features.md#operator-features)). They are all on
  by default (`all-ops`). Without the feature of an operator, none of
  the three backends claims it, whatever its types;
- kernels marked *float* also need feature `float`, kernels marked
  *int16* feature `int16` (the Feature column of the tables below).

## Conditions shared by all kernels

- One subgraph, no variable tensors, at most 16 inputs and 8 outputs per
  operator, static shapes of rank 6 at most. Element counts must fit an
  `i32`, as in TFLite.
- Quantized activations are quantized per tensor, with a finite, positive
  scale; the CMSIS-NN kernels and `rust/binary_q` also need the zero point
  inside the range of the type. int16 activations are symmetric: zero
  point 0 (the 16x8 scheme of TFLite); only `QUANTIZE` (except from int16
  to int16) accepts other int16 zero points.
- Quantized weights are symmetric (zero point 0), quantized per tensor or
  per output channel, with finite, non-negative scales.
- Fused activations of the quantized and float kernels: `NONE`, `RELU`,
  `RELU6`, `RELU_N1_TO_1`. Other fused activations (`TANH`, `SIGN_BIT`)
  are refused.
- Constant buffers must be aligned for their element type in memory
  (`Detail::Misaligned` otherwise); a model placed on a 16-byte boundary
  meets this, see [Memory](memory.md#model-bytes).

## Matrix

"int8" means int8 activations and, for operators with weights, int8
weights and int32 bias. "16x8" means int16 activations, int8 weights and
int64 bias.

| Operator | Types | Backend: kernel | Feature |
| --- | --- | --- | --- |
| `CONV_2D` | int8 | `cmsis-nn/conv2d_s8` | — |
| | 16x8, constant bias, bounded accumulators | `cmsis-nn/conv2d_s16` | int16 |
| | f32 | `rust/conv_2d_f32` | float |
| `TRANSPOSE_CONV` | int8, configurations CMSIS-NN computes exactly | `cmsis-nn/transpose_conv_s8` | — |
| | int8, otherwise | `rust/transpose_conv_s8` | — |
| | f32 | `rust/transpose_conv_f32` | float |
| `DEPTHWISE_CONV_2D` | int8 | `cmsis-nn/depthwise_conv2d_s8` | — |
| | 16x8, constant bias, bounded accumulators | `cmsis-nn/depthwise_conv2d_s16` | int16 |
| | f32 | `rust/depthwise_conv_2d_f32` | float |
| `FULLY_CONNECTED` | int8 | `cmsis-nn/fully_connected_s8` | — |
| | 16x8, per-tensor weights, small shift, constant bias, bounded accumulators | `cmsis-nn/fully_connected_s16` | int16 |
| | 16x8, otherwise | `rust/fully_connected_16x8` | int16 |
| | f32 | `rust/fully_connected_f32` | float |
| `AVERAGE_POOL_2D` | int8 | `cmsis-nn/average_pool2d_s8` | — |
| | int16 | `cmsis-nn/average_pool2d_s16` | int16 |
| | f32 | `rust/pool_f32` | float |
| `MAX_POOL_2D` | int8 | `cmsis-nn/max_pool2d_s8` | — |
| | int16 | `cmsis-nn/max_pool2d_s16` | int16 |
| | f32 | `rust/pool_f32` | float |
| `SOFTMAX` | int8 | `cmsis-nn/softmax_s8` | — |
| | int16 | `rust/softmax_i16` | int16 |
| | f32 | `rust/softmax_f32` | float |
| `ADD`, `MUL` | int8, same shapes | `cmsis-nn/add_s8`, `cmsis-nn/mul_s8` | — |
| | int16, same shapes | `cmsis-nn/add_s16`, `cmsis-nn/mul_s16` | int16 |
| | int8, int16, broadcasting | `rust/binary_q` | — |
| | f32 | `rust/binary` | — |
| `SUB` | int8, int16 | `rust/binary_q` | — |
| | f32 | `rust/binary` | — |
| `MAXIMUM`, `MINIMUM` | int8, same shapes or one single value, rank ≤ 4 | `cmsis-nn/maximum_s8`, `cmsis-nn/minimum_s8` | — |
| | f32, int8, uint8, int16, int32, int64 | `rust/binary` | — |
| `PAD`, `PADV2` | int8, rank 1 to 4 | `cmsis-nn/pad_s8` | — |
| | any whole-byte type | `rust/pad` | — |
| `RESHAPE`, `SQUEEZE`, `EXPAND_DIMS` | any type | `rust/reshape` | — |
| `CONCATENATION` | any whole-byte type | `rust/concatenation` | — |
| `TRANSPOSE` | any type of 1, 2, 4 or 8 bytes | `rust/transpose` | — |
| `SLICE` | any type of 1, 2, 4 or 8 bytes | `rust/slice` | — |
| `STRIDED_SLICE` | any type of 1, 2, 4 or 8 bytes | `rust/strided_slice` | — |
| `GATHER` | f32, f16, bf16, int8, uint8, int16, int32, int64, bool | `rust/gather` | — |
| `SPLIT`, `SPLIT_V` | f32, int8, uint8, int16, int32, int64 | `rust/split` | — |
| `ARG_MAX`, `ARG_MIN` | f32, int8, uint8, int16, int32, bool → int32, int64 | `rust/arg_min_max` | — |
| `RESIZE_NEAREST_NEIGHBOR` | f32, int8, uint8, int16, int32, bool | `rust/resize_nearest_neighbor` | — |
| `RESIZE_BILINEAR` | f32 | `rust/resize_bilinear_f32` | float |
| | int8, uint8, int16 | `rust/resize_bilinear_q` | — |
| `QUANTIZE` | f32 → int8, uint8, int16 | `rust/quantize` | — |
| | int8, uint8 → int8, uint8, int16; int16 → int8, int16; int32 → int8, int16 | `rust/requantize` | — |
| `DEQUANTIZE` | int8, uint8, int16, f16 → f32 | `rust/dequantize` | — |
| `MEAN` | f32, int8, uint8, int16 | `rust/mean` | — |
| `SUM` | f32, int8, uint8, int16, int32, int64 | `rust/sum` | — |
| `REDUCE_MAX`, `REDUCE_MIN` | f32, int8, uint8, int16, int32, int64 | `rust/reduce_min_max` | — |
| `PACK` | f32, int8, uint8, int16, int32, uint32, int64 | `rust/pack` | — |
| `UNPACK` | f32, f16, bf16, int8, uint8, int16, int32, bool | `rust/unpack` | — |
| `SPACE_TO_DEPTH` | f32, int8, uint8, int32, int64 | `rust/space_to_depth` | — |
| `DEPTH_TO_SPACE` | f32, int8, uint8, int32, int64 | `rust/depth_to_space` | — |
| `SPACE_TO_BATCH_ND` | f32, int8, uint8, int16, int32, int64 | `rust/space_to_batch_nd` | — |
| `BATCH_TO_SPACE_ND` | f32, int8, uint8, int16, int32, int64 | `rust/batch_to_space_nd` | — |
| `PRELU` | f32 | `rust/prelu_f32` | float |
| | int8, uint8, int16 (int8 `alpha`) | `rust/prelu_q` | — |
| `ABS`, `NEG`, `SQRT`, `RSQRT`, `EXP`, `LOG` | f32 | `rust/unary_f32` | float |
| `ABS` | int8, int16 | `rust/unary_q` | — |
| | int32, int16 without quantization | `rust/unary_int` | — |
| `NEG` | int32, int64 | `rust/unary_int` | — |
| `RSQRT` | int8 | `rust/unary_q` | — |
| `DIV` | f32, int32 | `rust/binary` | — |
| `SQUARED_DIFFERENCE` | f32, int32 | `rust/binary` | — |
| | int8, int16 | `rust/binary_q` | — |
| `EQUAL`, `NOT_EQUAL` | f32, int32, int64, bool; int8, uint8 (rescaled) → bool | `rust/comparison` | — |
| `LESS`, `LESS_EQUAL`, `GREATER`, `GREATER_EQUAL` | f32, int32, int64; int8, uint8 (rescaled) → bool | `rust/comparison` | — |
| `SELECT_V2` | f32, bool, int8, uint8, int16, int32, uint32, int64 | `rust/select_v2` | — |
| `CAST` | between bool, int8, uint8, int16, int32, uint32, int64, f32 | `rust/cast` | — |
| `RELU`, `RELU6`, `RELU_N1_TO_1` | f32 | `rust/relu_f32` | float |
| | int8, uint8, int16 | `rust/relu_q` | — |
| `LEAKY_RELU` | f32 | `rust/leaky_relu_f32` | float |
| | int8, uint8, int16 | `rust/leaky_relu_q` | — |
| `LOGISTIC` | f32 | `rust/logistic_f32` | float |
| | int8, uint8 | `rust/lut_q` | — |
| | int16 | `rust/logistic_i16` | int16 |
| `TANH` | f32 | `rust/tanh_f32` | float |
| | int8, uint8 | `rust/lut_q` | — |
| | int16 | `rust/tanh_i16` | int16 |
| `HARD_SWISH` | f32 | `rust/hard_swish_f32` | float |
| | int8, uint8 | `rust/hard_swish_q` | — |

Any other operator, any custom operator, and any type not listed (for
example int16 `TRANSPOSE_CONV` or `HARD_SWISH`, uint8 `ADD`, int4 weights)
is unsupported.

## ESP-NN

With feature `esp-nn`, the `esp-nn` backend is asked first. It claims an
int8 operator only when ESP-NN computes it exactly like TFLite **and**
the CMSIS-NN kernel of the matrix above claims it too: while another
context (thread, task, interrupt handler or core) is running ESP-NN,
whose scratch buffers are global variables, the operator runs that
CMSIS-NN kernel instead, with the same result. Otherwise the operator
goes to CMSIS-NN or the Rust kernels as in the matrix.

| Operator | Kernel | Conditions beyond the CMSIS-NN kernel's |
| --- | --- | --- |
| `CONV_2D` (int8) | `esp-nn/conv2d_s8` | no dilation; strides and paddings up to 32767 |
| `DEPTHWISE_CONV_2D` (int8) | `esp-nn/depthwise_conv2d_s8` | no dilation; strides and paddings up to 32767 |
| `FULLY_CONNECTED` (int8) | `esp-nn/fully_connected_s8` | input and output depths up to 65535 |
| `AVERAGE_POOL_2D`, `MAX_POOL_2D` (int8) | `esp-nn/average_pool2d_s8`, `esp-nn/max_pool2d_s8` | strides, filter sizes and paddings up to 32767 |
| `SOFTMAX` (int8) | `esp-nn/softmax_s8` | input scaling left shift up to 30 (31 for CMSIS-NN) |
| `ADD`, `MUL` (int8, same shapes) | `esp-nn/add_s8`, `esp-nn/mul_s8` | — |

The backend claims nothing when the `single-rounding` feature is on
(ESP-NN only rounds twice), nor on targets without atomic
compare-and-swap, such as Cortex-M0 and `riscv32imc` (its lock needs
them). On targets other than the ESP32-S3, ESP-NN runs its ANSI C
kernels, which are slower than CMSIS-NN's portable C: enable `esp-nn`
there only to test the backend.

## Restrictions per operator

### CONV_2D

All kernels: input `[batches, height, width, channels]` and output of
rank 4, filter `[output channels, height, width, input channels / groups]`,
bias (optional) of one value per output channel, any stride, dilation
and `SAME`/`VALID` padding, and the output size TFLite computes from
them.

- `cmsis-nn/conv2d_s8`: weights per tensor or per output channel (axis 0);
  grouped convolutions (input channels a multiple of the filter's, output
  channels a multiple of the group count). Every dimension in
  `1..=65535`, strides and dilations in `1..=32767` (CMSIS-NN keeps them
  in 16-bit variables).
- `cmsis-nn/conv2d_s16`: as above, but no groups, input and output zero
  points 0, an int64 bias (int32 biases are refused: LiteRT reads the
  bias as int64 by default and nanoglia does not decode the
  `quantized_bias_type` option), at most 511 filter values per output
  channel (`height × width × input channels`; CMSIS-NN accumulates some
  outputs in 32 bits before adding the bias), a requantization shift
  below 8 for every channel, and a constant bias (or none) with bounded
  accumulators, as for [`depthwise_conv2d_s16`](#depthwise_conv_2d).
- `rust/conv_2d_f32`: grouped convolutions included.

int16 convolutions outside these conditions are unsupported: there is no
Rust kernel for them, so the model does not load.

### TRANSPOSE_CONV

All kernels: inputs `output_shape`, weights, input and an optional bias,
as in TFLite. `output_shape` is a constant int32 vector of four values
equal to the output tensor's shape; input `[batches, height, width,
channels]`, weights `[output channels, height, width, input channels]`,
bias of one value per output channel, every dimension positive. The input
must have the size the forward convolution computes from the output
(`SAME` or `VALID`, strides in `1..=32767`), whose padding must fit an
int16. The fused activation comes from the options (operator version 4;
absent before). int16 (16x8) transpose convolutions and hybrid ones (f32
input, int8 weights) are refused.

- `cmsis-nn/transpose_conv_s8`: weights per tensor or per output channel
  (axis 0), int32 bias with zero point 0, every dimension in `1..=65535`.
  The fused activation is applied, as in LiteRT's reference kernel
  (TensorFlow Lite Micro's CMSIS-NN kernel ignores it). CMSIS-NN v8.0.0
  convolves the upsampled input with the reversed filter for strides up
  to 2 and more than 16 input channels: every output size, with the
  upsampled height and width up to 65535. Otherwise it runs
  `arm_transpose_conv_s8`, claimed only when `in_h × stride_h - pad_h` is
  between 0 and the output height, `in_h × stride_h + max(filter_h -
  stride_h, 0) ≥ pad_h + out_h`, and `pad_w + out_w ≤ (in_w - 1) ×
  stride_w + max(filter_w, stride_w)`: outside these bounds it writes
  past the output, leaves output rows unwritten or reads past its buffer
  (for example a `SAME` 2×2 stride 2 transpose to an odd size, or the
  largest `VALID` output of a filter larger than the stride).
- `rust/transpose_conv_s8`: what `cmsis-nn/transpose_conv_s8` refuses
  (or every int8 transpose convolution without the `cmsis-nn` feature),
  same weights and bias; one `i32` of scratch per output value.
- `rust/transpose_conv_f32`: f32 weights and bias.

### DEPTHWISE_CONV_2D

- `cmsis-nn/depthwise_conv2d_s8`: filter `[1, height, width, output
  channels]`, output channels a multiple of input channels (the depth
  multiplier is derived from the shapes, as LiteRT does), weights per
  tensor or per channel along axis 3, int32 bias. Same dimension, stride
  and dilation limits as `conv2d_s8`, plus `(output height - 1) ×
  stride_h - padding_h` and the same for the width at most 32767, with
  paddings up to 32767: CMSIS-NN keeps the window origin in an `int16_t`.
- `cmsis-nn/depthwise_conv2d_s16`: as `depthwise_conv2d_s8`, with input
  and output zero points 0 and an int64 bias (int32 biases are refused, as
  for `conv2d_s16`), plus two conditions under which CMSIS-NN computes
  LiteRT's reference result exactly:
  - a requantization shift below 8 for every channel;
  - a constant bias (or none), small enough that the largest possible
    accumulator of each channel, `2^15 × Σ|weight| + |bias|` over the
    channel's constant weights (each weight computed at run time counts
    as 128), multiplied by the 16-bit multiplier CMSIS-NN derives from the
    channel's multiplier, stays below
    `2^min(45 - shift, 62) - 2^(14 - shift)`, with the accumulator
    itself below `2^62`.
    CMSIS-NN v8.0.0 keeps `product >> (14 - shift)` in an `int32_t` and
    adds 1 to it, where TFLite keeps 64 bits
    ([#38](https://github.com/AlixANNERAUD/nanoglia/issues/38)); beyond
    this bound, saturated outputs can take the wrong sign. For example, a
    channel whose shift is -9 (effective scale between 2^-10 and 2^-9)
    accepts accumulators up to about 2^39. The same bound applies to
    `conv2d_s16` and `fully_connected_s16`.

  Depth multiplier 1, no dilation and fewer than 512 filter values per
  channel run `arm_depthwise_conv_fast_s16` (its DSP or Helium code on
  Cortex-M, its generic code elsewhere), the other configurations
  `arm_depthwise_conv_s16`.
- `rust/depthwise_conv_2d_f32`: same shapes.

int16 depthwise convolutions outside these conditions are unsupported:
there is no Rust kernel for them, nor for int8.

### FULLY_CONNECTED

All kernels: the default weights format (shuffled weights are refused),
weights of rank 2 `[units, depth]`, an output whose last dimension is
`units`, an input of `batches × depth` elements (it is flattened),
`keep_num_dims` either way, optional bias. A non-standard quantization
contract (`FullyConnectedOptions.quant_spec`) is refused by the reader.

- `cmsis-nn/fully_connected_s8`: weights per tensor or per channel
  (axis 0), int32 bias. Outputs of rank 3 or more whose input depth is a
  multiple of 4 run as a 1×1 convolution, the others through CMSIS-NN's
  fully-connected kernel.
- `cmsis-nn/fully_connected_s16`: per-tensor weights only (as TensorFlow
  Lite Micro), int64 bias, input and output zero points 0, requantization
  shift below 8, and a constant bias (or none) with bounded accumulators
  (`2^15 × Σ|weight| + |bias|` for each output), as for
  [`depthwise_conv2d_s16`](#depthwise_conv_2d). With a depth of 512 or
  more, the weights must be constant and no row may start with 512
  weights of -128: CMSIS-NN's DSP and Helium code sums the first 512
  products of a row in an `int32_t`, which those rows overflow.
- `rust/fully_connected_16x8`: what `fully_connected_s16` refuses: weights
  per tensor or per channel (axis 0), int64 bias, zero points 0, shift up
  to 14.
- `rust/fully_connected_f32`: f32 weights and bias.

### AVERAGE_POOL_2D, MAX_POOL_2D

- `cmsis-nn/*_pool2d_s8`: rank-4 input and output, input and output
  sharing their quantization (scales within 1e-6, equal zero points),
  filter sizes and strides in `1..=32767`.
- `cmsis-nn/*_pool2d_s16`: the same with int16 tensors and zero points 0.
  Average pooling rounds half away from zero (`(sum ± count / 2) /
  count`), like LiteRT's reference int16 kernel, and is claimed for at
  most 65535 values per window (`filter_h × filter_w`), so that the 32-bit
  sum and its rounding cannot overflow.
- `rust/pool_f32`: rank-4 f32.

### SOFTMAX

Over the last dimension, input and output of the same shape.

- `cmsis-nn/softmax_s8`: output zero point -128 and scale 1/256 (within
  0.1 %), finite `beta`.
- `rust/softmax_i16`: input zero point 0, output zero point 0 and scale
  1/32768 (within 0.1 %). CMSIS-NN's int16 softmax is not used: it does
  not clamp its result and can overflow.
- `rust/softmax_f32`.

### ADD, SUB, MUL

Two inputs of the output's type, optional fused activation.

- `cmsis-nn/add_s8`, `mul_s8`, `add_s16`, `mul_s16`: both inputs and the
  output of exactly the same shape; no broadcasting. `SUB` has no
  CMSIS-NN kernel.
- `rust/binary_q`: broadcasting up to rank 6, int8 zero points in
  `-128..=127`, int16 zero points 0.
- `rust/binary`: f32, broadcasting up to rank 6.
- int16 `ADD` and `SUB` are refused by both backends when all three
  scales are powers of two: TFLite then runs another, power-of-two kernel
  selected by the `pot_scale_int16` option, which nanoglia does not
  decode.

### MAXIMUM, MINIMUM

Values are compared without requantization, as in LiteRT and TensorFlow
Lite Micro. Inputs and output of the same type.

- `cmsis-nn/maximum_s8`, `minimum_s8`: rank 4 at most, both inputs of the
  output's shape, or one of them a single value.
- `rust/binary`: broadcasting up to rank 6.

### PAD, PADV2

Paddings are a constant `[rank, 2]` int32 or int64 tensor of non-negative
values; the output shape must match. Without a constant value input,
quantized tensors are padded with the output zero point, others with
zero bits.

- `cmsis-nn/pad_s8`: rank 1 to 4; the `PADV2` value is a constant int8
  single value whose zero point is the output's.
- `rust/pad`: rank up to 6; the `PADV2` value has the input's type, one
  element, the output's scale and zero point when quantized, and may be
  computed at run time.

### RESHAPE, SQUEEZE, EXPAND_DIMS

The output shape is read from the output tensor (it must be static); the
input and output have the same type and element count. The bytes are
copied unchanged.

### CONCATENATION

Up to 16 inputs of the output's type, matching on every axis but the
concatenation axis, no fused activation. Quantized inputs must have the
output's scale and zero point (the reference kernel refuses the other
case too).

### TRANSPOSE, SLICE, STRIDED_SLICE, GATHER, SPLIT, SPLIT_V

These operators only move bytes. Neither LiteRT nor TensorFlow Lite Micro
checks the quantization of their tensors, and nanoglia does not either:
quantized values are copied unchanged (converters give the input and the
outputs the same quantization). Ranks go up to 6, and every output tensor
must have the shape TFLite computes from the inputs and options (TFLite
resizes the output; nanoglia plans its memory from the model, so a
different declared shape is refused).

- `rust/transpose`: a constant int32 permutation with one value per input
  dimension, in `-rank..rank` (negative values wrap). Each axis must appear
  once: TFLite does not check it and reads outside the input otherwise.
  Rank 1 or more.
- `rust/slice`: constant `begin` and `size` vectors, both int32 or both
  int64, one value per dimension; `begin` in `0..=dim`, `size` -1 (to the
  end of the axis) or at most `dim - begin`.
- `rust/strided_slice`: constant int32 `begin`, `end` and `strides` of the
  same length (at most 31), with the `begin_mask`, `end_mask`,
  `ellipsis_mask`, `new_axis_mask`, `shrink_axis_mask` and `offset`
  options, negative indices and strides. Refused: a zero or `INT32_MIN`
  stride, several ellipsis bits, a shrunk axis that selects no element
  (TFLite would leave the output unwritten), and an `offset` slice whose
  output shape TFLite computes differently from what its loop writes (sizes
  overrunning the input, or an ellipsis, new axes or fewer indices than
  dimensions, where TFLite reads `end` at another index).
- `rust/gather`: the `axis` and `batch_dims` options (negative values
  wrap; TFLite's range checks), int16, int32 or int64 indices, constant or
  computed at run time. An index that is negative or not below the size of
  the gathered axis makes `compute` fail with `InvalidArgument` /
  `IndexOutOfRange` (`index` is its position in the indices tensor), as
  TFLite fails; nothing is copied then.
- `rust/split`: a constant int32 axis tensor of one element (negative
  values wrap), `num_splits` equal to the number of outputs, at most 8.
  `SPLIT` divides the axis evenly; `SPLIT_V` takes constant int32 or
  int64 sizes, at most one of them -1 (the rest of the axis), the others
  non-negative and summing to the axis.

### ARG_MAX, ARG_MIN

- `rust/arg_min_max`: a constant int32 or int64 axis tensor of one element
  (negative values wrap) over a non-empty axis; the output has the input
  dimensions without the axis and the type of the `output_type` option,
  int32 or int64 (the schema default, float32, is refused like in
  TFLite). Ties give the first index; f32 NaN values are skipped (an
  all-NaN slice gives 0), as in the reference comparison. Quantized
  inputs are compared as raw integers, as in TFLite. TFLite has no int16
  version; nanoglia compares int16 values the same way.

### RESIZE_NEAREST_NEIGHBOR, RESIZE_BILINEAR

Input and output of rank 4 and of the same type; the size is a constant
int32 vector of two values, the output height and width, in
`0..=1048576` like the input's (so that the fixed-point coordinates fit
an `i32`). The `align_corners` and `half_pixel_centers` options are
supported. Like LiteRT and TensorFlow Lite Micro, these operators do not
look at quantization: values are resized as raw integers, so the input and
the output share their quantization in converted models. An empty input
with a non-empty output is refused (TFLite would read before the input).

- `rust/resize_nearest_neighbor`: any combination of the options; copies
  whole pixels.
- `rust/resize_bilinear_f32`, `rust/resize_bilinear_q`: a size of at least
  1, and not both options (TFLite refuses them together). f32 and uint8
  are interpolated in `float` (uint8 adds 0.5 and truncates), int8 and
  int16 in 1/1024 fixed point (`ResizeBilinearInteger`), as LiteRT's
  reference registration runs them. The float path reproduces the fused
  multiply-adds Clang compiles LiteRT with (on AArch64, where the golden
  data comes from): without them, uint8 results differ by one on some
  values. TensorFlow Lite Micro compiled by another compiler may fuse
  differently.

### QUANTIZE, DEQUANTIZE

- `rust/quantize`: per-tensor output quantization.
- `rust/requantize`: per-tensor quantization on both sides; int16 to
  int16 needs zero points 0.
- `rust/dequantize`: int8 and uint8 per tensor or per channel, int16 per
  tensor with zero point 0, f16.

### MEAN, SUM, REDUCE_MAX, REDUCE_MIN

The axes are a constant int32 or int64 tensor (negative axes wrap,
duplicates are ignored); input and output of the same type, a non-empty
input, an output with the element count of the kept dimensions
(`keep_dims` either way); int16 needs zero points 0.

- `rust/mean`, `rust/sum`: quantized inputs are summed in an `i32` minus
  their zero point, then requantized in fixed point with
  `QuantizeMultiplier(input_scale / output_scale)` (computed in `f64`;
  `MEAN` folds the division by the element count into it), as TensorFlow
  Lite Micro does. Refused when that multiplier shifts left by more than
  30 bits. f32, int32 and int64 sums add the inputs in order; integer sums
  wrap around. TensorFlow Lite Micro has `SUM` on f32, int8 and int16
  only; the uint8, int32 and int64 versions follow LiteRT's reference
  kernel.
- `rust/reduce_min_max`: quantized inputs and output must share their
  scale and zero point (values are compared as integers, as TensorFlow
  Lite Micro requires). Each output starts from the lowest (highest)
  finite value of the type and takes the next input only if it compares
  greater (smaller): f32 NaN inputs are skipped, and a slice of `-inf`
  (`+inf`) gives the lowest (highest) finite f32, as in TensorFlow Lite
  Micro (LiteRT's reference kernel starts a reduction over every axis from
  the first input instead). TensorFlow Lite Micro has f32 and int8 only;
  the other types follow LiteRT's reference kernel.

### PACK, UNPACK

`PACK` stacks `values_count` inputs of the same shape along a new axis
of the output (`axis` in `-rank - 1..=rank` of the inputs, negative values
wrap), at most 16 inputs; `UNPACK` splits the `num` slices of an input
axis into as many outputs, which drop that axis, at most 8 outputs; the
input must not be empty. Every tensor has the same type, scale and zero
point (LiteRT refuses the other cases), and the declared output shapes
must be the ones TFLite computes. TensorFlow Lite Micro has f32, int8,
int16, int32 and int64 `PACK`, f32, int8, int16 and int32 `UNPACK`; the
other types follow LiteRT's reference kernels.

### SPACE_TO_DEPTH, DEPTH_TO_SPACE, SPACE_TO_BATCH_ND, BATCH_TO_SPACE_ND

These operators only move bytes; the declared output shape must be the
one TFLite computes. TensorFlow Lite Micro has f32 and int8 only; the
other types follow LiteRT's reference kernels.

- `rust/space_to_depth`, `rust/depth_to_space`: rank 4, a `block_size`
  of at least 1 that divides the input height and width
  (`SPACE_TO_DEPTH`), or whose square divides the input depth
  (`DEPTH_TO_SPACE`). As in TFLite, quantization is not checked: values
  are copied unchanged.
- `rust/space_to_batch_nd`, `rust/batch_to_space_nd`: rank 3 or 4 (a
  rank 3 tensor has one spatial dimension), a constant int32 block shape
  of one value per spatial dimension, each at least 1, and constant int32
  paddings or crops of shape `[spatial dimensions, 2]`, non-negative.
  Padded (uncropped) sizes must fit an `i32`, and the crops must leave a
  non-negative size. Quantized inputs and outputs share their scale and
  zero point (int16: zero point 0). `SPACE_TO_BATCH_ND` pads 8-bit
  tensors with the byte of the output zero point and other types with
  zeros, as the reference kernel's `memset` does.

### PRELU

`alpha` broadcasts to the input (up to rank 6), whose shape the output
has; `alpha` may be a constant or computed at run time.

- `rust/prelu_q`: int8 and uint8 take an `alpha` of their type, int16 an
  int8 one; zero points in the range of their type, 0 for int16
  activations. Non-negative inputs are requantized with
  `input_scale / output_scale`, negative ones (times `alpha`) with
  `input_scale × alpha_scale / output_scale`, both computed in `f64` as
  in TensorFlow Lite Micro (LiteRT divides in `f32` first; the golden
  cases check that both agree on their data). TensorFlow Lite Micro has
  no uint8 `PRELU`: uint8 uses the same `f64` formula, so on a rounding
  tie its output can differ by 1 from LiteRT's, the only uint8 reference.
  Refused when a multiplier shifts left by more than 30 bits.
- `rust/prelu_f32`: `x >= 0 ? x : x * alpha`.

### ABS, NEG, SQRT, RSQRT, EXP, LOG

Input and output of the same type and shape.

- `rust/unary_f32`: `EXP` matches glibc's `expf` bit for bit; `LOG` uses
  `libm`'s `logf`, within the float tolerance of glibc's; `SQRT` and
  `RSQRT` (`1 / sqrt(x)`) are correctly rounded square roots.
- `rust/unary_q`: zero points in the range of their type, 0 for int16.
  `ABS` (int8, int16) rescales `|x - input_zero_point|` with
  `input_scale / output_scale` when the scales differ. `RSQRT` (int8)
  computes `1 / sqrt(x - input_zero_point)` with TFLite's fixed-point
  `GetInvSqrtQuantizedMultiplierExp` and requantizes with
  `1 / (sqrt(input_scale) × output_scale)`; an input equal to the zero
  point gives 127, and an input below it (a negative value) makes
  `compute` fail with `InvalidArgument` ([Errors](errors.md)), as TFLite
  fails. Multipliers are computed in `f64` as in TensorFlow Lite Micro
  (LiteRT computes the ratios in `f32`), and refused when they would
  shift by more than 30 bits left or 31 right.
- `rust/unary_int`: LiteRT's `ABS` on int32 and on int16 tensors without
  quantization, `NEG` on int32 and int64; the most negative value maps to
  itself. TensorFlow Lite Micro has none of them.

### DIV, SQUARED_DIFFERENCE

Two inputs of the output's type, broadcasting up to rank 6.

- `rust/binary`: f32 `DIV` with an optional fused activation (a zero
  denominator gives an infinity or NaN, which the activation range clamps
  as in TFLite: `x / 0` becomes the largest finite float of its sign);
  int32 `DIV` without quantization, truncating like C++, where any zero in
  the second input makes `compute` fail with `InvalidArgument`
  ([Errors](errors.md)), as TFLite fails; `SQUARED_DIFFERENCE` on f32 and
  int32 (wrapping around), without activation.
- `rust/binary_q`: `SQUARED_DIFFERENCE` as TensorFlow Lite Micro computes
  it: both inputs rescaled to twice the larger input scale with
  `left_shift = 7` (int8) or `0` (int16), the difference squared in
  `i32`, then requantized. LiteRT has no int16 version (unit tests cover
  it).
- Quantized `DIV` is refused: TensorFlow Lite Micro's `DivElementwise`
  shifts right by more than 31 bits when the quotient is small, which
  C++ leaves undefined and Cortex-M, AArch64 and x86 compute differently,
  so no result is bit-exact with "TFLite" there.

### EQUAL, NOT_EQUAL, LESS, LESS_EQUAL, GREATER, GREATER_EQUAL, SELECT_V2, CAST

- `rust/comparison`: two inputs of the same type, broadcasting up to
  rank 6, a bool output. int8 and uint8 inputs are rescaled to their own
  scale before comparing, as in TFLite: `(x - zero_point) << 8` times
  `QuantizeMultiplierSmallerThanOneExp(scale)`, so each scale must be
  below 1 (TensorFlow Lite Micro requires it) and each zero point in the
  range of its type. bool supports `EQUAL` and `NOT_EQUAL` only. int16 is
  refused: TensorFlow Lite Micro computes int16 `GREATER` without its
  rescaling parameters, and LiteRT compares int16 raw or rescaled
  depending on the operator. TensorFlow Lite Micro has no uint8 version.
- `rust/select_v2`: a bool condition and two value inputs of the
  output's type, all three broadcast to the output's shape (up to rank
  6). Values are copied unchanged, whatever their quantization, as in
  TFLite. `SELECT` (version 1, with its row semantics) is not supported.
- `rust/cast`: input and output of the same shape. Integers convert like
  C++'s `static_cast` (narrowing keeps the low bits), to bool as `x != 0`;
  f32 converts to integers with TensorFlow Lite Micro's `SafeCast`
  (truncation toward zero, saturation, NaN to 0; LiteRT 2.3 maps NaN to
  the largest value instead). Quantization is not looked at: raw values
  are converted. TensorFlow Lite Micro has no uint8 or int64 version.

### Activations

- `rust/relu_q`: int16 needs zero points 0 and is not supported for
  `RELU_N1_TO_1`.
- `rust/leaky_relu_q`: int16 needs zero points 0.
- `rust/lut_q`: a 256-entry table, as TFLite; `LOGISTIC` needs an output
  scale of exactly 1/256.
- `rust/logistic_i16`, `rust/tanh_i16`: TFLite's int16 path
  (`reference_integer_ops::Logistic` and `Tanh`), which interpolates a
  256-entry sigmoid table. Zero points 0 and an output scale of 2^-15
  (TFLite's `CheckedLog2`, within 0.07 %; exactly 1/32768 for `LOGISTIC`,
  as LiteRT's reference kernel requires, where TensorFlow Lite Micro only
  checks the power of two). Any input scale from about 3.8e-14 to 5.33: a power of
  two that TFLite treats specially (2^-12, and 2^-11 for `TANH`) uses the
  multiplier 3; other scales get the multiplier and shift TFLite computes in
  `double`. Outside this range TFLite's multiplier truncates to 0 or the
  product `input × multiplier` overflows an `int32_t`, both undefined in
  C.
- `rust/hard_swish_q`: refused when the requantization multipliers fall
  outside the range of TFLite's fixed-point implementation.

## Coverage by tests

The golden cases of `testdata/` exercise `CONV_2D`, `DEPTHWISE_CONV_2D`,
`FULLY_CONNECTED`, `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `SOFTMAX`, `ADD`,
`SUB`, `MUL`, `MAXIMUM`, `MINIMUM`, `RESHAPE`, `PAD`, `MEAN`, `SUM`,
`REDUCE_MAX`, `REDUCE_MIN`, `PACK`, `UNPACK`, `SPACE_TO_DEPTH`,
`DEPTH_TO_SPACE`, `SPACE_TO_BATCH_ND`, `BATCH_TO_SPACE_ND`, `PRELU`, `ABS`, `NEG`, `SQRT`, `RSQRT`, `EXP`, `LOG`,
`DIV`, `SQUARED_DIFFERENCE`, `EQUAL`, `NOT_EQUAL`, `LESS`, `LESS_EQUAL`, `GREATER`,
`GREATER_EQUAL`, `SELECT_V2`, `CAST`, `LOGISTIC`,
`TANH`, `RELU`, `RELU6`, `CONCATENATION`, `QUANTIZE`, `DEQUANTIZE`,
`TRANSPOSE`, `SLICE`, `STRIDED_SLICE`, `GATHER`, `SPLIT`, `SPLIT_V`,
`ARG_MAX`, `ARG_MIN`, `RESIZE_NEAREST_NEIGHBOR`, `RESIZE_BILINEAR` and
`TRANSPOSE_CONV` against LiteRT (see `testdata/README.md`), and checks that
the int16 cases of `CONV_2D`, `DEPTHWISE_CONV_2D` and pooling run the
CMSIS-NN kernels and those of `LOGISTIC` and `TANH` the Rust ones; the other kernels have unit
tests in `crates/nanoglia-kernels/src/tests/` (int16 `ARG_MAX` / `ARG_MIN`
among them: LiteRT has no int16 version to compare with), and
`crates/nanoglia-cmsis-nn/tests/kernels/` compares every CMSIS-NN kernel,
`transpose_conv_s8` over a sweep of strides, filters, paddings and output
sizes, with ports of the reference kernels. The golden cases also run
with the `esp-nn` backend first (its ANSI C kernels on the host), and
`crates/nanoglia-esp-nn/tests/kernels/` checks its kernels against a
TFLite reference. Host tests run the portable C code of CMSIS-NN and
ESP-NN. `mise run test-emulated` runs the golden cases with CMSIS-NN's
DSP code paths (Cortex-M4F, M33) under an emulator, and the hardware
test bench ran them once on one Cortex-M4F, an Arduino Nano 33 BLE
([run the golden cases on a board](../how-to-guides/run-the-golden-cases-on-a-board.md#what-has-run-on-a-board));
its Helium code paths and the ESP32-S3 kernels of ESP-NN have not run,
and no other chip has (issue #2).
