# Supported operators

Every TFLite builtin operator nanoglia runs, for which tensor types, on
which backend, and under which conditions. The conditions come from the
`select` and `plan` functions of the backends
(`crates/nanoglia-esp-nn/src/*.rs`, `crates/nanoglia-cmsis-nn/src/*.rs`
and `crates/nanoglia-kernels/src/*.rs`): an operator that does not meet them
is not claimed, and loading fails with `Detail::UnsupportedOp`
([Errors](errors.md)).

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
([Cargo features](features.md)): kernels marked *float* need feature
`float`, kernels marked *int16* need feature `int16`. Everything else is
always there when its backend is.

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
| | 16x8 | `cmsis-nn/conv2d_s16` | int16 |
| | f32 | `rust/conv_2d_f32` | float |
| `DEPTHWISE_CONV_2D` | int8 | `cmsis-nn/depthwise_conv2d_s8` | — |
| | f32 | `rust/depthwise_conv_2d_f32` | float |
| `FULLY_CONNECTED` | int8 | `cmsis-nn/fully_connected_s8` | — |
| | 16x8, per-tensor weights, small shift | `cmsis-nn/fully_connected_s16` | int16 |
| | 16x8, otherwise | `rust/fully_connected_16x8` | int16 |
| | f32 | `rust/fully_connected_f32` | float |
| `AVERAGE_POOL_2D` | int8 | `cmsis-nn/average_pool2d_s8` | — |
| | f32 | `rust/pool_f32` | float |
| `MAX_POOL_2D` | int8 | `cmsis-nn/max_pool2d_s8` | — |
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
| `QUANTIZE` | f32 → int8, uint8, int16 | `rust/quantize` | — |
| | int8, uint8 → int8, uint8, int16; int16 → int8, int16; int32 → int8, int16 | `rust/requantize` | — |
| `DEQUANTIZE` | int8, uint8, int16, f16 → f32 | `rust/dequantize` | — |
| `MEAN` | f32, int8, uint8, int16 | `rust/mean` | — |
| `RELU`, `RELU6`, `RELU_N1_TO_1` | f32 | `rust/relu_f32` | float |
| | int8, uint8, int16 | `rust/relu_q` | — |
| `LEAKY_RELU` | f32 | `rust/leaky_relu_f32` | float |
| | int8, uint8, int16 | `rust/leaky_relu_q` | — |
| `LOGISTIC` | f32 | `rust/logistic_f32` | float |
| | int8, uint8 | `rust/lut_q` | — |
| `TANH` | f32 | `rust/tanh_f32` | float |
| | int8, uint8 | `rust/lut_q` | — |
| `HARD_SWISH` | f32 | `rust/hard_swish_f32` | float |
| | int8, uint8 | `rust/hard_swish_q` | — |

Any other operator, any custom operator, and any type not listed (for
example int16 `DEPTHWISE_CONV_2D` or pooling, uint8 `ADD`, int4 weights)
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
  outputs in 32 bits before adding the bias), and a requantization shift
  below 8 for every channel.
- `rust/conv_2d_f32`: grouped convolutions included.

### DEPTHWISE_CONV_2D

- `cmsis-nn/depthwise_conv2d_s8`: filter `[1, height, width, output
  channels]`, output channels a multiple of input channels (the depth
  multiplier is derived from the shapes, as LiteRT does), weights per
  tensor or per channel along axis 3, int32 bias. Same dimension, stride
  and dilation limits as `conv2d_s8`.
- `rust/depthwise_conv_2d_f32`: same shapes.

There is no int16 depthwise convolution.

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
  shift below 8.
- `rust/fully_connected_16x8`: what `fully_connected_s16` refuses: weights
  per tensor or per channel (axis 0), int64 bias, zero points 0, shift up
  to 14.
- `rust/fully_connected_f32`: f32 weights and bias.

### AVERAGE_POOL_2D, MAX_POOL_2D

- `cmsis-nn/*_pool2d_s8`: rank-4 input and output, input and output
  sharing their quantization (scales within 1e-6, equal zero points),
  filter sizes and strides in `1..=32767`.
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

### QUANTIZE, DEQUANTIZE

- `rust/quantize`: per-tensor output quantization.
- `rust/requantize`: per-tensor quantization on both sides; int16 to
  int16 needs zero points 0.
- `rust/dequantize`: int8 and uint8 per tensor or per channel, int16 per
  tensor with zero point 0, f16.

### MEAN

The axes are a constant int32 or int64 tensor (negative axes wrap,
duplicates are ignored); input and output of the same type; int16 needs
zero points 0.

### Activations

- `rust/relu_q`: int16 needs zero points 0 and is not supported for
  `RELU_N1_TO_1`.
- `rust/leaky_relu_q`: int16 needs zero points 0.
- `rust/lut_q`: a 256-entry table, as TFLite; `LOGISTIC` needs an output
  scale of exactly 1/256.
- `rust/hard_swish_q`: refused when the requantization multipliers fall
  outside the range of TFLite's fixed-point implementation.

## Coverage by tests

The golden cases of `testdata/` exercise `CONV_2D`, `DEPTHWISE_CONV_2D`,
`FULLY_CONNECTED`, `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `SOFTMAX`, `ADD`,
`SUB`, `MUL`, `MAXIMUM`, `MINIMUM`, `RESHAPE`, `PAD`, `MEAN`, `LOGISTIC`,
`TANH`, `RELU`, `RELU6`, `CONCATENATION`, `QUANTIZE`, `DEQUANTIZE`,
`TRANSPOSE`, `SLICE`, `STRIDED_SLICE`, `GATHER`, `SPLIT` and `SPLIT_V`
against LiteRT (see `testdata/README.md`); the other kernels have unit
tests in `crates/nanoglia-kernels/src/tests/`. The golden cases also run
with the `esp-nn` backend first (its ANSI C kernels on the host), and
`crates/nanoglia-esp-nn/tests/kernels/` checks its kernels against a
TFLite reference. The Arm DSP and Helium code paths of CMSIS-NN and the
ESP32-S3 kernels of ESP-NN have not run on hardware yet (issue #2): host
tests run their portable C code.
