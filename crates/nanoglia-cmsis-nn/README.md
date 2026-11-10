# nanoglia-cmsis-nn

**Arm CMSIS-NN acceleration for TensorFlow Lite models in Rust:** int8 and int16 convolutions, int8 transpose convolutions, fully-connected layers, pooling and softmax on Cortex-M (DSP, Helium), bit-exact with TensorFlow Lite Micro, `no_std` and allocation-free. Most applications depend on [`nanoglia`](https://crates.io/crates/nanoglia) instead, which enables this crate through its Cargo features.

The [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) kernel backend of [nanoglia](https://github.com/AlixANNERAUD/nanoglia).

[`BACKEND`] implements [`nanoglia_core::backend::Backend`]: it claims the operators CMSIS-NN runs exactly like the TFLite reference kernels and leaves the others to the next backend. Its parameter computations (requantization multipliers, paddings, activation ranges, softmax scaling...) are ported from the CMSIS-NN kernels of TensorFlow Lite Micro and run once, at prepare time; evaluation only fills the CMSIS-NN structs and calls the C function. Nothing allocates: parameters live in the persistent buffer and CMSIS-NN works in the shared scratch buffer the interpreter provides.

On Cortex-M4/M7/M33 CMSIS-NN uses the DSP extension, on Armv8.1-M Helium; on other targets (host, RISC-V...) its portable C code.

| Operator | Types | Conditions |
|---|---|---|
| `CONV_2D` | int8 in/out, int8 weights (per-tensor or per-channel), int32 bias | rank 4, any stride/dilation/padding, grouped convolutions |
| `CONV_2D` (16x8) | int16 in/out (zero point 0), int8 weights (per-tensor or per-channel), constant int64 bias | rank 4, no groups, at most 511 filter values per output channel, requantization shift below 8, accumulators bounded so that CMSIS-NN's 32-bit intermediate cannot overflow |
| `DEPTHWISE_CONV_2D` | int8 in/out, int8 weights (per-tensor or per-channel), int32 bias | rank 4, any depth multiplier, stride, dilation, padding |
| `DEPTHWISE_CONV_2D` (16x8) | int16 in/out (zero point 0), int8 weights (per-tensor or per-channel), constant int64 bias | same, requantization shift below 8, accumulators bounded so that CMSIS-NN's 32-bit intermediate cannot overflow, `(output size - 1) × stride - padding` within int16 |
| `TRANSPOSE_CONV` | int8 in/out, int8 weights (per-tensor or per-channel), optional int32 bias | rank 4, constant output shape, `SAME`/`VALID`; strides up to 2 with more than 16 input channels take any output size, other configurations only those `arm_transpose_conv_s8` computes within its buffers |
| `FULLY_CONNECTED` | int8 in/out, int8 weights `[out, in]` (zero point 0, per-tensor or per-channel), int32 bias | input flattened to `[batches, in]`, `keep_num_dims` either way, default weight format |
| `FULLY_CONNECTED` (16x8) | int16 in/out (zero point 0), int8 weights (per-tensor), constant int64 bias | same, requantization shift below 8, accumulators bounded so that CMSIS-NN's 32-bit intermediate cannot overflow |
| `AVERAGE_POOL_2D`, `MAX_POOL_2D` | int8, int16 (zero point 0) | rank 4, same input/output quantization; int16 average pooling over at most 65535 values |
| `SOFTMAX` | int8 in/out | output zero point -128, scale 1/256 |
| `ADD`, `MUL` | int8, int16 (zero point 0) | inputs and output of the same shape (no broadcasting); int16 `ADD` not with power-of-two scales |
| `MAXIMUM`, `MINIMUM` | int8 | rank up to 4, inputs of the output's shape or a single value |
| `PAD`, `PADV2` | int8 | rank up to 4, constant int32/int64 paddings |

Fused activations: none, `RELU`, `RELU6`, `RELU_N1_TO_1`.

16x8 operators take int64 biases only: LiteRT chooses how to read the bias from the `quantized_bias_type` option, which nanoglia does not decode, and defaults to int64. `SOFTMAX` on int16 is left to the Rust backend: `arm_softmax_s16` does not clamp its result (a row of one element gives -32768 where LiteRT gives 32767) and can overflow while normalizing the sum of the exponentials.

The crate depends on [`nanoglia-cmsis-nn-sys`](https://crates.io/crates/nanoglia-cmsis-nn-sys), which compiles CMSIS-NN with a C compiler for the target.
