# nanoglia-cmsis-nn

The [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) kernel backend of [nanoglia](https://github.com/AlixANNERAUD/nanoglia).

[`BACKEND`] implements [`nanoglia_core::backend::Backend`]: it claims the operators CMSIS-NN runs exactly like the TFLite reference kernels and leaves the others to the next backend. Its parameter computations (requantization multipliers, paddings, activation ranges, softmax scaling...) are ported from the CMSIS-NN kernels of TensorFlow Lite Micro and run once, at prepare time; evaluation only fills the CMSIS-NN structs and calls the C function. Nothing allocates: parameters live in the persistent buffer and CMSIS-NN works in the shared scratch buffer the interpreter provides.

On Cortex-M4/M7/M33 CMSIS-NN uses the DSP extension, on Armv8.1-M Helium; on other targets (host, RISC-V...) its portable C code.

| Operator | Types | Conditions |
|---|---|---|
| `CONV_2D` | int8 in/out, int8 weights (per-tensor or per-channel), int32 bias | rank 4, any stride/dilation/padding, grouped convolutions |
| `CONV_2D` (16x8) | int16 in/out (zero point 0), int8 weights (per-tensor or per-channel), int64 bias | rank 4, no groups, at most 511 filter values per output channel |
| `DEPTHWISE_CONV_2D` | same | rank 4, any depth multiplier, stride, dilation, padding |
| `FULLY_CONNECTED` | int8 in/out, int8 weights `[out, in]` (zero point 0, per-tensor or per-channel), int32 bias | input flattened to `[batches, in]`, `keep_num_dims` either way, default weight format |
| `FULLY_CONNECTED` (16x8) | int16 in/out (zero point 0), int8 weights (per-tensor), int64 bias | same |
| `AVERAGE_POOL_2D`, `MAX_POOL_2D` | int8 | rank 4, same input/output quantization |
| `SOFTMAX` | int8 in/out | output zero point -128, scale 1/256 |
| `ADD`, `MUL` | int8, int16 (zero point 0) | inputs and output of the same shape (no broadcasting); int16 `ADD` not with power-of-two scales |
| `MAXIMUM`, `MINIMUM` | int8 | rank up to 4, inputs of the output's shape or a single value |
| `PAD`, `PADV2` | int8 | rank up to 4, constant int32/int64 paddings |

Fused activations: none, `RELU`, `RELU6`, `RELU_N1_TO_1`.

16x8 operators take int64 biases only: LiteRT chooses how to read the bias from the `quantized_bias_type` option, which nanoglia does not decode, and defaults to int64. `SOFTMAX` on int16 is left to the Rust backend: `arm_softmax_s16` does not clamp its result (a row of one element gives -32768 where LiteRT gives 32767) and can overflow while normalizing the sum of the exponentials.

The crate depends on [`nanoglia-cmsis-nn-sys`](https://crates.io/crates/nanoglia-cmsis-nn-sys), which compiles CMSIS-NN with a C compiler for the target.
