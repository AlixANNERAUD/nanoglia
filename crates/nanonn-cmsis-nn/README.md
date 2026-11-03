# nanonn-cmsis-nn

The [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) kernel backend of [nanonn](https://github.com/AlixANNERAUD/nanonn).

[`BACKEND`] implements [`nanonn_core::backend::Backend`]: it claims the operators CMSIS-NN runs exactly like the TFLite reference kernels and leaves the others to the next backend. Its parameter computations (requantization multipliers, paddings, activation ranges, softmax scaling...) are ported from the CMSIS-NN kernels of TensorFlow Lite Micro and run once, at prepare time; evaluation only fills the CMSIS-NN structs and calls the C function. Nothing allocates: parameters live in the persistent buffer and CMSIS-NN works in the shared scratch buffer the interpreter provides.

On Cortex-M4/M7/M33 CMSIS-NN uses the DSP extension, on Armv8.1-M Helium; on other targets (host, RISC-V...) its portable C code.

| Operator | Types | Conditions |
|---|---|---|
| `CONV_2D` | int8 in/out, int8 weights (per-tensor or per-channel), int32 bias | rank 4, any stride/dilation/padding, grouped convolutions |
| `DEPTHWISE_CONV_2D` | same | rank 4, any depth multiplier, stride, dilation, padding |
| `FULLY_CONNECTED` | int8 in/out, int8 weights `[out, in]` (zero point 0, per-tensor or per-channel), int32 bias | input flattened to `[batches, in]`, `keep_num_dims` either way, default weight format |
| `AVERAGE_POOL_2D`, `MAX_POOL_2D` | int8 | rank 4, same input/output quantization |
| `SOFTMAX` | int8 in/out | output zero point -128, scale 1/256 |
| `ADD`, `MUL` | int8 | inputs and output of the same shape (no broadcasting) |

Fused activations: none, `RELU`, `RELU6`, `RELU_N1_TO_1`.

The crate depends on [`nanonn-cmsis-nn-sys`](https://crates.io/crates/nanonn-cmsis-nn-sys), which compiles CMSIS-NN with a C compiler for the target.
