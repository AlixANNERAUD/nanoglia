# nanonn-kernels

Pure-Rust kernels of [nanonn](https://github.com/AlixANNERAUD/nanonn), exposed as the `"rust"` [`Backend`](nanonn_core::backend::Backend) ([`BACKEND`]).

They cover what the optimized C libraries lack: shape and quantization operators, a few int8 operators, and `f32` versions of the main operators. Each kernel ports the TensorFlow Lite reference kernel (`BUILTIN_REF` resolver) and its preparation code, including how multipliers and lookup tables are computed, so that results are bit-exact.

| Operator | Types |
|---|---|
| `RESHAPE`, `SQUEEZE`, `EXPAND_DIMS` | any |
| `QUANTIZE` | `f32` → `i8`/`u8`/`i16`; `i8`/`u8` → `i8`/`u8`/`i16`; `i16` → `i8`/`i16`; `i32` → `i8`/`i16` |
| `DEQUANTIZE` | `i8`/`u8` (per-tensor or per-channel), `i16`, `f16` → `f32` |
| `MEAN` | `f32`, `i8`, `u8`, `i16` |
| `RELU`, `RELU6`, `RELU_N1_TO_1` | `f32`, `i8`, `u8`, `i16` |
| `LEAKY_RELU` | `f32`, `i8`, `u8`, `i16` |
| `LOGISTIC`, `TANH` | `f32`, `i8`, `u8` |
| `HARD_SWISH` | `f32`, `i8`, `u8` |
| `PAD`, `PADV2` | any |
| `CONCATENATION` | any (quantized inputs must share the output's parameters) |
| `MAXIMUM`, `MINIMUM` | `f32`, `i8`, `u8`, `i16`, `i32`, `i64` (broadcasting) |
| `ADD`, `SUB`, `MUL` | `f32`, `i8`, `i16` (broadcasting; int16 `ADD`/`SUB` not with power-of-two scales) |
| `CONV_2D`, `DEPTHWISE_CONV_2D` | `f32` |
| `FULLY_CONNECTED` | `f32`; 16x8 (`i16` activations, `i8` weights per tensor or per channel, `i64` bias) |
| `AVERAGE_POOL_2D`, `MAX_POOL_2D` | `f32` |
| `SOFTMAX` | `f32`, `i16` |

Everything is computed when the operator is prepared; evaluation reads only the prepared parameters and the tensors, never allocates and never panics. `#![no_std]`, no allocation; the only dependency besides `nanonn-core` is [`libm`](https://crates.io/crates/libm) for `expf`, `tanhf` and the `exp` of the int16 softmax tables.
