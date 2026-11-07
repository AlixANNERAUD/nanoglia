# nanoglia-kernels

**Pure Rust neural-network kernels, bit-exact with TensorFlow Lite, for `no_std` targets:** quantization, activations, element-wise operations with broadcasting, int16 and float32 operators, without allocating or panicking. Most applications depend on [`nanoglia`](https://crates.io/crates/nanoglia) instead, which enables this crate through its Cargo features.

Pure-Rust kernels of [nanoglia](https://github.com/AlixANNERAUD/nanoglia), exposed as the `"rust"` [`Backend`](nanoglia_core::backend::Backend) ([`BACKEND`]).

They cover what the optimized C libraries lack: shape, data-movement and quantization operators, a few int8 operators, and `f32` versions of the main operators. Each kernel ports the TensorFlow Lite reference kernel (`BUILTIN_REF` resolver) and its preparation code, including how multipliers and lookup tables are computed, so that results are bit-exact.

| Operator | Types |
|---|---|
| `RESHAPE`, `SQUEEZE`, `EXPAND_DIMS` | any |
| `QUANTIZE` | `f32` → `i8`/`u8`/`i16`; `i8`/`u8` → `i8`/`u8`/`i16`; `i16` → `i8`/`i16`; `i32` → `i8`/`i16` |
| `DEQUANTIZE` | `i8`/`u8` (per-tensor or per-channel), `i16`, `f16` → `f32` |
| `MEAN` | `f32`, `i8`, `u8`, `i16` |
| `SUM`, `REDUCE_MAX`, `REDUCE_MIN` | `f32`, `i8`, `u8`, `i16`, `i32`, `i64` |
| `PACK`, `UNPACK` | `f32`, `i8`, `u8`, `i16`, `i32` (`PACK`: `u32`, `i64`; `UNPACK`: `f16`, `bf16`, `bool`), up to 16 inputs or 8 outputs |
| `SPACE_TO_DEPTH`, `DEPTH_TO_SPACE` | `f32`, `i8`, `u8`, `i32`, `i64` |
| `SPACE_TO_BATCH_ND`, `BATCH_TO_SPACE_ND` | `f32`, `i8`, `u8`, `i16`, `i32`, `i64`, rank 3 or 4, constant block shape and paddings or crops |
| `RELU`, `RELU6`, `RELU_N1_TO_1` | `f32`, `i8`, `u8`, `i16` |
| `LEAKY_RELU` | `f32`, `i8`, `u8`, `i16` |
| `PRELU` | `f32`, `i8`, `u8`, `i16` (`i8` alpha), broadcasting |
| `LOGISTIC`, `TANH` | `f32`, `i8`, `u8`, `i16` (TFLite's interpolated sigmoid table) |
| `HARD_SWISH` | `f32`, `i8`, `u8` |
| `PAD`, `PADV2` | any |
| `CONCATENATION` | any (quantized inputs must share the output's parameters) |
| `TRANSPOSE`, `SLICE`, `STRIDED_SLICE` | any type of 1, 2, 4 or 8 bytes, rank up to 6, constant indices |
| `GATHER` | `f32`, `f16`, `bf16`, `i8`, `u8`, `i16`, `i32`, `i64`, `bool`; `i16`/`i32`/`i64` indices, possibly computed at run time |
| `SPLIT`, `SPLIT_V` | `f32`, `i8`, `u8`, `i16`, `i32`, `i64`, constant axis and sizes, up to 8 outputs |
| `ARG_MAX`, `ARG_MIN` | `f32`, `i8`, `u8`, `i16`, `i32`, `bool` → `i32`/`i64` indices, constant axis |
| `RESIZE_NEAREST_NEIGHBOR` | `f32`, `i8`, `u8`, `i16`, `i32`, `bool`, constant size, `align_corners`, `half_pixel_centers` |
| `RESIZE_BILINEAR` | `f32`, `u8` (in float), `i8`, `i16` (in fixed point), constant size, `align_corners` or `half_pixel_centers` |
| `MAXIMUM`, `MINIMUM` | `f32`, `i8`, `u8`, `i16`, `i32`, `i64` (broadcasting) |
| `ADD`, `SUB`, `MUL` | `f32`, `i8`, `i16` (broadcasting; int16 `ADD`/`SUB` not with power-of-two scales) |
| `CONV_2D`, `DEPTHWISE_CONV_2D` | `f32` |
| `TRANSPOSE_CONV` | `f32`; `i8` (weights per tensor or per channel, `i32` bias), constant output shape |
| `FULLY_CONNECTED` | `f32`; 16x8 (`i16` activations, `i8` weights per tensor or per channel, `i64` bias) |
| `AVERAGE_POOL_2D`, `MAX_POOL_2D` | `f32` |
| `SOFTMAX` | `f32`, `i16` |

Everything is computed when the operator is prepared; evaluation reads only the prepared parameters and the tensors, never allocates and never panics. `#![no_std]`, no allocation; the only dependency besides `nanoglia-core` is [`libm`](https://crates.io/crates/libm) for `expf`, `tanhf` and the `exp` of the int16 softmax tables.
