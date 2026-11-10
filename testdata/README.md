# Test data

Models and golden outputs for the bit-exact tests of nanoglia. The golden outputs
are computed by the LiteRT interpreter with its reference kernels
(`OpResolverType.BUILTIN_REF`), which give the same results as TensorFlow Lite
Micro.

```
testdata/
├── ops/<case>/                         one operator per model
│   ├── model.tflite
│   ├── input_<i>.bin, output_<i>.bin
│   └── case.json
└── mlperf-tiny/
    ├── *.tflite                        MLPerf Tiny reference models (see its README)
    └── golden/<model>/sample_<k>/      3 random inputs per model
        ├── input_<i>.bin, output_<i>.bin
        └── case.json
```

## Cases

Every case directory holds:

- `input_<i>.bin` / `output_<i>.bin`: the raw little-endian bytes of the
  subgraph input / output tensor `i`, in the order of the model's subgraph
  inputs and outputs;
- `case.json`:

  ```json
  {
    "model": "testdata/ops/conv2d_3x3_same_int8/model.tflite",
    "inputs": [
      {"name": "input", "dtype": "int8", "shape": [1, 8, 8, 4], "scale": 0.01557568833231926, "zero_point": -65}
    ],
    "outputs": [
      {"name": "output", "dtype": "int8", "shape": [1, 8, 8, 8], "scale": 0.05364882946014404, "zero_point": 18}
    ],
    "tolerance": 0
  }
  ```

  - `model`: path of the model, relative to the repository root;
  - `dtype`: one of `int8`, `int16`, `uint8`, `int32`, `uint32`, `int64`
    (indices fed to `GATHER`, indices computed by `ARG_MAX` / `ARG_MIN`),
    `bool` (comparisons, `SELECT_V2`, `CAST`), `float32`;
  - `scale`, `zero_point`: per-tensor quantization of the tensor; `0.0` and `0`
    for a tensor that is not quantized (float);
  - `tolerance`: how far an output may be from the golden one. For quantized
    outputs, the maximum absolute difference in quantized units: always `0`,
    the outputs must be bit-exact. For float outputs, a relative bound:
    `|actual - expected| <= tolerance * max(1, |expected|)` (`1e-5`; `0` for
    `dequantize_int8_f32`, whose result is a single correctly rounded product,
    and for the data-movement operators below, which copy floats unchanged).

## Per-op cases

`ops/` has one case per operator configuration, named
`<operator>_<variant>_<type>`:

- `_int8`: int8 activations with asymmetric quantization (nonzero zero points),
  int8 weights with symmetric per-channel quantization (zero point 0; per-tensor
  in `fully_connected_per_tensor_int8`), int32 biases with scale
  `input_scale * weight_scale`. Output ranges are calibrated on the float model,
  as a converter would.
- `_int16`: 16x8 quantization: int16 activations with zero point 0, int8
  weights, int64 biases. The int16 `LOGISTIC` and `TANH` cases fix the input
  scale instead of calibrating it, to take each path of TFLite's prepare step
  (power-of-two scales, a multiplier doubled up to the table's scale, scales
  above 16383.5 / 12288 without a shift, a tiny scale) on inputs that cover the
  whole int16 range; their output scale is 2^-15.
- `_f32`: float32 everything.

The variants of a case share their float data: `conv2d_3x3_same_int8` is the
quantized version of `conv2d_3x3_same_f32`.

The data-movement operators (`TRANSPOSE`, `SLICE`, `STRIDED_SLICE`, `GATHER`,
`SPLIT`, `SPLIT_V`) only copy bytes: their cases cover several ranks (up to 6),
element types (`_f32`, `_int8`, `_uint8`, `_int16`, `_int32`; quantized inputs
and outputs share one quantization, as converters write them) and options
(negative axes and permutations, int64 indices, `size = -1`, the masks and
`offset` of `STRIDED_SLICE`, `batch_dims`, `SPLIT_V` sizes with -1). The oracle
also checks each of their LiteRT outputs against NumPy. LiteRT 2.3's reference
resolver registers `STRIDED_SLICE` versions 1 to 4 only, all with the same
kernel: the ellipsis, new-axis and offset cases declare version 4 instead of
the 6 or 8 a converter would write.

The vision operators:

- `ARG_MAX` / `ARG_MIN` (`_f32`, `_int8`, `_uint8`, `_int32` inputs; LiteRT
  2.3 has no int16 version) draw their values from nine levels so that ties
  are frequent, over several ranks and axes (negative, int64), with int32 or
  int64 indices; the oracle checks them against NumPy's `argmax` / `argmin`.
- `RESIZE_NEAREST_NEIGHBOR` and `RESIZE_BILINEAR` (`_f32`, `_int8`,
  `_uint8`, `_int16`; `_int32` for nearest neighbor) upscale and downscale odd
  sizes with every combination of `align_corners` and `half_pixel_centers`
  that LiteRT accepts; input and output share one quantization. The oracle
  checks nearest neighbor and the fixed-point bilinear path (int8, int16)
  exactly against NumPy, the float bilinear path (f32, uint8) within a
  rounding.
- `TRANSPOSE_CONV` (`_int8` per-channel or per-tensor, `_f32`) covers strides
  1 to 3, `SAME` and `VALID`, odd output sizes, the largest `VALID` output,
  more and fewer than 16 input channels (the two CMSIS-NN paths), two batches,
  fused activations (version 4), with, without and with an omitted bias.

The reductions `SUM`, `REDUCE_MAX` and `REDUCE_MIN` (`_f32`, `_int8`, `_uint8`,
`_int16`, `_int32`) cover several ranks, negative and duplicate axes,
`keep_dims`, reductions over every axis and a saturated int8 sum.
`REDUCE_MAX` / `REDUCE_MIN` keep the input quantization (TFLite requires it)
and the oracle checks them against NumPy. LiteRT 2.3's reference `SUM` reads
int8 and int16 data as uint8 bytes, so its int8 and int16 outputs are wrong:
the golden outputs of these cases come from the equivalent uint8 model (data
and zero points plus 128), which LiteRT computes right, shifted back. The
int16 cases keep their inputs in the int8 range and their sums from saturating
in uint8 for that, and declare `SUM` version 2 (LiteRT 2.3 registers versions
1 and 2 only).

The layout operators `PACK`, `UNPACK`, `SPACE_TO_DEPTH`, `DEPTH_TO_SPACE`,
`SPACE_TO_BATCH_ND` and `BATCH_TO_SPACE_ND` (`_f32`, `_int8`, `_uint8`,
`_int16`, `_int32`, `_int64`) cover negative axes, scalars, rank 3 and 4,
several block sizes, paddings (quantized ones with the zero point) and crops;
the oracle checks them against NumPy. LiteRT 2.3's reference resolver
registers `DEPTH_TO_SPACE` version 1 only: its int8 case declares version 1
instead of 2.

`PRELU` (`_f32`, `_int8`, `_uint8`, `_int16` with an int8 `alpha`) covers a
per-channel `alpha`, an `alpha` of the input's shape fed at run time, and a
saturated output. LiteRT computes the two requantization ratios in float,
TensorFlow Lite Micro in double: the oracle checks each quantized output
against TFLM's formula (a Python port of its fixed-point requantization), so
that the golden data holds for both. LiteRT 2.3's reference resolver registers
`PRELU` version 1 only, which every case declares.

The unary operators `ABS`, `NEG`, `SQRT`, `RSQRT`, `EXP` and `LOG` (`_f32`
with zeros, infinities and values that make `LOG` return `-inf`; quantized
`ABS` on int8 and int16, with and without rescaling; int8 `RSQRT` with an
input at the zero point; `ABS` and `NEG` on raw integers). LiteRT computes the
quantized `ABS` and `RSQRT` ratios in float, TensorFlow Lite Micro in double:
these cases use scales whose ratios both compute to the same multiplier (the
oracle checks it), so that the golden data holds for both.

`DIV` (`_f32` with broadcasting, a fused activation, zero denominators and
`0 / 0`; `_int32` with and without activation) and `SQUARED_DIFFERENCE`
(`_f32`, `_int32`, `_int8`, with broadcasting). LiteRT 2.3 has no int16
`SQUARED_DIFFERENCE`, and quantized `DIV` is not supported by nanoglia.

The comparisons (`_f32`, `_int32`, `_int64`, `_bool`, and quantized `_int8`
and `_uint8` inputs of different scales, with broadcasting) draw their values
from a few levels so that ties are frequent; `SELECT_V2` (`_f32`, `_int8`,
`_int16`, `_int32`) broadcasts its three inputs, one case with a scalar
condition; `CAST` covers every conversion family (float to integer with
truncation and saturation, integer narrowing and widening, bool, uint32), with
no NaN input (TensorFlow Lite Micro converts NaN to 0, LiteRT 2.3 to the
largest value). The oracle checks them against NumPy, except the rescaled
quantized comparisons and the saturating casts.

Operators covered: `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`,
`AVERAGE_POOL_2D`, `MAX_POOL_2D`, `SOFTMAX`, `ADD`, `SUB`, `MUL`, `MAXIMUM`,
`MINIMUM`, `RESHAPE`, `PAD`, `MEAN`, `LOGISTIC`, `TANH`, `RELU`, `RELU6`,
`CONCATENATION`, `QUANTIZE`, `DEQUANTIZE`, `TRANSPOSE`, `SLICE`, `STRIDED_SLICE`,
`GATHER`, `SPLIT`, `SPLIT_V`, `ARG_MAX`, `ARG_MIN`, `RESIZE_NEAREST_NEIGHBOR`,
`RESIZE_BILINEAR`, `TRANSPOSE_CONV`, `SUM`, `REDUCE_MAX`, `REDUCE_MIN`, `PACK`, `UNPACK`,
`SPACE_TO_DEPTH`, `DEPTH_TO_SPACE`, `SPACE_TO_BATCH_ND`, `BATCH_TO_SPACE_ND`, `PRELU`, `ABS`, `NEG`, `SQRT`,
`RSQRT`, `EXP`, `LOG`, `DIV`, `SQUARED_DIFFERENCE`, `EQUAL`, `NOT_EQUAL`, `LESS`,
`LESS_EQUAL`, `GREATER`, `GREATER_EQUAL`, `SELECT_V2`, `CAST`.

Not covered: int8 `CONCATENATION` with different input and output
quantization, which the reference kernel rejects (as TensorFlow Lite Micro does).

## Pending cases

`pending.json` lists the cases expected to fail for now, each with the
reason (`{"case": "<path under testdata/>", "reason": "..."}`). Every
runner reads it: `crates/nanoglia/tests/golden.rs` on the host, and
`emulation/tools/golden.py` for `emulation/tools/run.py` under the
emulator and `hardware/tools` on a board. A listed case that passes
fails the run, so the list never goes stale.

## MLPerf Tiny golden outputs

`mlperf-tiny/golden/<model>/sample_<k>/` runs `mlperf-tiny/<model>.tflite` on
inputs drawn uniformly over the whole int8 range.

## Regenerating

```sh
mise run oracle
```

This runs `tools/oracle/generate.py` (a [uv](https://docs.astral.sh/uv/)
project) which writes the per-op models with `tools/oracle/tflite_builder.py`,
a small `.tflite` writer built on the `flatbuffers` package, and runs every
case through LiteRT. Generation is deterministic: every case seeds its own
random generator from its name, and re-running the script rewrites identical
files. It deletes and rewrites `ops/` and `mlperf-tiny/golden/`.

## A PyTorch export

`mise run pytorch-export` (x86_64 Linux or macOS arm64, where litert-torch's
converter has wheels; CI runs it on ubuntu) exports the small classifier of
`tools/oracle/pytorch_export.py` to a fully int8 model with litert-torch,
writes cases in the format above into `target/pytorch-export/sample_<k>/`
(with an absolute `model` path when it lies outside the repository), and runs
them with `NANOGLIA_GOLDEN_DIR` pointing there: `golden.rs` then runs that
directory instead of `testdata/`. These cases are not committed, as the
converted model may change with the versions of torch and litert-torch.
