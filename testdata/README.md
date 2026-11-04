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
  - `dtype`: one of `int8`, `int16`, `uint8`, `int32`, `int64` (indices fed
    to `GATHER`), `float32`;
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
  weights, int64 biases.
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

Operators covered: `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`,
`AVERAGE_POOL_2D`, `MAX_POOL_2D`, `SOFTMAX`, `ADD`, `SUB`, `MUL`, `MAXIMUM`,
`MINIMUM`, `RESHAPE`, `PAD`, `MEAN`, `LOGISTIC`, `TANH`, `RELU`, `RELU6`,
`CONCATENATION`, `QUANTIZE`, `DEQUANTIZE`, `TRANSPOSE`, `SLICE`, `STRIDED_SLICE`,
`GATHER`, `SPLIT`, `SPLIT_V`.

Not covered: int8 `CONCATENATION` with different input and output
quantization, which the reference kernel rejects (as TensorFlow Lite Micro does).

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
