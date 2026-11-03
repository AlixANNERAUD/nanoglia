# Test data

Models and golden outputs for the bit-exact tests of nanonn. The golden outputs
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
  - `dtype`: one of `int8`, `int16`, `uint8`, `int32`, `float32`;
  - `scale`, `zero_point`: per-tensor quantization of the tensor; `0.0` and `0`
    for a tensor that is not quantized (float);
  - `tolerance`: how far an output may be from the golden one. For quantized
    outputs, the maximum absolute difference in quantized units: always `0`,
    the outputs must be bit-exact. For float outputs, a relative bound:
    `|actual - expected| <= tolerance * max(1, |expected|)` (`1e-5`; `0` for
    `dequantize_int8_f32`, whose result is a single correctly rounded product).

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

Operators covered: `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`,
`AVERAGE_POOL_2D`, `MAX_POOL_2D`, `SOFTMAX`, `ADD`, `SUB`, `MUL`, `MAXIMUM`,
`MINIMUM`, `RESHAPE`, `PAD`, `MEAN`, `LOGISTIC`, `TANH`, `RELU`, `RELU6`,
`CONCATENATION`, `QUANTIZE`, `DEQUANTIZE`.

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
