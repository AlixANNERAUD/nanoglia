# Convert a PyTorch model

nanoglia runs `.tflite` (LiteRT) models. This guide turns a PyTorch
model into a fully quantized int8 `.tflite` file with Google's
converter, checks that nanoglia can run it, and checks that nanoglia's
output matches LiteRT's. It ends with the routes from an existing ONNX
model and from Keras.

The tutorial [From training to the device](../tutorials/train-and-deploy.md)
follows these steps from start to finish on a small model, with a
PyTorch script and a Keras script you can run.

The converter is a fast-moving project: this guide shows the calls of
its documentation at the time of writing and links to it. If a name has
changed, follow the upstream page.

## What nanoglia expects

Microcontroller kernels compute in integers. Convert to **full integer
quantization**:

- int8 activations, quantized per tensor and asymmetric (any zero point);
- int8 weights, quantized per output channel and symmetric (zero point
  0); per-tensor weights work too;
- int32 biases, with the scale `input scale × weight scale`.

This is TFLite's int8 scheme, which CMSIS-NN implements. The **16x8**
scheme is supported as well for the operators listed in
[Supported operators](../reference/supported-operators.md): int16
activations with zero point 0, int8 weights, int64 biases.

Avoid these variants:

- **Dynamic range quantization** (int8 weights, float activations,
  quantized at run time): its hybrid kernels are not supported, and a
  model converted this way fails to load with `Detail::UnsupportedOp`.
- **Float16 weights**: they load (each weight goes through `DEQUANTIZE`
  to f32), but the model then computes in f32, which is slow on a
  microcontroller and needs the `float` feature, and the dequantized
  weights take room in every execution context.
- A float32 model runs (feature `float`), but is four times larger than
  its int8 version and much slower.

## 1. Install the converter

The converter is [litert-torch](https://github.com/google-ai-edge/litert-torch)
(formerly `ai-edge-torch`, whose PyPI package is now deprecated in its
favour). Version 0.10.0 requires Python 3.10 or later and depends on
`litert-converter` 0.5, which ships wheels for **x86_64 Linux and macOS
on Apple silicon only**: on another system (for example Linux on
aarch64), convert in an x86_64 Linux container or machine.

```sh
pip install litert-torch torchvision ai-edge-litert
```

## 2. Quantize and convert

litert-torch quantizes with PyTorch's PT2E flow: it inserts observers
in the exported graph, you run representative inputs through it
(calibration), and it converts the observed ranges into quantization
parameters. The code below follows the
[PyTorch converter documentation](https://github.com/google-ai-edge/litert-torch/blob/main/docs/pytorch_converter/README.md)
of litert-torch, with one change: upstream, the example passes
`is_dynamic=True`, which produces dynamic range quantization. Leave it
at its default, `False`, for static (full integer) quantization, and keep
per-channel weights.

```python
import torch
import litert_torch
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
from litert_torch.quantize.quant_config import QuantConfig

model = MyModel().eval()                     # your trained model
sample_args = (torch.randn(1, 49, 10, 1),)   # one input, with the shape the device will feed

quantizer = PT2EQuantizer().set_global(
    get_symmetric_quantization_config(is_per_channel=True)  # static: is_dynamic=False
)
exported = torch.export.export(model, sample_args).module()  # PyTorch 2.6 or later
prepared = prepare_pt2e(exported, quantizer)

# Calibration: run representative inputs (a few hundred real samples).
with torch.no_grad():
    for batch in calibration_samples:
        prepared(batch)

quantized = convert_pt2e(prepared, fold_quantize=False)
edge_model = litert_torch.convert(
    quantized, sample_args, quant_config=QuantConfig(pt2e_quantizer=quantizer)
)
edge_model.export("model.tflite")
```

What `get_symmetric_quantization_config` asks for, in litert-torch's
[source](https://github.com/google-ai-edge/litert-torch/blob/main/litert_torch/quantize/pt2e_quantizer.py):
int8 activations quantized per tensor and asymmetric (`-128..127`), int8
weights symmetric (`-127..127`), per channel along axis 0 with
`is_per_channel=True`. That is the scheme above.

Calibrate with real data: the observed ranges set the scales of every
activation, and random inputs give ranges that real inputs do not have.

Two limits of litert-torch 0.10.0 show in the converted model:

- **The input and output stay float32.** The model starts with a
  `QUANTIZE` operator and ends with a `DEQUANTIZE`; `litert_torch.convert` has no
  documented option for an int8 interface. nanoglia runs both. To feed
  int8 values instead, as a model from TensorFlow's converter takes
  them, remove the two operators from the file: `strip_float_interface` in
  [`examples/train-to-device/training/train_pytorch.py`](../../examples/train-to-device/training/train_pytorch.py)
  does it in about thirty lines with the flatbuffer classes of `ai-edge-litert`.
- **`torch.softmax` stays float32**, behind a `DEQUANTIZE`: the
  quantizer annotates `aten._softmax`, not the `aten.softmax.int` that
  `torch.softmax` exports as. Call
  `torch.ops.aten._softmax(x, 1, False)` in the model (dimension 1, not
  half precision) to get an int8 `SOFTMAX`, or leave the softmax out and
  take the largest logit. Decomposing the whole program first
  (`run_decompositions()`) also reaches `aten._softmax`, but turns
  `nn.Linear` into an `addmm` whose weights the quantizer makes
  asymmetric: nanoglia then refuses the `FULLY_CONNECTED` operator,
  whose weights must have zero point 0.

Both were observed on 2026-11-10 with litert-torch 0.10.0 and torch
2.14.1 on x86_64 Linux, on the model of that example.

`tools/oracle/pytorch_export.py` in the repository is a complete,
runnable version of this step for a small image classifier, an LSTM and
a GRU, which nanoglia's CI converts on x86_64 Linux and runs against
LiteRT (`mise run pytorch-export`); `examples/train-to-device` trains a
model first (`mise run train-to-device`).

Besides the layers, a PyTorch export contains data-movement operators:
`TRANSPOSE` (`permute`, and where the converter moves between PyTorch's
channels-first layout and LiteRT's channels-last one), `SLICE` or
`STRIDED_SLICE` (indexing, `narrow`), `GATHER` (`index_select`,
embeddings), `SPLIT` or `SPLIT_V` (`split`, `chunk`), `CONCATENATION`
(`cat`) and `RESHAPE` (`view`, `flatten`). nanoglia runs all of them, on
int8 tensors as on the other types, with constant permutations, slice
bounds and split sizes; `GATHER` indices may be computed by the model.
onnx2tf produces the same operators. See
[Supported operators](../reference/supported-operators.md#transpose-slice-strided_slice-gather-split-split_v)
for the exact conditions.

### Recurrent layers

litert-torch converts `nn.LSTM` and `nn.GRU` by unrolling them over the
time steps: each step becomes a `FULLY_CONNECTED`, a few `SLICE`s,
`LOGISTIC`, `TANH`, `MUL` and `ADD` (and `SUB` for a GRU). Its quantizer
quantizes only the inputs and outputs of the layer, so the cells compute
in float32, with float32 weights. nanoglia runs these models with the
`float` feature, and its CI checks them against LiteRT bit for bit. The
cells do not use CMSIS-NN, and the number of operators grows with the
number of time steps: the LSTM of the example has 137 operators for 8
steps.

- The number of time steps is fixed at export: the sequence length is
  part of the graph.
- To process a stream, make the state a model input and output
  (`forward(self, x, state)` returning `(y, state)`), and copy the
  output state into the input before the next call. The `gru` model of
  the example does this.
- litert-torch 0.10.0's quantizer fails on an `nn.GRU` called without an
  initial state, or whose final state is unused (issue #32). The
  streaming form avoids both.

TensorFlow's converter turns Keras recurrent layers into
`UNIDIRECTIONAL_SEQUENCE_LSTM` or `SVDF` operators with variable tensors
instead. nanoglia does not support those yet (issue #22).

## 3. Check that nanoglia runs it

Load the model with the features your firmware will use, and list the
kernels:

```rust
let graph = Graph::load(&[&MODEL.0], Encoding::Autodetect, Target::Auto)?;
for i in 0..graph.operator_count() {
    println!("{i}: {}", graph.kernel_name(i).unwrap_or("?"));
}
let input = graph.input(0)?;
println!("input {:?} {:?}", input.dtype, input.dims);
```

- If loading fails with `Detail::UnsupportedOp { op_index, opcode }`,
  operator `op_index` (TFLite builtin code `opcode`) is not supported
  with its types or parameters: compare with
  [Supported operators](../reference/supported-operators.md), and change
  the model or the export.
- If the input is `F32` while the kernels are int8 ones, the model
  converts its input with a `QUANTIZE` operator (and usually its output
  with a `DEQUANTIZE`). nanoglia runs both, so you can feed floats.
- `rust/...` kernels on convolutions or fully-connected layers mean
  they stayed in float: the quantization did not cover them.

## 4. Check the output against LiteRT

nanoglia's outputs match LiteRT's reference kernels bit for bit
([Bit-exactness](../explanation/bit-exactness.md)). Compare them on a
few inputs. With the LiteRT interpreter (package `ai-edge-litert`, which
runs on more systems than the converter), write an input and the
expected output:

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

interpreter = Interpreter(
    model_path="model.tflite",
    experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# A random input over the whole int8 range.
info = np.iinfo(input_details["dtype"])
x = np.random.default_rng(0).integers(
    info.min, info.max, size=input_details["shape"], dtype=input_details["dtype"], endpoint=True
)
interpreter.set_tensor(input_details["index"], x)
interpreter.invoke()

x.tofile("input.bin")
interpreter.get_tensor(output_details["index"]).tofile("expected.bin")
```

`OpResolverType.BUILTIN_REF` selects LiteRT's reference kernels, which
nanoglia's tests use as their oracle; LiteRT's default optimized kernels
may round differently. For a float input, draw floats instead of
integers.

Then run the same input with nanoglia and compare the bytes:

```rust
let graph = Graph::load(&[&MODEL.0], Encoding::Autodetect, Target::Auto)?;
let mut ctx = graph.init_execution_context()?;
ctx.set_input(0, &std::fs::read("input.bin")?)?;
ctx.compute()?;
let expected = std::fs::read("expected.bin")?;
assert_eq!(ctx.output(0)?.bytes(), &expected[..]);
```

`set_input` takes the raw little-endian bytes of the tensor, which is
what `tofile` writes on x86 and Arm computers. A model with several
fully-connected layers in a row may differ by one or two units in the
last ones: LiteRT's reference fully-connected kernel requantizes in
floating point, nanoglia in fixed point like TensorFlow Lite Micro
(issue #1).

## From an ONNX model

To run a model you only have in ONNX, convert it with
[onnx2tf](https://github.com/PINTO0309/onnx2tf), which writes LiteRT
models and transposes them to the channels-last layout on the way:

```sh
pip install -U onnx2tf
onnx2tf -i model.onnx -oiqt -qt per-channel -cind input calibration.npy "[[[[0.485,0.456,0.406]]]]" "[[[[0.229,0.224,0.225]]]]"
```

- `-oiqt` writes the integer-quantized models, in the output folder
  (`saved_model/` by default);
- `-qt per-channel` (the default) quantizes the weights per channel;
- `-cind` gives, for each model input, its name, a NumPy file of
  calibration data and the mean and standard deviation to normalize it
  with (the values above are only an example). Its data must be in the
  channels-last order of the converted model;
- `-iqd` and `-oqd` choose the input and output types of the full
  integer model: `int8` (the default), `uint8` or `float32`.

The default backend of onnx2tf writes LiteRT files directly and
covers a limited set of integer quantizations; if it refuses your model,
install `onnx2tf[tensorflow]` and add `-tb tf_converter` to use
TensorFlow's converter. Pick the full integer model among the files
written, then check it with steps 3 and 4. onnx2tf's README documents every option and the
int16-activation variants; for 16x8 quantization with TensorFlow's own
converter, see LiteRT's
[16x8 quantization guide](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant_16x8)
(`tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8`).

## From a Keras model

TensorFlow's converter writes full-integer models directly, with an int8
input and output:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset  # yields [input] lists of real samples
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
model_bytes = converter.convert()
```

As LiteRT's
[post-training integer quantization guide](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant)
explains, `TFLITE_BUILTINS_INT8` makes the conversion fail rather than
keep a float operator, and the two `inference_*_type` lines give the
int8 interface. Check the result with steps 3 and 4.
[`examples/train-to-device/training/train_keras.py`](../../examples/train-to-device/training/train_keras.py)
is a complete script (TensorFlow 2.21.0, Keras 3.15.1), which
[From training to the device](../tutorials/train-and-deploy.md) walks
through. Keras recurrent layers are not supported yet (see
[Recurrent layers](#recurrent-layers)).
