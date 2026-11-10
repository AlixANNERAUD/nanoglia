"""Train the gesture classifier with PyTorch and export it to a full-integer int8 LiteRT model.

    uv run --project examples/train-to-device/training --group pytorch \
        python examples/train-to-device/training/train_pytorch.py target/train-to-device/pytorch

Trains the network on the synthetic windows of `gesture.py`, quantizes it with
PyTorch's PT2E flow and litert-torch's quantizer (static int8, calibrated on
training windows), converts it with `litert_torch.convert`, gives it an int8
interface (`strip_float_interface`), then checks the `.tflite` model with
LiteRT and writes it with its test vectors (see `gesture.export`).

litert-torch depends on litert-converter, which ships x86_64 Linux and macOS
arm64 wheels only: run it there.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import flatbuffers
import torch
from ai_edge_litert import schema_py_generated as schema
from torch import nn

import litert_torch
from litert_torch.quantize.pt2e_quantizer import (
    PT2EQuantizer,
    get_symmetric_quantization_config,
)
from litert_torch.quantize.quant_config import QuantConfig
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

import gesture


class GestureNet(nn.Module):
    """Two convolutions over time, a global average, a linear layer: 835 parameters."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (5, 3))  # mixes the axes: [8, 46, 1]
        self.pool = nn.MaxPool2d((2, 1))  # [8, 23, 1]
        self.conv2 = nn.Conv2d(8, 16, (5, 1))  # [16, 19, 1]
        self.average = nn.AdaptiveAvgPool2d(1)  # [16, 1, 1]
        self.linear = nn.Linear(16, len(gesture.LABELS))

    def logits(self, x):  # 50 samples of 3 axes, as one channel: [1, 1, 50, 3]
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.average(torch.relu(self.conv2(x)))
        return self.linear(torch.flatten(x, 1))

    def forward(self, x):
        # Workaround: `torch.softmax` exports as `aten.softmax.int`, which litert-torch 0.10.0's
        # quantizer does not annotate: the softmax would stay in float32 behind a DEQUANTIZE.
        # It annotates `aten._softmax`, the form softmax decomposes into, which this calls.
        return torch.ops.aten._softmax(self.logits(x), 1, False)


def train(model: GestureNet, windows: torch.Tensor, labels: torch.Tensor) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(gesture.EPOCHS):
        order = torch.randperm(len(windows))
        total = 0.0
        for start in range(0, len(order), gesture.BATCH):
            batch = order[start : start + gesture.BATCH]
            optimizer.zero_grad()
            loss = loss_function(model.logits(windows[batch]), labels[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        print(f"epoch {epoch + 1}/{gesture.EPOCHS}: loss {total / len(order):.4f}")
    model.eval()


def strip_float_interface(model: bytes) -> bytes:
    """Give a quantized model with a float interface an int8 one.

    litert-torch keeps the float input and output of the PyTorch model: the
    converted model starts with a QUANTIZE operator and ends with a DEQUANTIZE.
    This removes both and makes their int8 tensors the model's input and
    output, as `inference_input_type = tf.int8` does in TensorFlow's converter.

    A workaround for litert-torch 0.10.0, which has no such option: remove it
    when `litert_torch.convert` can write an int8 interface. It expects the
    float input to feed the QUANTIZE operator only, and leaves the float
    tensors behind, unused.
    """
    tree = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(model, 0))
    subgraph = tree.subgraphs[0]

    def builtin(operator) -> int:
        code = tree.operatorCodes[operator.opcodeIndex]
        return max(code.builtinCode, code.deprecatedBuiltinCode)

    renamed = {}  # float tensor -> the int8 tensor that replaces it
    kept = []
    for operator in subgraph.operators:
        code = builtin(operator)
        if code == schema.BuiltinOperator.QUANTIZE and operator.inputs[0] in subgraph.inputs:
            renamed[int(operator.inputs[0])] = int(operator.outputs[0])
        elif code == schema.BuiltinOperator.DEQUANTIZE and operator.outputs[0] in subgraph.outputs:
            renamed[int(operator.outputs[0])] = int(operator.inputs[0])
        else:
            kept.append(operator)
    subgraph.operators = kept
    subgraph.inputs = [renamed.get(int(i), int(i)) for i in subgraph.inputs]
    subgraph.outputs = [renamed.get(int(i), int(i)) for i in subgraph.outputs]
    for signature in tree.signatureDefs or []:
        for tensor in list(signature.inputs) + list(signature.outputs):
            tensor.tensorIndex = renamed.get(tensor.tensorIndex, tensor.tensorIndex)

    builder = flatbuffers.Builder(len(model))
    builder.Finish(tree.Pack(builder), b"TFL3")
    return bytes(builder.Output())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, help="output directory (deleted first)")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)
    (train_x, train_y), (test_x, test_y) = gesture.datasets()
    # One channel, first: [N, 1, 50, 3].
    train_x, test_x = torch.from_numpy(train_x)[:, None], torch.from_numpy(test_x)[:, None]

    model = GestureNet()
    train(model, train_x, torch.from_numpy(train_y))
    with torch.no_grad():
        predictions = model(test_x).argmax(dim=1)
    float_accuracy = (predictions == torch.from_numpy(test_y)).float().mean().item()

    # Static int8 quantization (`is_dynamic=False`, the default) with per-channel weights.
    quantizer = PT2EQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    sample_args = (train_x[:1],)
    prepared = prepare_pt2e(torch.export.export(model, sample_args).module(), quantizer)
    with torch.no_grad():
        for i in range(gesture.CALIBRATION_WINDOWS):
            prepared(train_x[i : i + 1])
    quantized = convert_pt2e(prepared, fold_quantize=False)
    edge_model = litert_torch.convert(
        quantized, sample_args, quant_config=QuantConfig(pt2e_quantizer=quantizer)
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.tflite"
        edge_model.export(str(path))
        converted = path.read_bytes()

    gesture.export(args.out_dir.resolve(), strip_float_interface(converted), float_accuracy)


if __name__ == "__main__":
    main()
