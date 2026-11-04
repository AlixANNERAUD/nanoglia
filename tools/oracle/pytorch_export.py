"""Export small PyTorch models to fully int8 LiteRT models and write golden cases.

The models are realistic small networks:

- `smallnet`, an image classifier: conv / batch norm / ReLU blocks, a channel
  shuffle (`torch.split` and `torch.cat`), max pooling, a `permute` and a
  slice, a flatten and a linear head;
- `gru` and `lstm`, sequence classifiers (keyword spotting or IMU gestures
  shape: 8 frames of 10 features): one `nn.GRU` or `nn.LSTM` layer whose last
  output feeds a linear head.

Each is quantized with PyTorch's PT2E flow and litert-torch's quantizer
(static int8: per-tensor asymmetric activations calibrated on a fixed random
set, per-channel symmetric weights), converted with `litert_torch.convert`,
then run with the LiteRT reference kernels (`BUILTIN_REF`) on fixed inputs.
Each model gets a directory `<name>/` in the output directory, with:

- `model.tflite`, the converted model;
- `sample_<k>/`, golden cases in the format of `testdata/` (see its README), which
  `crates/nanoglia/tests/golden.rs` runs when `NANOGLIA_GOLDEN_DIR` points at
  the output directory.

It also prints the operators of each model with their tensor types, so a
failure to load in nanoglia can be traced to an operator.

litert-torch depends on litert-converter, which ships x86_64 Linux and macOS
arm64 wheels only: run it there, with the optional `torch` dependency group:

    uv run --group torch python -I pytorch_export.py OUT_DIR

or `mise run pytorch-export` from the repository root, which also runs the
case with nanoglia. Everything is seeded; the converted model may still change
with the versions of torch and litert-torch, so it is not committed.
"""

from __future__ import annotations

import argparse
import collections
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

import litert_torch
from litert_torch.quantize.pt2e_quantizer import (
    PT2EQuantizer,
    get_symmetric_quantization_config,
)
from litert_torch.quantize.quant_config import QuantConfig
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import REPO, make_interpreter, write_case  # noqa: E402

CALIBRATION_SAMPLES = 64
GOLDEN_SAMPLES = 3


class SmallNet(nn.Module):
    """A small CNN with the data-movement operators PyTorch exports produce."""

    def __init__(self, classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(2 * 4 * 16, classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))  # [1, 8, 16, 16]
        a, b = torch.split(x, [3, 5], dim=1)  # channel shuffle: SPLIT_V...
        x = torch.cat([b, a], dim=1)  # ... and CONCATENATION
        x = torch.relu(self.bn2(self.conv2(x)))  # [1, 16, 8, 8]
        x = self.pool(x)  # [1, 16, 4, 4]
        x = x.permute(0, 2, 3, 1)  # [1, 4, 4, 16]: TRANSPOSE
        x = x[:, 1:3]  # [1, 2, 4, 16]: SLICE or STRIDED_SLICE
        x = torch.flatten(x, 1)  # [1, 128]: RESHAPE
        return self.fc(x)  # FULLY_CONNECTED


class RecurrentNet(nn.Module):
    """A sequence classifier: one recurrent layer, then a linear head on its last output."""

    def __init__(self, rnn: type[nn.Module], features: int = 10, hidden: int = 16,
                 classes: int = 4):
        super().__init__()
        self.rnn = rnn(features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, classes)
        self.hidden = hidden
        self.lstm = rnn is nn.LSTM

    def forward(self, x):  # [1, frames, features]
        # An explicit initial state: litert-torch's GRU annotator reads it (#22).
        h0 = torch.zeros(1, x.shape[0], self.hidden)
        out, _ = self.rnn(x, (h0, h0) if self.lstm else h0)  # [1, frames, hidden]
        return self.fc(out[:, -1])  # [1, classes]


def build_smallnet() -> nn.Module:
    model = SmallNet()
    # Batch norm statistics as a trained model would have them.
    with torch.no_grad():
        for bn in (model.bn1, model.bn2):
            bn.running_mean.uniform_(-0.3, 0.3)
            bn.running_var.uniform_(0.5, 2.0)
            bn.weight.uniform_(0.5, 1.5)
            bn.bias.uniform_(-0.2, 0.2)
    return model


# Name -> (builder, input shape).
MODELS = {
    "smallnet": (build_smallnet, (1, 3, 16, 16)),
    "gru": (lambda: RecurrentNet(nn.GRU), (1, 8, 10)),
    "lstm": (lambda: RecurrentNet(nn.LSTM), (1, 8, 10)),
}


def export(name: str, out_dir: Path) -> Path:
    build, input_shape = MODELS[name]
    torch.manual_seed(0)
    model = build().eval()
    sample_args = (torch.randn(input_shape, generator=torch.Generator().manual_seed(1)),)

    # Static int8 quantization (`is_dynamic=False`, the default) with per-channel weights.
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True)
    )
    exported = torch.export.export(model, sample_args).module()
    prepared = prepare_pt2e(exported, quantizer)
    calibration = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for _ in range(CALIBRATION_SAMPLES):
            prepared(torch.randn(input_shape, generator=calibration))
    quantized = convert_pt2e(prepared, fold_quantize=False)

    edge_model = litert_torch.convert(
        quantized, sample_args, quant_config=QuantConfig(pt2e_quantizer=quantizer)
    )
    model_path = out_dir / "model.tflite"
    edge_model.export(str(model_path))
    return model_path


def report(model_path: Path) -> None:
    """Print the operators of the model with the types of their tensors."""
    it = make_interpreter(model_path=model_path)
    tensors = {d["index"]: d for d in it.get_tensor_details()}

    def types(indices):
        return ", ".join(np.dtype(tensors[i]["dtype"]).name for i in indices if i >= 0)

    counts = collections.Counter()
    float_ops = []
    print(f"operators of {model_path}:")
    for op in it._get_ops_details():  # noqa: SLF001 (no public equivalent)
        name = op["op_name"]
        counts[name] += 1
        print(f"  {op['index']:3} {name:24} ({types(op['inputs'])}) -> ({types(op['outputs'])})")
        computes_in_float = any(
            tensors[i]["dtype"] == np.float32 for i in list(op["inputs"]) + list(op["outputs"])
            if i >= 0
        )
        if computes_in_float and name not in ("QUANTIZE", "DEQUANTIZE"):
            float_ops.append(f"{op['index']} {name}")
    print("operator counts: " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
    if float_ops:
        print("not fully int8, float operators: " + ", ".join(float_ops))
    else:
        print("fully int8 (float only at the QUANTIZE input and DEQUANTIZE output)")


def write_model(name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True)
    model_path = export(name, out_dir)
    report(model_path)

    details = make_interpreter(model_path=model_path).get_input_details()
    rng = np.random.default_rng(3)
    for sample in range(GOLDEN_SAMPLES):
        inputs = []
        for d in details:
            if np.issubdtype(d["dtype"], np.floating):
                inputs.append(rng.standard_normal(d["shape"]).astype(d["dtype"]))
            else:
                info = np.iinfo(d["dtype"])
                inputs.append(rng.integers(info.min, info.max, size=d["shape"],
                                           dtype=d["dtype"], endpoint=True))
        # Bit-exact, float outputs included: they dequantize the int8 result.
        write_case(out_dir / f"sample_{sample}", model_path, inputs, 0)
    print(f"{GOLDEN_SAMPLES} golden cases in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", nargs="?", type=Path,
                        default=REPO / "target" / "pytorch-export",
                        help="output directory (default: target/pytorch-export)")
    parser.add_argument("--model", action="append", choices=list(MODELS),
                        help="model to export, repeatable (default: all)")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for name in args.model or MODELS:
        print(f"== {name}")
        write_model(name, out_dir / name)


if __name__ == "__main__":
    main()
