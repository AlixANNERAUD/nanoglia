"""What the two training scripts share: the data, the LiteRT evaluation, the test vectors.

The task is a gesture classifier on accelerometer windows: 50 samples at 25 Hz
(two seconds) of the three axes, in g, classified as `idle`, `shake` or
`circle`. The data is synthetic and seeded, so nothing is downloaded and every
run draws the same windows.

`train_pytorch.py` and `train_keras.py` each train the same small network, turn
it into a full-integer int8 `.tflite` model, then call `export()`, which checks
the model with LiteRT's reference kernels and writes, into the output
directory:

- `model.tflite`;
- `sample_<k>/`, test vectors in the golden-case format of `testdata/` (see its
  README): `input_0.bin` (the int8 input), `output_0.bin` (the int8 output
  LiteRT computes), `case.json`, plus `window.bin`, the float32 window before
  quantization. Sample `k` is a window of class `k % 3`.
"""

from __future__ import annotations

import collections
import json
import shutil
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

SAMPLES = 50  # per window
RATE_HZ = 25.0
AXES = 3
LABELS = ("idle", "shake", "circle")

TRAINING_WINDOWS = 1500
TEST_WINDOWS = 600
CALIBRATION_WINDOWS = 200
TEST_VECTORS = 6
EPOCHS = 12
BATCH = 32
# A lower accuracy, float or int8, means a broken training or conversion.
MINIMUM_ACCURACY = 0.9

# The nanoglia repository, when the script runs from a checkout of it.
REPOSITORY = Path(__file__).resolve().parents[3]


def make_windows(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw `count` windows: float32 `[count, SAMPLES, AXES]` in g, and their classes.

    Window `i` has class `i % 3`. Every window holds gravity, seen by a board
    lying roughly flat, a slow drift of the hand and sensor noise; a shake adds
    a fast oscillation along one direction, a circle a slower rotation in the
    horizontal plane. Weak gestures come close to the noise, so that a few
    windows are hard to tell apart.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(SAMPLES) / RATE_HZ
    labels = np.arange(count) % len(LABELS)
    windows = np.empty((count, SAMPLES, AXES), np.float32)
    for i, label in enumerate(labels):
        gravity = np.array([rng.normal(0.0, 0.15), rng.normal(0.0, 0.15), 1.0])
        signal = np.tile(gravity / np.linalg.norm(gravity), (SAMPLES, 1))
        drift = np.sin(2.0 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0.0, 2.0 * np.pi))
        signal += rng.uniform(0.0, 0.15) * drift[:, None] * rng.normal(0.0, 1.0, AXES)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        if LABELS[label] == "shake":
            direction = rng.normal(0.0, 1.0, AXES)
            wave = np.sin(2.0 * np.pi * rng.uniform(3.0, 6.0) * t + phase)
            signal += rng.uniform(0.2, 1.2) * wave[:, None] * direction / np.linalg.norm(direction)
        elif LABELS[label] == "circle":
            turn = rng.choice([-1.0, 1.0]) * 2.0 * np.pi * rng.uniform(0.8, 2.0) * t + phase
            radius = rng.uniform(0.15, 0.7)
            signal[:, 0] += radius * np.cos(turn)
            signal[:, 1] += radius * np.sin(turn)
        signal += rng.normal(0.0, 0.12, signal.shape)
        windows[i] = signal
    return windows, labels


def datasets() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """The training and test sets, the same for both frameworks."""
    return make_windows(TRAINING_WINDOWS, seed=1), make_windows(TEST_WINDOWS, seed=2)


def make_interpreter(model_path: Path) -> Interpreter:
    """LiteRT with its reference kernels, which nanoglia matches bit for bit."""
    interpreter = Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    return interpreter


def quantize(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """`round(x / scale) + zero_point` saturated to int8, computed in float32.

    Halves round away from zero, as `f32::round` in Rust and `roundf` in C do
    (`np.round` rounds them to even), so that a device quantizing the same
    window gets the same bytes.
    """
    scaled = values.astype(np.float32) / np.float32(scale)
    rounded = np.trunc(scaled)
    rounded += np.where(np.abs(scaled - rounded) >= 0.5, np.sign(scaled), 0.0)
    return np.clip(rounded.astype(np.int64) + zero_point, -128, 127).astype(np.int8)


def run(interpreter: Interpreter, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize one float window, run the model: the int8 input and the int8 output."""
    details = interpreter.get_input_details()[0]
    scale, zero_point = details["quantization"]
    quantized = quantize(window, scale, zero_point).reshape(details["shape"])
    interpreter.set_tensor(details["index"], quantized)
    interpreter.invoke()
    return quantized, interpreter.get_tensor(interpreter.get_output_details()[0]["index"]).copy()


def tensor_json(details: dict) -> dict:
    scale, zero_point = details["quantization"]
    return {
        "name": details["name"],
        "dtype": np.dtype(details["dtype"]).name,
        "shape": [int(d) for d in details["shape"]],
        "scale": float(scale),
        "zero_point": int(zero_point),
    }


def report(interpreter: Interpreter) -> list[str]:
    """Print the operators with the types of their tensors; return those computing in float."""
    tensors = {d["index"]: d for d in interpreter.get_tensor_details()}

    def types(indices):
        return ", ".join(np.dtype(tensors[i]["dtype"]).name for i in indices if i >= 0)

    counts = collections.Counter()
    float_operators = []
    for op in interpreter._get_ops_details():  # noqa: SLF001 (no public equivalent)
        name = op["op_name"]
        counts[name] += 1
        print(f"  {op['index']:2} {name:16} ({types(op['inputs'])}) -> ({types(op['outputs'])})")
        used = [i for i in list(op["inputs"]) + list(op["outputs"]) if i >= 0]
        if any(tensors[i]["dtype"] == np.float32 for i in used):
            float_operators.append(f"{op['index']} {name}")
    print("operator counts: " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
    return float_operators


def export(out_dir: Path, model: bytes, float_accuracy: float) -> None:
    """Write the model, check it with LiteRT on the test set, write the test vectors."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    model_path = out_dir / "model.tflite"
    model_path.write_bytes(model)
    print(f"{model_path}: {len(model)} bytes")

    interpreter = make_interpreter(model_path)
    float_operators = report(interpreter)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    for name, details in (("input", input_details), ("output", output_details)):
        scale, zero_point = details["quantization"]
        print(f"{name}: {np.dtype(details['dtype']).name} {[int(d) for d in details['shape']]}, "
              f"scale {scale:.8f}, zero point {zero_point}")
    if float_operators or input_details["dtype"] != np.int8 or output_details["dtype"] != np.int8:
        raise SystemExit("not a full-integer int8 model, float operators: "
                         + (", ".join(float_operators) or "none"))

    _, (windows, labels) = datasets()
    correct = sum(int(np.argmax(run(interpreter, w)[1]) == y) for w, y in zip(windows, labels))
    int8_accuracy = correct / len(labels)
    print(f"float accuracy: {float_accuracy:.4f}")
    print(f"int8 accuracy (LiteRT reference kernels): {int8_accuracy:.4f}")

    in_repository = model_path.is_relative_to(REPOSITORY) and (REPOSITORY / "testdata").is_dir()
    case = {
        # Relative to the nanoglia repository when inside it, as its golden test expects.
        "model": (model_path.relative_to(REPOSITORY) if in_repository else model_path).as_posix(),
        "inputs": [tensor_json(input_details)],
        "outputs": [tensor_json(output_details)],
        "tolerance": 0,
    }
    scale, zero_point = output_details["quantization"]
    for k in range(TEST_VECTORS):
        quantized, output = run(interpreter, windows[k])
        case_dir = out_dir / f"sample_{k}"
        case_dir.mkdir()
        (case_dir / "window.bin").write_bytes(windows[k].astype("<f4").tobytes())
        (case_dir / "input_0.bin").write_bytes(quantized.tobytes())
        (case_dir / "output_0.bin").write_bytes(output.tobytes())
        (case_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        best = int(np.argmax(output))
        score = scale * (int(output.flat[best]) - zero_point)
        print(f"sample_{k}: {LABELS[labels[k]]:6} -> {output.flatten().tolist()} "
              f"{LABELS[best]} {score:.3f}")
    print(f"{TEST_VECTORS} test vectors in {out_dir}")

    if min(float_accuracy, int8_accuracy) < MINIMUM_ACCURACY:
        raise SystemExit(f"accuracy below {MINIMUM_ACCURACY}")
