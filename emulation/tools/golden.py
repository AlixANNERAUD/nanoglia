"""The comparison of crates/nanoglia/tests/golden.rs, kept in step with it: same tolerance rules
(testdata/README.md), same pending list (testdata/pending.json), same kernel expectations.

Shared by the runs that execute the golden cases somewhere else than on the host: run.py (the
test-runner firmware under Unicorn) and hardware/tools (a board, or its emulated runner). A result
is anything with the attributes `status` (`DONE` when the case ran to the end), `message`,
`outputs` (bytes of each output), `kernels` (the kernel of each operator) and `kernels_truncated`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# Status of a result whose case ran to the end (`Status::Done` of emulation/src/main.rs).
DONE = 1

DTYPES = {
    "int8": "<i1",
    "uint8": "<u1",
    "bool": "<u1",
    "int16": "<i2",
    "int32": "<i4",
    "uint32": "<u4",
    "int64": "<i8",
    "float32": "<f4",
}


def max_diff(dtype: str, actual: bytes, expected: bytes) -> float:
    """`max_diff` of golden.rs: the largest difference, in elements of `dtype`."""
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype {dtype}")
    np_dtype = np.dtype(DTYPES[dtype])
    n = min(len(actual), len(expected)) // np_dtype.itemsize * np_dtype.itemsize
    a = np.frombuffer(actual[:n], np_dtype)
    e = np.frombuffer(expected[:n], np_dtype)
    if dtype == "float32":
        # As golden.rs: equal values (so equal infinities) and two NaN match; otherwise an infinity
        # or a NaN on either side never passes; finite values compare in f32, by
        # |a - e| / max(|e|, 1).
        same = (a == e) | (np.isnan(a) & np.isnan(e))
        finite = np.isfinite(a) & np.isfinite(e)
        if not np.all(same | finite):
            return float("inf")
        with np.errstate(invalid="ignore", over="ignore"):
            d = np.abs(a - e) / np.maximum(np.abs(e), np.float32(1.0))
        return float(np.max(np.where(same, np.float32(0.0), d).astype(np.float64), initial=0.0))
    if dtype == "int64":
        return float(max((abs(int(x) - int(y)) for x, y in zip(a, e)), default=0))
    return float(np.max(np.abs(a.astype(np.int64) - e.astype(np.int64)), initial=0))


def expected_kernel(name: str) -> str | None:
    """`expected_kernel` of golden.rs, with the firmware's features (cmsis-nn and int16 on)."""
    if not name.startswith("ops/") or not name.endswith("_int16"):
        return None
    case = name.removeprefix("ops/")
    kernels = [
        ("conv2d_", "cmsis-nn/conv2d_s16"),
        ("depthwise_conv2d_", "cmsis-nn/depthwise_conv2d_s16"),
        ("average_pool2d_", "cmsis-nn/average_pool2d_s16"),
        ("max_pool2d_", "cmsis-nn/max_pool2d_s16"),
        ("logistic_", "rust/logistic_i16"),
        ("tanh_", "rust/tanh_i16"),
    ]
    return next((k for prefix, k in kernels if case.startswith(prefix)), None)


@dataclass
class Failure:
    message: str
    # Whether only the values of the outputs differ: the case ran to the end, every other check
    # passed. The only failure a pending case may have.
    comparison: bool = False


def check(name: str, spec: dict, result, expected: list[bytes]) -> Failure | None:
    """The error golden.rs would report for this case, or None if it passes."""
    if result.status != DONE:
        return Failure(result.message)
    if result.kernels_truncated:
        return Failure("the kernel names do not fit in the report (`Report::kernels`)")
    if (kernel := expected_kernel(name)) is not None:
        for i, actual in enumerate(result.kernels):
            if actual != kernel:
                return Failure(f"operator {i} runs {actual!r}, expected {kernel}")
    tolerance = float(spec.get("tolerance", 0.0))
    for i, (output, want) in enumerate(zip(spec["outputs"], expected)):
        if i >= len(result.outputs):
            return Failure(f"output {i}: no such output")
        actual = result.outputs[i]
        if len(actual) != len(want):
            return Failure(f"output {i}: {len(actual)} bytes, expected {len(want)}")
        diff = max_diff(output["dtype"], actual, want)
        if diff > tolerance:
            return Failure(f"output {i}: max diff {diff} > tolerance {tolerance}", comparison=True)
    return None


def corrupt(dtype: str, data: bytes, tolerance: float) -> bytes | None:
    """`data` with one element moved beyond `tolerance` by the rules of `max_diff`, or None if it
    has no element that can be (empty, or only NaN and infinite floats, which the self-check leaves
    alone)."""
    np_dtype = np.dtype(DTYPES[dtype])
    a = np.frombuffer(data[: len(data) // np_dtype.itemsize * np_dtype.itemsize], np_dtype).copy()
    if dtype == "float32":
        finite = np.flatnonzero(np.isfinite(a))
        # |a - e| / max(|e|, 1) is |e| >= 1 when e goes from |e| >= 1 to 0, and over 1/3 when e
        # moves by 0.5 from |e| < 1: far beyond the float tolerance (1e-5).
        if not finite.size or tolerance >= 1 / 3:
            return None
        i = finite[0]
        a[i] = 0.0 if abs(a[i]) >= 1 else a[i] + np.float32(0.5)
    else:
        if not a.size:
            return None
        delta = int(tolerance) + 1
        info = np.iinfo(np_dtype)
        value = int(a[0])
        a[0] = value + delta if value + delta <= info.max else value - delta
    return a.tobytes() + data[a.nbytes :]


def self_check(canary: tuple[str, dict, object, list[bytes]]) -> bool:
    """Whether a corrupted expected output of `canary`, a passing case, fails the comparison."""
    name, spec, result, expected = canary
    tolerance = float(spec.get("tolerance", 0.0))
    for i, output in enumerate(spec["outputs"]):
        corrupted = corrupt(output["dtype"], expected[i], tolerance)
        if corrupted is None:
            continue
        if check(name, spec, result, [*expected[:i], corrupted, *expected[i + 1 :]]) is None:
            print(f"FAIL     self-check: a corrupted output {i} of {name} passes")
            return False
        print(f"self-check: a corrupted output {i} of {name} fails, as it must")
        return True
    print(f"FAIL     self-check: no output of {name} can be corrupted")
    return False


def load_pending() -> dict[str, str]:
    """testdata/pending.json, the list golden.rs reads too."""
    entries = json.loads((REPO / "testdata" / "pending.json").read_text())
    return {e["case"]: e["reason"] for e in entries}


def find_cases(root: Path, filter: str = "") -> list[tuple[str, Path]]:
    """The name and the directory of every case of `root` whose name contains `filter`, in the
    order golden.rs runs them."""
    cases = sorted(p.parent for p in root.rglob("case.json"))
    names = ((directory.relative_to(root).as_posix(), directory) for directory in cases)
    return [(name, directory) for name, directory in names if filter in name]
