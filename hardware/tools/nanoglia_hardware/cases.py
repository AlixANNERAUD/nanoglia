"""The run of the golden cases of testdata/ on a device, compared as
crates/nanoglia/tests/golden.rs and emulation/tools/run.py compare them (the comparison itself is
emulation/tools/golden.py): same cases, same tolerances, same pending list, same kernel checks.

Two things are measured, and kept apart:

- correctness, for every case: the outputs against those LiteRT computed;
- time, only for the cases whose model is linked into the firmware and runs in place from flash,
  as in an application (the MLPerf Tiny models). A model received into RAM (the single-operator
  cases) has its cycles recorded too, marked as no performance figure: a kernel that reads its
  weights from RAM is not timed as it would be in an application.
"""

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from . import protocol as p
from .board import SetupError, Unreachable
from .shared import REPO, golden

# The cases whose model must run from flash: the realistic models.
FLASH_CASES = "mlperf-tiny/"
# Seconds a run may take before the device is given up: more than the watchdog of the firmware.
RUN_TIMEOUT_S = 30.0
# Cases lost in a row (the device stopped answering) after which the run stops.
MAX_LOST_IN_A_ROW = 3
RAM_NOTE = "model in RAM, not a performance figure"


def align(value: int, to: int = 16) -> int:
    return (value + to - 1) // to * to


@dataclass
class CaseResult:
    name: str
    # "ok", "pending", "skipped", or a failure: "FAIL", "FIXED".
    status: str
    message: str = ""
    model: str = ""
    model_bytes: int = 0
    # "flash" (linked into the firmware) or "ram" (received over the link).
    model_location: str = ""
    image: str = ""
    # Whether the cycles are a performance figure: model in flash, and a counter that is a clock.
    timing: bool = False
    timing_note: str = ""
    operators: list[dict] = field(default_factory=list)
    load_cycles: int | None = None
    context_cycles: int | None = None
    # Cycles of the inference: the fastest of the repeats, then each of them.
    compute_cycles: int | None = None
    compute_cycles_runs: list[int] = field(default_factory=list)
    compute_ms: float | None = None
    graph_bytes: int | None = None
    context_bytes: int | None = None
    arena_used: int | None = None
    arena_available: int | None = None
    inputs_bytes: int | None = None
    stack_peak: int | None = None
    # For the run only, not in the results file: whether the failure is one of the comparison, and
    # what the self-check needs of a passing case.
    comparison: bool = field(default=False, repr=False)
    ran: tuple | None = field(default=None, repr=False)

    @property
    def failed(self) -> bool:
        return self.status in ("FAIL", "FIXED")


class Lost(Exception):
    """The device stopped answering during a case."""


@dataclass
class Session:
    """A device, the firmware on it and what the host knows of its RAM store."""

    board: object
    device: p.Device
    info: p.Info
    image: str
    # Index of each embedded model whose bytes were checked, by the path of its file.
    embedded: dict[str, int] = field(default_factory=dict)
    # CRC of the model at the start of the RAM store, if the host put one there.
    ram_model: int | None = None

    def embedded_index(self, path: Path, model: bytes) -> int | None:
        """The index of `model` among the models linked into the firmware, checked by its CRC."""
        key = str(path)
        if key not in self.embedded:
            self.embedded[key] = -1
            for index, (name, length) in enumerate(self.info.models):
                if name == path.name and length == len(model):
                    if self.device.crc(p.EMBEDDED, index, length) == zlib.crc32(model):
                        self.embedded[key] = index
                    break
        return None if self.embedded[key] < 0 else self.embedded[key]


def run_case(session: Session, name: str, directory: Path, spec: dict, repeat: int, flash_only: bool) -> CaseResult | None:
    """Runs one case and compares its outputs. None if this firmware cannot run it but another
    image may (its model must run from flash and is not linked into this one)."""
    device, info = session.device, session.info
    model_path = REPO / spec["model"]
    model = model_path.read_bytes()
    inputs = [(directory / f"input_{i}.bin").read_bytes() for i in range(len(spec["inputs"]))]
    expected = [(directory / f"output_{i}.bin").read_bytes() for i in range(len(spec["outputs"]))]
    result = CaseResult(name, "FAIL", model=spec["model"], model_bytes=len(model), image=session.image)

    index = session.embedded_index(model_path, model)
    if index is None and flash_only:
        return None
    result.model_location = "ram" if index is None else "flash"

    # The RAM store: the model if it is not in flash, the inputs, then the arena up to the end.
    offset = 0 if index is not None else align(len(model))
    spans = []
    for data in inputs:
        spans.append((offset, len(data)))
        offset = align(offset + len(data))
    result.inputs_bytes = sum(len(data) for data in inputs)
    result.arena_available = info.ram_bytes - offset
    if offset >= info.ram_bytes:
        result.status = "skipped"
        result.message = f"does not fit: {offset} bytes of {'inputs' if index is not None else 'model and inputs'}, RAM store of {info.ram_bytes}"
        return result

    if index is None and session.ram_model != zlib.crc32(model):
        session.ram_model = None
        device.write(0, model)
        if device.crc(p.RAM, 0, len(model)) != zlib.crc32(model):
            result.message = "the model read back from the RAM store differs from the one sent"
            return result
        session.ram_model = zlib.crc32(model)
    for (at, _), data in zip(spans, inputs):
        device.write(at, data)
        if device.crc(p.RAM, at, len(data)) != zlib.crc32(data):
            result.message = "an input read back from the RAM store differs from the one sent"
            return result

    best = None
    for _ in range(repeat):
        if index is None:
            run = device.run(p.RAM, 0, len(model), offset, spans, RUN_TIMEOUT_S)
        else:
            run = device.run(p.EMBEDDED, index, len(model), offset, spans, RUN_TIMEOUT_S)
        if run.status != p.OK:
            stage = p.STAGES.get(run.status, f"status {run.status}")
            if run.error == p.TOO_LARGE and run.status in (p.LOAD, p.CONTEXT):
                result.status = "skipped"
                result.message = f"does not fit: {stage}: {run.message}, with an arena of {result.arena_available} bytes"
            else:
                result.message = f"{stage}: {run.message}"
            return result
        result.compute_cycles_runs.append(run.compute_cycles)
        if best is None or run.compute_cycles < best[0].compute_cycles:
            # The results stay on the device until the next command: read them now.
            operators = device.operators(run.operators)
            outputs = [device.output(i, length) for i, length in enumerate(run.output_lengths)]
            best = (run, operators, outputs)
    run, operators, outputs = best

    clock = info.counter_hz > 0
    result.timing = clock and index is not None
    result.timing_note = "" if result.timing else (RAM_NOTE if index is None else "the counter of the device is not a clock")
    to_ms = (lambda cycles: round(cycles * 1000 / info.counter_hz, 4)) if result.timing else (lambda cycles: None)
    result.operators = [
        {"index": i, "kernel": kernel, "cycles": cycles, "ms": to_ms(cycles)} for i, (cycles, kernel) in enumerate(operators)
    ]
    result.load_cycles, result.context_cycles = run.load_cycles, run.context_cycles
    result.compute_cycles, result.compute_ms = run.compute_cycles, to_ms(run.compute_cycles)
    result.graph_bytes, result.context_bytes = run.graph_bytes, run.context_bytes
    result.arena_used, result.stack_peak = run.arena_used, run.stack_peak

    ran = SimpleNamespace(
        status=golden.DONE,
        message="",
        outputs=outputs,
        kernels=[kernel for _, kernel in operators],
        kernels_truncated=False,
    )
    failure = golden.check(name, spec, ran, expected)
    result.comparison = failure is not None and failure.comparison
    result.ran = (spec, ran, expected)
    if failure is None:
        result.status, result.message = "ok", ""
    else:
        result.message = failure.message
    if run.cycles_truncated:
        result.timing_note = (result.timing_note + "; " if result.timing_note else "") + "cycles kept for the first operators only"
    return result


@dataclass
class Report:
    results: list[CaseResult] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    # 0: every case passed; 1: a case failed; 3: the board was lost.
    exit_code: int = 0

    def counts(self) -> dict[str, int]:
        counts = {"cases": len(self.results), "ok": 0, "pending": 0, "skipped": 0, "failed": 0}
        for result in self.results:
            counts["failed" if result.failed else result.status] += 1
        counts["timed"] = sum(result.timing for result in self.results)
        return counts


def describe(info: p.Info) -> dict:
    return {
        "firmware": info.firmware,
        "revision": info.revision,
        "nanoglia": info.nanoglia,
        "build": info.build,
        "board": info.board,
        "clock": info.clock,
        "counter_hz": info.counter_hz,
        "ram_store_bytes": info.ram_bytes,
        "models_in_flash": [name for name, _ in info.models],
    }


def run_cases(board, images: list, root: Path, filter: str = "", repeat: int = 1, log=print, pending: dict[str, str] | None = None) -> Report:
    """Runs the cases of `root` whose name contains `filter` on `board`, flashing each image of
    `images` in turn (none: whatever firmware is on the board). The single-operator cases run
    once, on the first image; a case of `FLASH_CASES` runs on the image that holds its model."""
    cases = golden.find_cases(root, filter)
    pending = golden.load_pending() if pending is None else pending
    report = Report()
    done: dict[str, CaseResult] = {}
    canaries = {}
    lost_in_a_row = 0

    def finish(result: CaseResult) -> None:
        nonlocal lost_in_a_row
        reason = pending.get(result.name)
        if result.status == "ok" and reason is not None:
            result.status, result.message = "FIXED", "remove it from testdata/pending.json"
        elif result.status == "FAIL" and reason is not None and result.comparison:
            result.status, result.message = "pending", f"{result.message} ({reason})"
        elif result.status == "FAIL" and reason is not None:
            result.message += " (pending, but it must fail on the comparison only)"
        elif result.status == "ok" and result.ran is not None:
            spec, ran, expected = result.ran
            exact = float(spec.get("tolerance", 0.0)) == 0 and all(o["dtype"] != "float32" for o in spec["outputs"])
            if any(expected):
                canaries.setdefault(exact, (result.name, spec, ran, expected))
        done[result.name] = result
        where = f"[{result.model_location}" + (f", {result.compute_ms:.3f} ms" if result.compute_ms is not None else "") + "]"
        log(f"{result.status:<8} {result.name} {where if result.model_location else ''}{': ' + result.message if result.message else ''}")

    try:
        for number, image in enumerate(images or [None]):
            last = number == len(images or [None]) - 1
            if image is not None:
                board.flash(image)
            device = board.open()
            info = device.info()
            label = getattr(getattr(image, "path", None), "name", None) or f"image {number + 1}"
            session = Session(board, device, info, label if image is not None else "as found on the board")
            report.images.append({"image": session.image, **describe(info)})
            log(f"{session.image}: {info.firmware} ({info.revision}), nanoglia {info.nanoglia}, {info.build}")
            log(f"  {info.board}; {info.clock}; RAM store of {info.ram_bytes} bytes; in flash: {', '.join(n for n, _ in info.models) or 'no model'}")
            if crash := info.describe_crash():
                log(f"  {crash}")
            for name, directory in cases:
                if name in done:
                    continue
                spec = json.loads((directory / "case.json").read_text())
                try:
                    try:
                        result = run_case(session, name, directory, spec, repeat, name.startswith(FLASH_CASES))
                    except (p.NoAnswer, p.Disconnected, p.ProtocolError) as error:
                        raise Lost(str(error)) from None
                    except p.DeviceError as error:
                        result = CaseResult(name, "FAIL", f"the device refused a request: {error}", image=session.image)
                    lost_in_a_row = 0
                except Lost as lost:
                    # The device stopped answering: say why if it comes back, and go on.
                    result = CaseResult(name, "FAIL", f"the device stopped answering ({lost})", image=session.image)
                    lost_in_a_row += 1
                    try:
                        session.device = board.recover()
                        session.info = session.device.info()
                        session.ram_model = None
                        if crash := session.info.describe_crash():
                            result.message += f"; {crash}"
                    except (p.NoAnswer, p.Disconnected, p.ProtocolError) as error:
                        finish(result)
                        raise Unreachable(f"the board came back but does not answer: {error}") from None
                    except Unreachable:
                        finish(result)
                        raise
                    if lost_in_a_row >= MAX_LOST_IN_A_ROW:
                        finish(result)
                        raise Unreachable(f"the device stopped answering on {lost_in_a_row} cases in a row") from None
                if result is None:
                    if not last:
                        continue
                    result = CaseResult(name, "skipped", "its model must run from flash and is in none of the firmware images", model=spec["model"], image=session.image)
                finish(result)
    except Unreachable as error:
        report.problems.append(str(error))
        report.exit_code = 3
        log(f"STOPPED  {error}")
    report.results = [done[name] for name, _ in cases if name in done]
    logged = len(report.problems)

    # An entry of pending.json that names no case hides nothing, but it is stale.
    if not filter and root == REPO / "testdata" and report.exit_code == 0:
        for stale in sorted(set(pending) - {name for name, _ in cases}):
            report.problems.append(f"{stale}: in testdata/pending.json, but there is no such case")
        if len(cases) <= 100:
            report.problems.append(f"test data not found: {len(cases)} cases in {root}")
    if not cases:
        report.problems.append(f"no case{f' matches --filter {filter!r}' if filter else ''} in {root}")

    # Negative check, as run.py: an expected output moved beyond the tolerance of a passing case
    # must fail the comparison, or the run proves nothing.
    canary = canaries.get(True) or canaries.get(False)
    if canary is not None:
        if not golden.self_check(canary):
            report.problems.append("self-check: a corrupted expected output passes the comparison")
    elif any(result.status == "ok" for result in report.results):
        report.problems.append("self-check: no passing case has a non-zero expected output")

    for problem in report.problems[logged:]:
        log(f"FAIL     {problem}")
    if report.exit_code == 0 and (report.problems or any(result.failed for result in report.results)):
        report.exit_code = 1
    return report


def results_json(report: Report, device: str, date: str, tool_revision: str) -> dict:
    """The results file."""
    cases = []
    for result in report.results:
        case = {key: value for key, value in vars(result).items() if key not in ("ran", "comparison")}
        cases.append(case)
    return {
        "format": 1,
        "date": date,
        "host_tool_revision": tool_revision,
        "device": device,
        "what_is_measured": {
            "correctness": "every case: the outputs against those LiteRT's reference kernels computed (testdata/), bit for bit unless the case has a tolerance",
            "timing": "only the cases with \"timing\": true: the model is linked into the firmware and runs in place from flash, and the counter of the device is a clock. The cycles of the other cases are no performance figures.",
        },
        "summary": {**report.counts(), "exit_code": report.exit_code, "problems": report.problems},
        "firmware_images": report.images,
        "cases": cases,
    }
