"""The run of the golden cases against the simulated device: comparison, results, and what the
tool does when the device corrupts frames, hangs, crashes, lands in its bootloader or dies."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hardware  # noqa: E402
from nanoglia_hardware import protocol as p  # noqa: E402
from nanoglia_hardware.board import SimulatedBoard  # noqa: E402
from nanoglia_hardware.cases import RAM_NOTE, results_json, run_cases  # noqa: E402
from nanoglia_hardware.shared import REPO  # noqa: E402
from nanoglia_hardware.simulated import Faults, SimulatedDevice  # noqa: E402

ROOT = REPO / "testdata"
# Seven cases: four single-operator ones (model in RAM) and three of a model in flash.
FILTER_OPS = "ops/abs_"
KWS = "kws_ref_model/sample"


def run(filter, faults=None, oracle=None, images=None, ram=None, **options):
    images = hardware.simulated_images() if images is None else images
    device = SimulatedDevice(oracle or hardware.golden_oracle(ROOT), images[0] if images else [], faults)
    if ram is not None:
        device.ram = bytearray(ram)
    log = []
    report = run_cases(SimulatedBoard(device), images, ROOT, filter, log=log.append, **options)
    return report, device, log


def by_status(report):
    statuses = {}
    for result in report.results:
        statuses.setdefault(result.status, []).append(result.name)
    return statuses


def test_every_case_passes_and_the_results_say_where_the_model_was():
    report, device, log = run("")
    counts = report.counts()
    assert report.exit_code == 0, log[-5:]
    assert counts["cases"] > 300 and counts["failed"] == 0 and counts["skipped"] == 0
    assert counts["pending"] == 3
    results = {result.name: result for result in report.results}
    kws = results["mlperf-tiny/golden/kws_ref_model/sample_0"]
    assert (kws.model_location, kws.image) == ("flash", "image 1")
    vww = results["mlperf-tiny/golden/vww_96_int8/sample_0"]
    assert (vww.model_location, vww.image) == ("flash", "image 2")
    single = results["ops/abs_f32"]
    assert (single.model_location, single.image, single.timing, single.timing_note) == ("ram", "image 1", False, RAM_NOTE)
    # The simulated device has no clock: nothing is a performance figure.
    assert counts["timed"] == 0 and kws.compute_ms is None

    data = results_json(report, "simulated", "2026-01-01T00:00:00+00:00", "abc")
    json.dumps(data)
    assert data["summary"]["exit_code"] == 0 and len(data["firmware_images"]) == 2
    case = next(case for case in data["cases"] if case["name"] == kws.name)
    assert case["model_location"] == "flash" and case["operators"][0]["kernel"]
    assert "ran" not in case and "comparison" not in case


def test_cycles_become_milliseconds_only_for_a_model_in_flash():
    class Clocked(SimulatedDevice):
        def _command(self, kind, seq, payload):
            if kind != p.INFO:
                return super()._command(kind, seq, payload)
            start = len(self.outgoing)
            super()._command(kind, seq, payload)
            # The same answer, from a device whose counter is a clock of 64 MHz.
            body = bytearray(self.outgoing[start + p.HEADER : -p.TRAILER])
            body[9:13] = (64_000_000).to_bytes(4, "little")
            del self.outgoing[start:]
            self.answers -= 1
            self._send(kind | p.RESPONSE, seq, bytes(body))

    images = hardware.simulated_images()
    device = Clocked(hardware.golden_oracle(ROOT), images[0])
    report = run_cases(SimulatedBoard(device), images, ROOT, "sample_0", log=lambda _: None)
    results = {result.name: result for result in report.results}
    kws = results["mlperf-tiny/golden/kws_ref_model/sample_0"]
    assert kws.timing and kws.timing_note == ""
    assert kws.compute_ms == pytest.approx(kws.compute_cycles * 1000 / 64_000_000, abs=1e-4)
    assert kws.operators[0]["ms"] == pytest.approx(kws.operators[0]["cycles"] / 64_000, abs=1e-4)

    device = Clocked(hardware.golden_oracle(ROOT), [])
    report = run_cases(SimulatedBoard(device), [], ROOT, FILTER_OPS, log=lambda _: None)
    assert report.results and all(not r.timing and r.compute_ms is None for r in report.results)
    assert all(r.timing_note == RAM_NOTE and r.operators[0]["ms"] is None for r in report.results)
    assert all(r.operators[0]["cycles"] > 0 for r in report.results), "the cycles are recorded all the same"


def test_corrupted_and_lost_answers_are_asked_again():
    faults = Faults(corrupt_answers={2, 9}, lost_answers={5, 12})
    report, device, log = run(FILTER_OPS, faults)
    assert report.exit_code == 0, log
    assert all(result.status == "ok" for result in report.results)


def test_a_wrong_output_fails_the_case():
    good = hardware.golden_oracle(ROOT)
    calls = []

    def oracle(model, inputs):
        outputs, kernels = good(model, inputs)
        calls.append(1)
        if len(calls) == 2:
            outputs = [bytes([outputs[0][0] ^ 0x80]) + outputs[0][1:]] + outputs[1:]
        return outputs, kernels

    report, _, log = run(FILTER_OPS, oracle=oracle)
    assert report.exit_code == 1
    statuses = by_status(report)
    assert len(statuses["FAIL"]) == 1 and "max diff" in report.results[1].message


def test_a_pending_case_that_passes_fails_the_run():
    # The outputs of LiteRT, exactly: what a pending case must not give.
    def exact(model, inputs):
        import zlib

        for name, directory in hardware.golden.find_cases(ROOT, "ad01_int8/sample_0"):
            spec = json.loads((directory / "case.json").read_text())
            return [(directory / f"output_{i}.bin").read_bytes() for i in range(len(spec["outputs"]))], ["x"]

    report, _, _ = run("ad01_int8/sample_0", oracle=exact)
    assert report.exit_code == 1
    assert [result.status for result in report.results] == ["FIXED"]


def test_a_hang_fails_its_case_and_the_run_goes_on():
    report, device, log = run(FILTER_OPS, Faults(hang_runs={2}))
    statuses = by_status(report)
    assert report.exit_code == 1
    assert len(statuses["FAIL"]) == 1 and len(statuses["ok"]) == len(report.results) - 1
    failed = next(result for result in report.results if result.failed)
    assert "stopped answering" in failed.message and "restarted after a watchdog" in failed.message


def test_a_panic_is_reported_with_its_message():
    report, _, _ = run(FILTER_OPS, Faults(panic_runs={1}))
    assert report.exit_code == 1
    assert "restarted after a panic: panicked at kernel.rs" in report.results[0].message
    assert all(result.status == "ok" for result in report.results[1:])


def test_a_board_back_in_its_bootloader_is_flashed_again():
    report, device, log = run(KWS, Faults(hang_runs={2}, back_in_bootloader=True))
    assert report.exit_code == 1
    assert [result.status for result in report.results] == ["ok", "FAIL", "ok"]
    assert device.state == "running"


def test_a_board_that_never_comes_back_stops_the_run():
    report, device, log = run(FILTER_OPS, Faults(hang_runs={2}, dead=True))
    assert report.exit_code == 3
    assert [result.status for result in report.results] == ["ok", "FAIL"]
    assert any(line.startswith("STOPPED") for line in log)
    assert report.problems


def test_three_cases_lost_in_a_row_stop_the_run():
    report, _, log = run(FILTER_OPS, Faults(hang_runs={1, 2, 3}))
    assert report.exit_code == 3
    assert len(report.results) == 3 and all(result.failed for result in report.results)
    # Two in a row, then a case that runs: the count starts again.
    report, _, _ = run(FILTER_OPS, Faults(hang_runs={1, 2, 4}))
    assert report.exit_code == 1 and len(report.results) > 3


def test_what_does_not_fit_is_skipped_and_said():
    # No room for the model in RAM.
    report, _, _ = run(FILTER_OPS, ram=64)
    assert report.exit_code == 0
    assert all(r.status == "skipped" and "does not fit" in r.message for r in report.results)
    # Room for the model and the inputs, none for the arena: nanoglia says too large.
    report, _, _ = run(KWS, oracle=lambda model, inputs: (p.TOO_LARGE, "too large: OutOfMemory { requested: 7616 }"))
    assert all(r.status == "skipped" and "does not fit: load: too large" in r.message for r in report.results)
    # Any other error of nanoglia is a failure.
    report, _, _ = run(KWS, oracle=lambda model, inputs: (4, "unsupported operation"))
    assert report.exit_code == 1 and all(r.status == "FAIL" for r in report.results)


def test_a_model_that_must_run_from_flash_waits_for_its_image():
    # No image holds the model: skipped, with the reason, never run from RAM.
    report, device, _ = run(KWS, images=[[], []])
    assert all(r.status == "skipped" and "in none of the firmware images" in r.message for r in report.results)
    assert device.runs == 0
    # The second image holds it: the first one leaves the cases alone.
    images = hardware.simulated_images()
    report, device, _ = run(KWS, images=[images[1], images[0]])
    assert all(r.status == "ok" and r.image == "image 2" for r in report.results)


def test_no_case_is_a_failure():
    report, _, _ = run("no such case")
    assert report.exit_code == 1 and "no case" in report.problems[0]


def test_command_line(tmp_path, capsys):
    results = tmp_path / "results" / "run.json"
    assert hardware.main(["--dry-run", "--filter", FILTER_OPS, "--results", str(results)]) == 0
    data = json.loads(results.read_text())
    assert data["device"] == "simulated" and data["summary"]["failed"] == 0 and data["summary"]["cases"] >= 2
    assert "timing" in data["what_is_measured"]
    assert hardware.main(["--dry-run", "--filter", "no such case"]) == 1
    assert hardware.main(["--dry-run", "--self-test"]) == 0
    assert hardware.main(["--dry-run", "--info"]) == 0
    assert "firmware: simulated" in capsys.readouterr().out
    assert hardware.main(["--list-ports"]) == 0
    with pytest.raises(SystemExit):
        hardware.main(["--device", "emulated"])
    # No board on this machine: a setup error, not a crash.
    assert hardware.main(["--info"]) in (2, 3)
    assert hardware.main(["--image", str(tmp_path / "missing.elf")]) == 2
