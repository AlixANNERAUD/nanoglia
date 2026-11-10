"""The emulated runner under Unicorn: the Rust code of the board firmware, minus the board, driven
by the code of the host tool. Needs the firmware of hardware/emulated, built for
thumbv7em-none-eabihf (`mise run test-hardware`): `NANOGLIA_EMULATED_ELF` names it; without the
variable, the tests are skipped if it is not where Cargo puts it."""

import os
import struct
from pathlib import Path

import pytest

from nanoglia_hardware import protocol as p
from nanoglia_hardware.board import EmulatedBoard
from nanoglia_hardware.cases import run_cases
from nanoglia_hardware.shared import REPO

DEFAULT = REPO / "hardware/target/thumbv7em-none-eabihf/release/nanoglia-hardware-emulated"
ELF = Path(os.environ.get("NANOGLIA_EMULATED_ELF", DEFAULT))
if "NANOGLIA_EMULATED_ELF" in os.environ:
    assert ELF.is_file(), f"NANOGLIA_EMULATED_ELF names no file: {ELF}"
pytestmark = pytest.mark.skipif(not ELF.is_file(), reason="the emulated runner is not built")


@pytest.fixture
def board():
    return EmulatedBoard(ELF)


def test_the_crash_message_is_kept_above_what_the_bootloader_uses(board):
    # The record of a crash lives in the last 256 bytes of the RAM: the bootloader of the board
    # keeps everything, its stacks included, below 0x20009e14 (its initial stack pointer is
    # 0x20008d48).
    record = next(address for name, address in board.emulator.fw.symbols.items() if "crash" in name and "RECORD" in name)
    assert record == 0x2004_0000 - 256
    assert board.emulator.fw.symbols["_stack_start"] == record


def test_info(board):
    info = board.open().info()
    assert info.opt_level == "3" and info.rustc.startswith("rustc ") and info.settings.startswith("none")
    assert info.protocol == p.VERSION and info.max_payload == p.MAX_PAYLOAD
    assert info.board.startswith("emulated") and info.counter_hz == 0
    assert info.crash is None and info.failed_boots == 0
    # The RAM of the Arduino Nano 33 BLE, minus the stack and the static data.
    assert 220 * 1024 < info.ram_bytes < 232 * 1024
    assert [name for name, _ in info.models] == [
        "kws_ref_model.tflite",
        "pretrainedResnet_quant.tflite",
        "str_ww_ref_model.tflite",
        "ad01_int8.tflite",
        "vww_96_int8.tflite",
    ]


def test_golden_cases_from_flash_and_from_ram(board):
    log = []
    report = run_cases(board, [], REPO / "testdata", "sample_0", log=log.append)
    assert report.exit_code == 0, log
    results = {result.name.split("/")[2]: result for result in report.results}
    assert len(results) == 5 and all(result.model_location == "flash" for result in results.values())
    assert results["ad01_int8"].status == "pending"
    kws = results["kws_ref_model"]
    assert kws.status == "ok" and len(kws.operators) == 13
    assert kws.operators[0]["kernel"] == "cmsis-nn/conv2d_s8"
    # The counter of the emulator moves by a fixed step at each reading: every operator is one
    # step, and the inference the sum of its operators.
    assert all(op["cycles"] == 1000 for op in kws.operators)
    assert kws.compute_cycles == 1000 * 13
    assert not kws.timing and kws.compute_ms is None
    assert 0 < kws.stack_peak < 24 * 1024
    vww = results["vww_96_int8"]
    assert vww.status == "ok" and vww.arena_used + vww.inputs_bytes < board.open().info().ram_bytes

    report = run_cases(board, [], REPO / "testdata", "ops/conv2d_", log=log.append)
    assert report.exit_code == 0 and len(report.results) > 3
    assert all(result.model_location == "ram" and not result.timing for result in report.results)


def test_the_firmware_survives_broken_frames(board):
    emulator, device = board.emulator, board.open()
    # A wrong CRC, straight to the firmware.
    frame = bytearray(p.encode(p.INFO, 77))
    frame[-1] ^= 1
    emulator.write(bytes(frame))
    answer = bytes(emulator.outgoing)
    emulator.outgoing.clear()
    assert answer[3] == p.ERROR and answer[4] == 77 and answer[7] == p.BAD_CRC
    # A frame cut short, then silence: the firmware drops it when its timeout passes, and says so.
    emulator.write(p.encode(p.WRITE, 78, bytes(100))[:40])
    assert not emulator.outgoing
    answer = emulator.read(1024, 1.5)
    assert answer[3] == p.ERROR and answer[7] == p.FRAME_TIMEOUT
    # Silence between frames is no error, and the next request is served.
    assert emulator.read(1024, 5.0) == b""
    assert device.info().firmware.startswith("nanoglia-hardware-emulated")
    # Requests the firmware refuses.
    with pytest.raises(p.DeviceError) as error:
        device.write(device.info().ram_bytes - 2, bytes(4))
    assert error.value.code == p.BAD_ARGUMENT
    with pytest.raises(p.DeviceError) as error:
        device.output(0, 4)
    assert error.value.code == p.NO_RUN
    with pytest.raises(p.DeviceError) as error:
        device.link.request(0x55)
    assert error.value.code == p.UNKNOWN_COMMAND
    # A model that is none: the load fails, with the error of nanoglia.
    device.write(0, bytes(64))
    run = device.run(p.RAM, 0, 64, 64, [], 5)
    assert run.status == p.LOAD and "invalid encoding" in run.message
    # No room for the arena: too large.
    model = (REPO / "testdata/mlperf-tiny/kws_ref_model.tflite").read_bytes()
    run = device.run(p.EMBEDDED, 0, len(model), device.info().ram_bytes - 1024, [], 5)
    assert run.status == p.LOAD and run.error == p.TOO_LARGE


def test_crashes_are_reported_after_the_restart(board):
    import hardware

    log = []
    assert hardware.self_test(board, log.append), log
    assert any("restarted after a panic: panicked at" in line for line in log)
    assert any("restarted after a watchdog" in line for line in log)
    # Three resets by the firmware itself, then one the host asked for.
    assert board.emulator.resets == [p.APPLICATION] * 4


def test_reset_into_the_bootloader(board):
    board.open().reset(p.BOOTLOADER)
    assert board.emulator.bootloader and board.emulator.resets == [p.BOOTLOADER]
    with pytest.raises(p.Disconnected):
        board.device.info()
    # A flash starts the firmware again.
    board.flash(None)
    assert board.open().info().crash is None


def test_a_firmware_that_crashes_while_starting_ends_in_the_bootloader(board):
    emulator = board.emulator
    # From now on the firmware panics before it serves anything.
    emulator.inject = 1
    try:
        board.open().fault(p.PANIC)
    except (p.NoAnswer, p.Disconnected):
        pass
    # The panic the host asked for, then a panic at each start, before the runner even exists:
    # the handler of the third crash in a row gives the hand to the bootloader.
    assert emulator.bootloader
    assert emulator.resets == [p.APPLICATION, p.APPLICATION, p.BOOTLOADER]
    assert emulator.retained == 0xA0, "the count starts again for the next firmware"
    # A good firmware flashed from there starts with a clean count.
    emulator.inject = 0
    board.flash(None)
    info = board.open().info()
    assert info.crash is None and info.failed_boots == 0


def test_a_crash_after_a_request_does_not_count_towards_the_bootloader(board):
    for _ in range(5):
        try:
            board.open().fault(p.PANIC)
        except (p.NoAnswer, p.Disconnected):
            pass
        info = board.recover().info()
        assert info.crash == "panic" and info.failed_boots == 1
    assert not board.emulator.bootloader


def test_the_stack_is_measured(board):
    device = board.open()
    model = (REPO / "testdata/mlperf-tiny/kws_ref_model.tflite").read_bytes()
    case = REPO / "testdata/mlperf-tiny/golden/kws_ref_model/sample_0"
    data = (case / "input_0.bin").read_bytes()
    device.write(0, data)
    run = device.run(p.EMBEDDED, 0, len(model), 496, [(0, len(data))], 10)
    assert run.status == p.OK
    assert 1024 < run.stack_peak < 16 * 1024
    assert run.output_lengths == [12]
    assert device.output(0, 12) == (case / "output_0.bin").read_bytes()
    (crc,) = struct.unpack("<I", device.link.request(p.CRC, struct.pack("<BII", p.EMBEDDED, 0, len(model))))
    import zlib

    assert crc == zlib.crc32(model)
