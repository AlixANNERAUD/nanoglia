"""What the host does to a board, without one: the check of an image and the bossac command."""

import struct
import sys

import pytest

from nanoglia_hardware import board
from nanoglia_hardware import protocol as p
from nanoglia_hardware.board import FirmwareDead, Image, Port, SerialBoard, SetupError, Unreachable


def binary(stack=0x2004_0000, reset=0x0001_0101, size=4096):
    return struct.pack("<II", stack, reset) + bytes(size - 8)


def test_an_image_must_be_an_application_of_the_board(tmp_path):
    path = tmp_path / "firmware.bin"
    path.write_bytes(binary())
    assert Image.load(path).binary == binary()
    # A vector table for another address (an image linked at 0, which would need the bootloader's
    # place), no vector table, too large, missing.
    for content in (binary(reset=0x0000_0101), binary(stack=0), binary(reset=0x0001_0100), bytes(4096)):
        path.write_bytes(content)
        with pytest.raises(SetupError, match="vector table"):
            Image.load(path)
    path.write_bytes(binary(size=board.APPLICATION_MAX + 4))
    with pytest.raises(SetupError, match="fit after the bootloader"):
        Image.load(path)
    with pytest.raises(SetupError):
        Image.load(tmp_path / "missing.elf")


def elf(address, content):
    """A minimal ELF32 for Arm with one loaded segment at `address`."""
    header = b"\x7fELF" + bytes([1, 1, 1]) + bytes(9)
    header += struct.pack("<HHIIIIIHHHHHH", 2, 40, 1, address | 1, 52, 0, 0, 52, 32, 1, 0, 0, 0)
    segment = struct.pack("<IIIIIIII", 1, 84, address, address, len(content), len(content), 5, 4)
    return header + segment + content


def test_an_elf_is_flattened_from_the_start_of_the_application(tmp_path):
    path = tmp_path / "firmware.elf"
    path.write_bytes(elf(board.APPLICATION_START, binary()))
    assert Image.load(path).binary == binary()
    # Linked at 0, where the bootloader is: refused before anything is sent to the board.
    path.write_bytes(elf(0, binary(reset=0x101)))
    with pytest.raises(SetupError, match="linked at 0x0"):
        Image.load(path)


class Ports:
    """The serial ports of a machine, as a script: each call to `list_ports` moves on."""

    def __init__(self, *steps):
        self.steps = list(steps)

    def __call__(self):
        step = self.steps[0]
        if len(self.steps) > 1:
            self.steps.pop(0)
        return step


RUNNER = Port("/dev/ttyACM0", *board.RUNNER, board.RUNNER_PRODUCT, "ABC")
BOOTLOADER = Port("/dev/ttyACM1", *board.BOOTLOADER, "Nano 33 BLE", "")
SKETCH = Port("/dev/ttyACM0", *board.ARDUINO_SKETCH, "Nano 33 BLE", "")


def test_ports_are_told_apart():
    assert (RUNNER.role, BOOTLOADER.role, SKETCH.role) == ("runner", "bootloader", "arduino sketch")
    assert Port("/dev/ttyS0", None, None, "", "").role == "other"
    assert Port("/dev/ttyACM3", *board.RUNNER, "another test device", "").role.startswith("other")


def fake_bossac(directory, version="1.9.1-17-g89f3556", fail=""):
    """A stand-in for bossac: says its version, records its arguments, fails on demand."""
    arguments = directory / "arguments"
    bossac = directory / "bossac"
    bossac.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if '--help' in sys.argv:\n"
        f"    print('Usage: bossac [OPTION...] [FILE]\\nBasic Open Source SAM-BA Application (BOSSA) Version {version}')\n"
        "    sys.exit(0)\n"
        f"open({str(arguments)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        + (f"print({fail!r})\nsys.exit(1)\n" if fail else "")
    )
    bossac.chmod(0o755)
    return str(bossac), arguments


@pytest.fixture
def quick(monkeypatch):
    """The waits of a board, shortened."""
    for name, seconds in (("BOOT_S", 0.6), ("RESTART_S", 2.0), ("START_S", 2.5), ("PORT_S", 0.6), ("SETTLE_S", 0.5)):
        monkeypatch.setattr(SerialBoard, name, seconds)


def test_the_waits_follow_the_firmware():
    # After a crash: the firmware may hang at each start, and reaches the bootloader only after
    # its watchdog fired `MAX_FAILED_BOOTS` times. After a flash: a firmware without a port gives
    # the hand back after `NO_CONTACT_S`.
    assert SerialBoard.RESTART_S >= p.MAX_FAILED_BOOTS * p.WATCHDOG_S + 3 * SerialBoard.BOOT_S == 90
    assert SerialBoard.START_S >= p.NO_CONTACT_S + 30 == 330
    assert (p.WATCHDOG_S, p.MAX_FAILED_BOOTS, p.NO_CONTACT_S) == (20, 3, 300)


def test_flash_runs_bossac_as_the_arduino_core_does(tmp_path, monkeypatch, quick):
    image_path = tmp_path / "firmware.bin"
    image_path.write_bytes(binary())
    bossac, arguments = fake_bossac(tmp_path)
    # In the bootloader, then gone with the reset, then the firmware.
    monkeypatch.setattr(board, "list_ports", Ports([BOOTLOADER], [BOOTLOADER], [], [RUNNER]))
    serial_board = SerialBoard(bossac=bossac, log=lambda _: None)
    monkeypatch.setattr(serial_board, "_open", lambda port: setattr(serial_board, "device", object()) or object())
    serial_board.flash(Image.load(image_path))
    written = tmp_path / "firmware.bin.bin"
    assert arguments.read_text() == f"-d --port=ttyACM1 -U -i -e -w -v {written} -R\n"
    assert written.read_bytes() == binary()
    assert serial_board.device is not None, "the firmware is waited for after the flash"

    # bossac fails: a setup error with its output.
    bossac, _ = fake_bossac(tmp_path, fail="No device found on ttyACM1")
    monkeypatch.setattr(board, "list_ports", Ports([BOOTLOADER]))
    with pytest.raises(SetupError, match="No device found"):
        serial_board.flash(Image.load(image_path))


def test_only_arduinos_bossac_is_used(tmp_path, monkeypatch):
    image_path = tmp_path / "firmware.bin"
    image_path.write_bytes(binary())
    monkeypatch.setattr(board, "list_ports", Ports([BOOTLOADER]))
    # The mainline bossac does not know the chip: refused before the board is touched.
    bossac, arguments = fake_bossac(tmp_path, version="1.9.1")
    with pytest.raises(SetupError, match="not Arduino's bossac"):
        SerialBoard(bossac=bossac, log=lambda _: None).flash(Image.load(image_path))
    assert not arguments.exists()
    assert board.bossac_version(fake_bossac(tmp_path)[0]) == "1.9.1-17-g89f3556"
    # No bossac at all.
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("BOSSAC", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SetupError, match="bossac not found"):
        SerialBoard(log=lambda _: None).flash(Image.load(image_path))


def test_a_firmware_that_never_answers_is_told_from_a_lost_board(tmp_path, monkeypatch, quick):
    image = tmp_path / "firmware.bin"
    image.write_bytes(binary())
    bossac, _ = fake_bossac(tmp_path)
    # Flashed, the port of the bootloader goes, nothing answers, then the bootloader is back and
    # stays: the firmware gave the hand back by itself.
    monkeypatch.setattr(board, "list_ports", Ports([BOOTLOADER], [BOOTLOADER], [], [], [], [], [BOOTLOADER]))
    with pytest.raises(FirmwareDead, match="does not work on this board"):
        SerialBoard(bossac=bossac, log=lambda _: None).flash(Image.load(image))
    # Nothing ever comes back: only then is a hand asked for.
    monkeypatch.setattr(board, "list_ports", Ports([BOOTLOADER], [BOOTLOADER], []))
    with pytest.raises(Unreachable, match="double-tap"):
        SerialBoard(bossac=bossac, log=lambda _: None).flash(Image.load(image))
    # The port of the bootloader seen in passing, during the reset, is not the bootloader.
    monkeypatch.setattr(board, "list_ports", Ports([BOOTLOADER], [BOOTLOADER], [], [BOOTLOADER], [], [RUNNER]))
    serial_board = SerialBoard(bossac=bossac, log=lambda _: None)
    monkeypatch.setattr(serial_board, "_open", lambda port: setattr(serial_board, "device", object()) or object())
    serial_board.flash(Image.load(image))


def test_after_a_crash_the_bootloader_is_waited_for_too(tmp_path, monkeypatch, quick):
    image = tmp_path / "firmware.bin"
    image.write_bytes(binary())
    # The firmware crashed at each restart: no port for a while, then the bootloader.
    monkeypatch.setattr(board, "list_ports", Ports([], [], [], [], [BOOTLOADER]))
    serial_board = SerialBoard(log=lambda _: None)
    with pytest.raises(FirmwareDead, match="pass the images"):
        serial_board.recover()
    flashed = []
    monkeypatch.setattr(board, "list_ports", Ports([], [], [], [], [BOOTLOADER]))
    serial_board.image = Image.load(image)
    monkeypatch.setattr(serial_board, "flash", lambda image: flashed.append(image))
    monkeypatch.setattr(serial_board, "open", lambda: "device")
    assert serial_board.recover() == "device" and flashed == [serial_board.image]
    # The firmware comes back: no flash.
    monkeypatch.setattr(board, "list_ports", Ports([], [], [RUNNER]))
    monkeypatch.setattr(serial_board, "_open", lambda port: setattr(serial_board, "device", "runner") or "runner")
    assert serial_board.recover() == "runner" and len(flashed) == 1
    # Nothing comes back.
    serial_board.device = None
    monkeypatch.setattr(board, "list_ports", Ports([]))
    with pytest.raises(Unreachable, match="double-tap"):
        serial_board.recover()


def test_leaving_the_bootloader(tmp_path, monkeypatch, quick):
    bossac, arguments = fake_bossac(tmp_path)
    monkeypatch.setattr(board, "list_ports", Ports([BOOTLOADER], [], [RUNNER]))
    serial_board = SerialBoard(bossac=bossac, log=lambda _: None)
    monkeypatch.setattr(serial_board, "_open", lambda port: setattr(serial_board, "device", object()) or object())
    assert serial_board.leave_bootloader()
    assert arguments.read_text() == "-d --port=ttyACM1 -U -R\n"
    # Without an application the bootloader stays.
    monkeypatch.setattr(board, "list_ports", Ports([BOOTLOADER]))
    assert not serial_board.leave_bootloader()
    monkeypatch.setattr(board, "list_ports", Ports([RUNNER]))
    with pytest.raises(SetupError, match="not in its bootloader"):
        serial_board.leave_bootloader()


def test_without_a_bootloader_port_the_board_needs_a_hand(monkeypatch):
    monkeypatch.setattr(board, "list_ports", Ports([]))
    monkeypatch.setattr(SerialBoard, "PORT_S", 0.3)
    serial_board = SerialBoard(log=lambda _: None)
    with pytest.raises(Unreachable, match="double-tap"):
        serial_board.enter_bootloader()
    with pytest.raises(SetupError, match="no board running the runner firmware"):
        serial_board.open()


def test_two_boards_need_a_port(monkeypatch):
    other = Port("/dev/ttyACM5", *board.BOOTLOADER, "Nano 33 BLE", "")
    monkeypatch.setattr(board, "list_ports", Ports([BOOTLOADER, other]))
    with pytest.raises(SetupError, match="several boards"):
        SerialBoard(log=lambda _: None).enter_bootloader()
    assert SerialBoard(port="/dev/ttyACM5", log=lambda _: None).enter_bootloader() == other
