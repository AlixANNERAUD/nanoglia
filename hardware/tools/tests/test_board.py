"""What the host does to a board, without one: the check of an image and the bossac command."""

import struct
import sys

import pytest

from nanoglia_hardware import board
from nanoglia_hardware.board import Image, Port, SerialBoard, SetupError, Unreachable


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


def test_flash_runs_bossac_as_the_arduino_core_does(tmp_path, monkeypatch):
    image_path = tmp_path / "firmware.bin"
    image_path.write_bytes(binary())
    arguments = tmp_path / "arguments"
    bossac = tmp_path / "bossac"
    bossac.write_text(f"#!{sys.executable}\nimport sys\nopen({str(arguments)!r}, 'w').write('\\n'.join(sys.argv[1:]))\n")
    bossac.chmod(0o755)
    monkeypatch.setattr(board, "list_ports", Ports([BOOTLOADER]))
    connected = []
    serial_board = SerialBoard(bossac=str(bossac), log=lambda _: None)
    monkeypatch.setattr(serial_board, "_connect", lambda seconds: connected.append(seconds) or object())
    serial_board.flash(Image.load(image_path))
    written = tmp_path / "firmware.bin.bin"
    assert arguments.read_text().split("\n") == ["-d", "--port=ttyACM1", "-U", "-i", "-e", "-w", str(written), "-R"]
    assert written.read_bytes() == binary()
    assert connected, "the firmware is waited for after the flash"

    # bossac fails: a setup error with its output; no bossac: the same, before touching the board.
    bossac.write_text(f"#!{sys.executable}\nimport sys\nprint('No device found on ttyACM1')\nsys.exit(1)\n")
    with pytest.raises(SetupError, match="No device found"):
        serial_board.flash(Image.load(image_path))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("BOSSAC", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SetupError, match="bossac not found"):
        SerialBoard(log=lambda _: None).flash(Image.load(image_path))


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
