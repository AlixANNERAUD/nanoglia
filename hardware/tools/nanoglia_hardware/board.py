"""The devices the host tool drives, behind one interface: a board on a serial port (flashed with
bossac through its bootloader), the emulated runner, a simulated device.

A board has `open()` (a `Device` that answers), `recover()` (the same after the device stopped
answering: wait for the firmware to restart, flash it again if the board sits in its bootloader),
`flash(image)` and `close()`. `Unreachable` means nobody can go further without a hand on the
reset button; `SetupError` that the host lacks something (a port, bossac, an image).
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from elftools.elf.elffile import ELFFile

from . import protocol as p

# USB identifiers. The firmware of hardware/nano33ble uses the test identifiers of pid.codes; the
# others are those of boards.txt in ArduinoCore-mbed (nano33ble.vid.0/pid.0 is the bootloader,
# vid.1/pid.1 a sketch of the Arduino core).
RUNNER = (0x1209, 0x0001)
RUNNER_PRODUCT = "nanoglia hardware runner"
BOOTLOADER = (0x2341, 0x005A)
ARDUINO_SKETCH = (0x2341, 0x805A)

# Where the Arduino bootloader of the Nano 33 BLE starts the application, and the flash after it
# (nano33ble.upload.maximum_size in boards.txt). bossac's addresses start there: it cannot reach
# the bootloader.
APPLICATION_START = 0x10000
APPLICATION_MAX = 983040
RAM = (0x2000_0000, 256 * 1024)

BOSSAC_VERSION = "1.9.1-arduino2"


class Unreachable(Exception):
    """The board does not answer and nothing the host can do brings it back."""


class SetupError(Exception):
    """The host lacks something to do what is asked."""


@dataclass
class Port:
    device: str
    vid: int | None
    pid: int | None
    product: str
    serial_number: str

    @property
    def role(self) -> str:
        key = (self.vid, self.pid)
        if key == RUNNER:
            return "runner" if RUNNER_PRODUCT in self.product or not self.product else "other (pid.codes test identifiers)"
        return {BOOTLOADER: "bootloader", ARDUINO_SKETCH: "arduino sketch"}.get(key, "other")


def list_ports() -> list[Port]:
    from serial.tools import list_ports as lister

    return [
        Port(port.device, port.vid, port.pid, port.product or "", port.serial_number or "")
        for port in sorted(lister.comports(), key=lambda port: port.device)
    ]


@dataclass
class Image:
    """A firmware image: what bossac writes after the bootloader."""

    path: Path
    binary: bytes

    @classmethod
    def load(cls, path: Path) -> "Image":
        """Reads an ELF (or a raw binary) and refuses what is not an application of this board:
        linked elsewhere than after the bootloader, too large, or without a vector table."""
        try:
            data = path.read_bytes()
        except OSError as error:
            raise SetupError(f"{path}: {error.strerror}") from None
        if data[:4] == b"\x7fELF":
            with open(path, "rb") as file:
                segments = sorted(
                    (segment["p_paddr"], segment.data())
                    for segment in ELFFile(file).iter_segments()
                    if segment["p_type"] == "PT_LOAD" and segment["p_filesz"] > 0
                )
            if not segments or segments[0][0] != APPLICATION_START:
                start = f"{segments[0][0]:#x}" if segments else "nowhere"
                raise SetupError(f"{path}: linked at {start}, the bootloader starts the application at {APPLICATION_START:#x}")
            binary = bytearray()
            for address, content in segments:
                # Gaps between segments read as erased flash.
                binary += b"\xff" * (address - APPLICATION_START - len(binary))
                binary += content
            data = bytes(binary)
        if len(data) > APPLICATION_MAX:
            raise SetupError(f"{path}: {len(data)} bytes, {APPLICATION_MAX} fit after the bootloader")
        stack, reset = (int.from_bytes(data[at : at + 4], "little") for at in (0, 4))
        in_ram = RAM[0] < stack <= RAM[0] + RAM[1]
        in_flash = APPLICATION_START <= reset < APPLICATION_START + APPLICATION_MAX and reset & 1
        if not (in_ram and in_flash):
            raise SetupError(f"{path}: no vector table of an application linked at {APPLICATION_START:#x}")
        return cls(path, data)


def find_bossac(explicit: str | None) -> str:
    """Arduino's bossac: the one named, `$BOSSAC`, the one on the path, or the one of an Arduino
    installation."""
    candidates = [explicit, os.environ.get("BOSSAC"), shutil.which("bossac")]
    for arduino in ("~/.arduino15", "~/Library/Arduino15"):
        candidates.append(os.path.expanduser(f"{arduino}/packages/arduino/tools/bossac/{BOSSAC_VERSION}/bossac"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SetupError(
        f"bossac not found: install Arduino's bossac {BOSSAC_VERSION} (the mainline one does not know the "
        "nRF52840) and pass --bossac, set $BOSSAC or put it on the path; see "
        "documentation/how-to-guides/run-the-golden-cases-on-a-board.md"
    )


class SerialTransport:
    """A serial port, as a transport."""

    def __init__(self, device: str):
        import serial

        self.serial_module = serial
        try:
            self.port = serial.Serial(device, 115200, timeout=0.1, write_timeout=5)
        except (serial.SerialException, OSError) as error:
            raise p.Disconnected(f"{device}: {error}") from None

    def write(self, data: bytes) -> None:
        try:
            self.port.write(data)
            self.port.flush()
        except (self.serial_module.SerialException, OSError) as error:
            raise p.Disconnected(str(error)) from None

    def read(self, size: int, timeout: float) -> bytes:
        try:
            self.port.timeout = timeout
            return self.port.read(size)
        except (self.serial_module.SerialException, OSError) as error:
            raise p.Disconnected(str(error)) from None

    def close(self) -> None:
        try:
            self.port.close()
        except (self.serial_module.SerialException, OSError):
            pass


class SerialBoard:
    """An Arduino Nano 33 BLE on a USB serial port."""

    name = "serial"
    # Time the firmware has to come back by itself: its watchdog (20 s), the reset through the
    # bootloader and the enumeration of the port.
    RESTART_S = 60
    # Time the bootloader, then the firmware, has to show its port.
    PORT_S = 15

    def __init__(self, port: str | None = None, bossac: str | None = None, log=print):
        self.port_hint = port
        self.bossac = bossac
        self.log = log
        self.device: p.Device | None = None
        self.image: Image | None = None

    def _find(self, role: str) -> Port | None:
        found = [port for port in list_ports() if port.role == role]
        if self.port_hint:
            found = [port for port in found if port.device == self.port_hint] or found
        if len(found) > 1:
            names = ", ".join(port.device for port in found)
            raise SetupError(f"several boards ({names}): pass --port")
        return found[0] if found else None

    def _wait(self, role: str, seconds: float) -> Port | None:
        deadline = time.monotonic() + seconds
        while True:
            if (port := self._find(role)) is not None:
                return port
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.25)

    def _connect(self, seconds: float) -> p.Device | None:
        """A device that answers INFO on the port of the runner, within `seconds`."""
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            port = self._wait("runner", remaining)
            if port is None:
                return None
            try:
                device = p.Device(p.Link(SerialTransport(port.device)))
                device.info(timeout=1.0, attempts=2)
            except (p.Disconnected, p.NoAnswer):
                # The port is there before the firmware serves it, or it just went away.
                time.sleep(0.5)
                continue
            self.device = device
            return device
        return None

    def close(self) -> None:
        if self.device is not None:
            self.device.link.close()
            self.device = None

    def open(self) -> p.Device:
        if self.device is None and self._connect(3) is None:
            ports = ", ".join(f"{port.device} ({port.role})" for port in list_ports()) or "none"
            raise SetupError(
                f"no board running the runner firmware (serial ports: {ports}): flash it by passing "
                "its images, after a double tap on the reset button if the board runs something else"
            )
        return self.device

    def recover(self) -> p.Device:
        self.close()
        self.log(f"  waiting up to {self.RESTART_S} s for the firmware to restart")
        if self._connect(self.RESTART_S) is not None:
            return self.device
        if self._find("bootloader") is not None and self.image is not None:
            self.log("  the board is in its bootloader: flashing the image again")
            self.flash(self.image)
            return self.open()
        raise Unreachable(
            "the board does not answer and did not restart: double-tap its reset button (the orange "
            "LED fades in and out in the bootloader), then run again"
        )

    def enter_bootloader(self) -> Port:
        """Brings the board into its bootloader, from whatever answers on its port."""
        if (port := self._find("bootloader")) is not None:
            return port
        if self.device is not None or self._connect(1) is not None:
            try:
                self.device.reset(p.BOOTLOADER)
            except (p.Disconnected, p.NoAnswer, p.DeviceError):
                pass
            self.close()
        if self._wait("bootloader", 3) is None:
            # The 1200 baud touch of Arduino: open the port at 1200 baud and close it. The runner
            # firmware and the sketches of the Arduino core both reset into the bootloader on it.
            for role in ("runner", "arduino sketch"):
                if (port := self._find(role)) is not None:
                    self.log(f"  1200 baud touch on {port.device} ({role})")
                    import serial

                    try:
                        serial.Serial(port.device, 1200).close()
                    except (serial.SerialException, OSError) as error:
                        self.log(f"  touch failed: {error}")
                    break
        port = self._wait("bootloader", self.PORT_S)
        if port is None:
            raise Unreachable(
                "the board did not enter its bootloader: double-tap its reset button (the orange LED "
                "fades in and out in the bootloader), then run again"
            )
        return port

    def flash(self, image: Image) -> None:
        bossac = find_bossac(self.bossac)
        port = self.enter_bootloader()
        binary = image.path.with_name(image.path.name + ".bin")
        binary.write_bytes(image.binary)
        # The upload pattern of ArduinoCore-mbed's platform.txt (tools.bossac.upload.pattern):
        # bossac takes the name of the port without /dev/, addresses the application from 0 (the
        # bootloader places it at 0x10000 and cannot be written), erases, writes and resets.
        command = [bossac, "-d", f"--port={Path(port.device).name}", "-U", "-i", "-e", "-w", str(binary), "-R"]
        self.log(f"  flashing {image.path.name} ({len(image.binary)} bytes): {' '.join(command)}")
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            raise Unreachable("bossac did not finish in 180 s: double-tap the reset button and run again") from None
        if done.returncode != 0:
            tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-15:])
            raise SetupError(f"bossac failed ({done.returncode}):\n{tail}")
        self.image = image
        if self._connect(self.PORT_S + 15) is None:
            raise Unreachable(
                f"{image.path.name} was written but the firmware does not answer: it may not start on "
                "this board. Double-tap the reset button to return to the bootloader"
            )


class EmulatedBoard:
    """The emulated runner: one firmware, which holds every model."""

    name = "emulated"

    def __init__(self, elf: Path, log=print):
        from .emulated import EmulatedDevice

        self.emulator = EmulatedDevice(elf)
        self.device = p.Device(p.Link(self.emulator))

    def open(self) -> p.Device:
        if self.emulator.bootloader:
            raise SetupError("the emulated board is in its bootloader")
        return self.device

    def recover(self) -> p.Device:
        # The emulated firmware restarts within the write that crashed it.
        if self.emulator.bootloader:
            self.emulator.flashed()
        self.device.link.buffer.clear()
        self.emulator.outgoing.clear()
        return self.device

    def flash(self, image) -> None:
        self.emulator.flashed()

    def close(self) -> None:
        pass


class SimulatedBoard:
    """A simulated device, whose firmware images are lists of models."""

    name = "simulated"

    def __init__(self, device, log=print):
        self.simulated = device
        self.device = p.Device(p.Link(device))
        self.image: list[tuple[str, bytes]] | None = None

    def open(self) -> p.Device:
        if self.simulated.state != "running":
            raise SetupError(f"the simulated board is {self.simulated.state}")
        return self.device

    def recover(self) -> p.Device:
        self.simulated.restart()
        if self.simulated.state == "bootloader" and self.image is not None:
            self.simulated.flashed(self.image)
        if self.simulated.state != "running":
            raise Unreachable(f"the simulated board is {self.simulated.state}")
        self.device.link.buffer.clear()
        return self.device

    def flash(self, image: list[tuple[str, bytes]]) -> None:
        self.image = image
        self.simulated.flashed(image)

    def close(self) -> None:
        pass
