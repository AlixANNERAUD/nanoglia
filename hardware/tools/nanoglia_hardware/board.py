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
# What that version says of itself with --help: the commit of its tag in arduino/BOSSA.
BOSSAC_BUILDS = ("1.9.1-17-g89f3556", "arduino")


class Unreachable(Exception):
    """The board does not answer and nothing the host can do brings it back."""


class FirmwareDead(Exception):
    """The firmware does not work on the board, which went back to its bootloader by itself: no
    hand is needed, a firmware that works is."""


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


def bossac_version(bossac: str) -> str:
    """What `bossac --help` says its version is; empty if it does not say."""
    try:
        done = subprocess.run([bossac, "--help"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    for line in (done.stdout + done.stderr).splitlines():
        if "Version" in line:
            return line.split("Version", 1)[1].strip()
    return ""


def find_bossac(explicit: str | None) -> str:
    """Arduino's bossac: the one named, `$BOSSAC`, the one on the path, or the one of an Arduino
    installation. Any other build is refused: the mainline one does not know the nRF52840."""
    candidates = [explicit, os.environ.get("BOSSAC"), shutil.which("bossac")]
    for arduino in ("~/.arduino15", "~/Library/Arduino15"):
        candidates.append(os.path.expanduser(f"{arduino}/packages/arduino/tools/bossac/{BOSSAC_VERSION}/bossac"))
    where = "documentation/how-to-guides/run-the-golden-cases-on-a-board.md"
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            version = bossac_version(candidate)
            if not any(build in version for build in BOSSAC_BUILDS):
                raise SetupError(
                    f"{candidate} is version {version or 'unknown'}, not Arduino's bossac {BOSSAC_VERSION} "
                    f"(which says {BOSSAC_BUILDS[0]}): the mainline bossac cannot flash this board; see {where}"
                )
            return candidate
    raise SetupError(
        f"bossac not found: install Arduino's bossac {BOSSAC_VERSION} (the mainline one does not know the "
        f"nRF52840) and pass --bossac, set $BOSSAC or put it on the path; see {where}"
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
    # Seconds a reset takes: the bootloader (which waits 500 ms for a second tap on the reset
    # button), the start of the firmware and the enumeration of its port.
    BOOT_S = 10
    # Seconds the firmware has to come back by itself after it stopped answering. The worst case
    # is a firmware that hangs at each start: its watchdog resets it `MAX_FAILED_BOOTS` times,
    # the last one into the bootloader.
    RESTART_S = p.MAX_FAILED_BOOTS * (p.WATCHDOG_S + BOOT_S)
    # Seconds a firmware just flashed has to answer. One whose port does not work gives the hand
    # back to the bootloader after `NO_CONTACT_S`: wait for that, rather than ask for a hand on
    # the reset button while the board is about to come back by itself.
    START_S = p.NO_CONTACT_S + 3 * BOOT_S
    # Seconds the bootloader has to show its port once asked for.
    PORT_S = 15
    # Seconds the port of the bootloader must stay to count: it may show during a reset.
    SETTLE_S = 3

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

    def _wait(self, role: str, seconds: float, present: bool = True) -> Port | None:
        """Waits for the port of `role` to be there (or, with `present` false, to be gone)."""
        deadline = time.monotonic() + seconds
        while True:
            port = self._find(role)
            if (port is not None) == present:
                return port
            if time.monotonic() >= deadline:
                return port if present else None
            time.sleep(0.25)

    def _open(self, port: Port) -> p.Device | None:
        """A device that answers INFO on `port`."""
        try:
            device = p.Device(p.Link(SerialTransport(port.device)))
            device.info(timeout=1.0, attempts=2)
        except (p.Disconnected, p.NoAnswer):
            # The port is there before the firmware serves it, or it just went away.
            return None
        self.device = device
        return device

    def _await(self, seconds: float) -> str | None:
        """Waits for the board to settle: "runner" when the firmware answers (`self.device` is
        then set), "bootloader" when the board sits in its bootloader, None when neither happens
        in `seconds`."""
        deadline = time.monotonic() + seconds
        in_bootloader_since = None
        while time.monotonic() < deadline:
            if (port := self._find("runner")) is not None:
                in_bootloader_since = None
                if self._open(port) is not None:
                    return "runner"
            elif self._find("bootloader") is not None:
                now = time.monotonic()
                in_bootloader_since = in_bootloader_since or now
                if now - in_bootloader_since >= self.SETTLE_S:
                    return "bootloader"
            else:
                in_bootloader_since = None
            time.sleep(0.25)
        return None

    def close(self) -> None:
        if self.device is not None:
            self.device.link.close()
            self.device = None

    def open(self) -> p.Device:
        if self.device is None and self._await(3) != "runner":
            ports = ", ".join(f"{port.device} ({port.role})" for port in list_ports()) or "none"
            raise SetupError(
                f"no board running the runner firmware (serial ports: {ports}): flash it by passing "
                "its images, after a double tap on the reset button if the board runs something else"
            )
        return self.device

    def recover(self) -> p.Device:
        self.close()
        self.log(f"  waiting up to {self.RESTART_S} s for the firmware to restart")
        state = self._await(self.RESTART_S)
        if state == "runner":
            return self.device
        if state == "bootloader":
            if self.image is None:
                raise FirmwareDead(
                    "the firmware gave the hand back to the bootloader (it crashed several times in a "
                    "row): the board is in its bootloader; pass the images to flash it again"
                )
            self.log("  the board is in its bootloader: flashing the image again")
            self.flash(self.image)
            return self.open()
        raise Unreachable(
            f"the board does not answer and did not restart in {self.RESTART_S} s: double-tap its reset "
            "button (the orange LED fades in and out in the bootloader), then run again"
        )

    def enter_bootloader(self) -> Port:
        """Brings the board into its bootloader, from whatever answers on its port."""
        if (port := self._find("bootloader")) is not None:
            return port
        if self.device is not None or ((port := self._find("runner")) is not None and self._open(port)):
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

    def _bossac(self, port: Port, arguments: list[str]) -> None:
        # bossac takes the name of the port without /dev/.
        command = [find_bossac(self.bossac), "-d", f"--port={Path(port.device).name}", "-U", *arguments]
        self.log(f"  {' '.join(command)}")
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            raise Unreachable("bossac did not finish in 180 s: double-tap the reset button and run again") from None
        if done.returncode != 0:
            tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-15:])
            raise SetupError(f"bossac failed ({done.returncode}):\n{tail}")

    def flash(self, image: Image) -> None:
        find_bossac(self.bossac)
        port = self.enter_bootloader()
        binary = image.path.with_name(image.path.name + ".bin")
        binary.write_bytes(image.binary)
        # The upload pattern of ArduinoCore-mbed's platform.txt (tools.bossac.upload.pattern), plus
        # the verification: bossac addresses the application from 0 (the bootloader adds 0x10000
        # to every address it is given, so it cannot be written), writes, reads the checksum of
        # what was written back, and resets.
        self.log(f"  flashing {image.path.name} ({len(image.binary)} bytes)")
        self._bossac(port, ["-i", "-e", "-w", "-v", str(binary), "-R"])
        self.image = image
        # The bootloader goes away with the reset; then either the firmware answers, or it gives
        # the hand back to the bootloader by itself.
        self._wait("bootloader", self.PORT_S, present=False)
        state = self._await(self.BOOT_S)
        if state is None:
            self.log(f"  the firmware does not answer yet: waiting up to {self.START_S} s for it, or for the bootloader")
            state = self._await(self.START_S - self.BOOT_S)
        if state == "bootloader":
            raise FirmwareDead(
                f"{image.path.name} was written and verified, but its firmware does not work on this board: "
                "it never answered and gave the hand back to the bootloader, where the board now waits. "
                "No hand is needed on the board; the firmware is"
            )
        if state is None:
            raise Unreachable(
                f"{image.path.name} was written but the firmware does not answer, and the board did not "
                f"return to its bootloader in {self.START_S} s: double-tap the reset button"
            )

    def leave_bootloader(self) -> bool:
        """Makes a board that sits in its bootloader start its application (`bossac -R`). False if
        the firmware does not answer afterwards: the bootloader stays when there is no application."""
        port = self._find("bootloader")
        if port is None:
            raise SetupError("the board is not in its bootloader")
        self._bossac(port, ["-R"])
        self._wait("bootloader", self.PORT_S, present=False)
        return self._await(self.BOOT_S) == "runner"


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
