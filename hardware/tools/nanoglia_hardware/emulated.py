"""The emulated runner (hardware/emulated) under the Unicorn engine, behind the transport
interface: the code of the board firmware, minus the board, on an emulated Cortex-M4F.

The firmware talks to the registers described in hardware/emulated/src/main.rs. It runs when the
host writes to it or waits for it, until it waits for the host in `emulated_wait`, where a hook
stops the emulation. A reset restarts it from its reset vector with the RAM as it was, as on a
chip; a reset into the bootloader leaves it stopped until `flashed()`.

The emulator checks instructions, not timing: the cycle counter moves by `CYCLES_STEP` at each
reading, so that every count is a known number, not a time.
"""

from collections import deque
from pathlib import Path

from unicorn import UC_ARCH_ARM, UC_HOOK_CODE, UC_MODE_THUMB, Uc, UcError
from unicorn.arm_const import UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_SP, UC_CPU_ARM_CORTEX_M4

from . import protocol as p
from .shared import REPO  # noqa: F401  (puts emulation/tools on the module path)

from run import SCS, Firmware  # noqa: E402  (emulation/tools/run.py)

REGISTERS = 0x4000_0000
RX_READY, RX_DATA, TX_DATA, CYCLES, RESET, RETAINED, TIME_MS, INJECT = range(0, 0x20, 4)
# What the cycle counter moves by at each reading.
CYCLES_STEP = 1000
# Wall time a firmware may run between two waits, in seconds: longer is a hang.
RUN_TIMEOUT_S = 120


class EmulatedDevice:
    """The emulated firmware and its transport in one."""

    # The firmware runs inside `write` and `read`: an empty read means nothing will come.
    instant = True

    def __init__(self, elf: Path):
        self.fw = Firmware.load(elf)
        self.wait_address = self.fw.symbols["emulated_wait"] & ~1
        self.flash = self.fw.region("emulator_flash")
        self.ram = self.fw.region("emulator_ram")
        self.uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB, cpu=UC_CPU_ARM_CORTEX_M4)
        if self.uc.ctl_get_cpu_model() != UC_CPU_ARM_CORTEX_M4:
            raise RuntimeError("Unicorn does not run its Cortex-M4 model")
        for base, size in (self.flash, self.ram, SCS):
            self.uc.mem_map(base, size)
        self.uc.mmio_map(REGISTERS, 0x1000, self._read, None, self._write, None)
        self.uc.hook_add(UC_HOOK_CODE, self._wait, begin=self.wait_address, end=self.wait_address)
        for address, data in self.fw.segments:
            self.uc.mem_write(address, data)
        self.incoming: deque[int] = deque()
        self.outgoing = bytearray()
        self.cycles = 0
        self.time_ms = 0
        self.retained = 0
        # Read by the firmware while it starts: 1 makes it panic there.
        self.inject = 0
        self.reset_request = 0
        self.resets: list[int] = []
        self.bootloader = False
        self.resume: tuple[int, int] | None = None
        self.power_on()

    # --- The registers.

    def _read(self, uc, offset, size, user):
        if offset == RX_READY:
            return 1 if self.incoming else 0
        if offset == RX_DATA:
            return self.incoming.popleft() if self.incoming else 0
        if offset == CYCLES:
            self.cycles = (self.cycles + CYCLES_STEP) & 0xFFFF_FFFF
            return self.cycles
        if offset == RETAINED:
            return self.retained
        if offset == TIME_MS:
            return self.time_ms & 0xFFFF_FFFF
        if offset == INJECT:
            return self.inject
        return 0

    def _write(self, uc, offset, size, value, user):
        if offset == TX_DATA:
            self.outgoing.append(value & 0xFF)
        elif offset == RESET:
            self.reset_request = value
        elif offset == RETAINED:
            self.retained = value & 0xFF

    def _wait(self, uc, address, size, user):
        # The firmware has nothing to do until the host moves. The hook is on the first
        # instruction of `emulated_wait`: leave as if the function had returned.
        self.resume = (uc.reg_read(UC_ARM_REG_LR), uc.reg_read(UC_ARM_REG_SP))
        uc.emu_stop()

    # --- The chip.

    def power_on(self) -> None:
        """A power-on: nothing survives, the firmware starts."""
        self.uc.mem_write(self.ram[0], bytes(self.ram[1]))
        self.retained = 0
        self.bootloader = False
        self._restart()
        self._run()

    def flashed(self) -> None:
        """A firmware was flashed (the same: there is one image) and started."""
        self.bootloader = False
        self.incoming.clear()
        self.outgoing.clear()
        self._restart()
        self._run()

    def _restart(self) -> None:
        # Unicorn does not take the reset exception: load SP and PC from the vector table. The
        # RAM is left as it is, as a reset leaves it.
        vectors = self.uc.mem_read(self.flash[0], 8)
        self.resume = (int.from_bytes(vectors[4:8], "little"), int.from_bytes(vectors[0:4], "little"))

    def _run(self) -> None:
        """Runs the firmware until it waits for the host, through its resets."""
        while not self.bootloader:
            pc, sp = self.resume
            self.resume = None
            self.uc.reg_write(UC_ARM_REG_SP, sp)
            try:
                self.uc.emu_start(pc | 1, 0xFFFF_FFFF, timeout=RUN_TIMEOUT_S * 1_000_000)
            except UcError as error:
                at = self.uc.reg_read(UC_ARM_REG_PC)
                raise p.Disconnected(f"emulator: {error} at pc=0x{at:08x}") from None
            if self.resume is None:
                raise p.Disconnected(f"emulator: the firmware ran {RUN_TIMEOUT_S} s without waiting for the host")
            if not self.reset_request:
                return
            # A reset: what the host sent and did not read is lost with the port.
            self.resets.append(self.reset_request - 1)
            self.bootloader = self.reset_request - 1 == p.BOOTLOADER
            self.reset_request = 0
            self.incoming.clear()
            self._restart()

    # --- Transport.

    def write(self, data: bytes) -> None:
        if self.bootloader:
            raise p.Disconnected("emulated device: in the bootloader")
        self.incoming.extend(data)
        self._run()

    def read(self, size: int, timeout: float) -> bytes:
        if not self.outgoing and not self.bootloader:
            # The host waits: time passes for the firmware, whose timeouts may fire.
            self.time_ms += max(int(timeout * 1000), 1)
            self._run()
        data = bytes(self.outgoing[:size])
        del self.outgoing[:size]
        return data

    def close(self) -> None:
        pass
