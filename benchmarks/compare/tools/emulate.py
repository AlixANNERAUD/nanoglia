"""Runs a comparison firmware (thumbv7em-none-eabihf ELF) once under the Unicorn engine, emulating a
Cortex-M4 with the nRF52840's memory map (1 MiB of flash at 0, 256 KiB of RAM at 0x20000000).

Before the reset handler runs, every word of RAM is painted with a pattern. The firmware runs its
inference and writes `BENCH_STATUS`; then the output is read from `BENCH_OUTPUT`, and the peak
stack is the distance from the top of RAM (`_stack_start`) down to the lowest word that no longer
holds the pattern, above the end of the static data (`__sheap`). This is stack painting: it
counts the stack the run actually wrote, for this input, without interrupts.

This is an instruction-set emulator: it says nothing about time on the chip.

    uv run python -I emulate.py <firmware.elf> [expected_output.bin]
"""

import sys
from dataclasses import dataclass

from elftools.elf.elffile import ELFFile
from unicorn import (
    UC_ARCH_ARM,
    UC_HOOK_MEM_WRITE,
    UC_MODE_MCLASS,
    UC_MODE_THUMB,
    Uc,
    UcError,
)
from unicorn.arm_const import UC_ARM_REG_PC, UC_ARM_REG_SP, UC_CPU_ARM_CORTEX_M4

FLASH = (0x0000_0000, 1024 * 1024)
RAM = (0x2000_0000, 256 * 1024)
# System control space (CPACR, SCB...): plain memory, as Unicorn does not model it.
SCS = (0xE000_0000, 1024 * 1024)
PATTERN = 0xA5C3_5AE1
TIMEOUT_S = 600


@dataclass
class Run:
    status: int  # 1: output written, 2: engine error, 0: did not finish
    output: bytes
    info: int
    stack_bytes: int
    error: str


def unicorn_version() -> str:
    import unicorn

    return unicorn.__version__


def symbols(elf: ELFFile) -> dict[str, int]:
    table = elf.get_section_by_name(".symtab")
    return {s.name: s["st_value"] for s in table.iter_symbols() if s.name}


def run(path: str) -> Run:
    with open(path, "rb") as f:
        elf = ELFFile(f)
        sym = symbols(elf)
        segments = [
            (seg["p_paddr"], seg.data())
            for seg in elf.iter_segments()
            if seg["p_type"] == "PT_LOAD" and seg["p_filesz"] > 0
        ]

    uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS, cpu=UC_CPU_ARM_CORTEX_M4)
    for base, size in (FLASH, RAM, SCS):
        uc.mem_map(base, size)
    uc.mem_write(RAM[0], PATTERN.to_bytes(4, "little") * (RAM[1] // 4))
    for address, data in segments:
        uc.mem_write(address, data)

    status_addr = sym["BENCH_STATUS"]
    finished = {}

    def on_status(uc, access, address, size, value, user):
        # The start-up code zeroes .bss, BENCH_STATUS included: only a non-zero value ends the run.
        if value != 0:
            finished["status"] = value
            uc.emu_stop()

    uc.hook_add(UC_HOOK_MEM_WRITE, on_status, begin=status_addr, end=status_addr + 3)

    vectors = uc.mem_read(FLASH[0], 8)
    sp = int.from_bytes(vectors[0:4], "little")
    reset = int.from_bytes(vectors[4:8], "little")
    uc.reg_write(UC_ARM_REG_SP, sp)
    error = ""
    try:
        uc.emu_start(reset | 1, 0xFFFF_FFFF, timeout=TIMEOUT_S * 1_000_000)
    except UcError as e:
        error = f"{e} at pc=0x{uc.reg_read(UC_ARM_REG_PC):08x}"

    output = bytes(uc.mem_read(sym["BENCH_OUTPUT"], 12))
    info = int.from_bytes(uc.mem_read(sym["BENCH_INFO"], 4), "little") if "BENCH_INFO" in sym else 0
    error = error or ("" if finished else f"stopped at pc=0x{uc.reg_read(UC_ARM_REG_PC):08x}")

    # Stack painting: the lowest word above the static data that the run wrote.
    top = sym["_stack_start"]
    bottom = sym.get("__sheap", sym.get("__ebss"))
    ram = bytes(uc.mem_read(bottom, top - bottom))
    pattern = PATTERN.to_bytes(4, "little")
    lowest = top
    for offset in range(0, len(ram), 4):
        if ram[offset : offset + 4] != pattern:
            lowest = bottom + offset
            break
    if lowest == bottom:
        error = error or "the stack reached the static data (overflow)"
    return Run(finished.get("status", 0), output, info, top - lowest, error)


def main() -> None:
    result = run(sys.argv[1])
    print(f"status {result.status}, info {result.info}, peak stack {result.stack_bytes} B")
    print("output", list(int.from_bytes([b], "little", signed=True) for b in result.output))
    if result.error:
        print("error:", result.error)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "rb") as f:
            expected = f.read()
        diff = max(abs((a ^ 0x80) - (b ^ 0x80)) for a, b in zip(result.output, expected))
        print("max |diff| vs expected:", diff)


if __name__ == "__main__":
    main()
