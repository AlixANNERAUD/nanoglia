"""Runs every golden case of testdata/ on the test-runner firmware of emulation/ (one ELF per
architecture) under the Unicorn engine, and compares the outputs as
crates/nanoglia/tests/golden.rs does: same tolerance rules (testdata/README.md), same pending list
(testdata/pending.json), a pending case that passes fails the run.

For each case, a fresh emulated Cortex-M gets the firmware's segments, the case (header, model,
inputs) at `_case_start`, and a painted RAM and arena; the firmware runs from its reset handler
until it writes `EMULATION_STATUS`, then the outputs are read back from the arena. Painting the
RAM gives the peak stack of each case and catches a stack that ran into the static data.

With `--expect-dsp`, every instruction of the Arm DSP extension in the ELF (found with
arm-none-eabi-objdump) gets a hook that fires once: the run fails unless a packed-SIMD
instruction (`smlad`, `sxtb16`, `pkhbt`...) of a CMSIS-NN function (`arm_*`) executed. GCC emits
some DSP instructions (`smlabb`...) from CMSIS-NN's portable C code too, so only these, which
CMSIS-NN writes as intrinsics under `ARM_MATH_DSP`, prove that its optimized code paths are
compiled in and run. With `--expect-no-dsp`, for the builds without the extension (Cortex-M0+,
Cortex-M33 with `-C target-feature=-dsp`), the run fails if the ELF holds any DSP instruction, so
none can execute; the hooks report which ones did.

Every run must have at least one case. It also corrupts the expected output of a passing case
(beyond its tolerance) and fails unless the comparison catches it. A pending case
(testdata/pending.json) must run to completion and fail only on the comparison of its outputs: a
fault, an error, a timeout or another check that fails is a failure.

This is an instruction-set emulator: it checks results, not timing, caches or flash wait states,
and Unicorn takes alignment faults on its Cortex-M0 only: an unaligned `ldm` on the Cortex-M4 or
M33, or a misaligned load on a RISC-V core without support for it, faults on a real core, runs
here and goes unnoticed.

    uv run --locked --project tools python -I tools/run.py --cpu cortex-m4 --expect-dsp <firmware.elf>

`NANOGLIA_GOLDEN_DIR` (an absolute path) runs the cases of that directory instead of testdata/,
as for golden.rs.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from elftools.elf.elffile import ELFFile
from unicorn import (
    UC_ARCH_ARM,
    UC_ARCH_RISCV,
    UC_HOOK_CODE,
    UC_HOOK_MEM_WRITE,
    UC_MODE_RISCV32,
    UC_MODE_THUMB,
    Uc,
    UcError,
)
from unicorn.arm_const import (
    UC_ARM_REG_PC,
    UC_ARM_REG_SP,
    UC_CPU_ARM_CORTEX_M0,
    UC_CPU_ARM_CORTEX_M4,
    UC_CPU_ARM_CORTEX_M33,
)
from unicorn.riscv_const import UC_CPU_RISCV32_SIFIVE_E31, UC_RISCV_REG_PC

REPO = Path(__file__).resolve().parents[2]

# Not `UC_MODE_THUMB | UC_MODE_MCLASS`: that mode selects Unicorn's Cortex-M33 whatever CPU model is asked for.
# `run` checks the model Unicorn settled on.
ARM = (UC_ARCH_ARM, UC_MODE_THUMB)
# CPU name: (architecture, mode, CPU model). Unicorn 2.1 has no Cortex-M0+ model: its Cortex-M0 runs
# the same Armv6-M instruction set (the M0+ differs by its pipeline, the optional MPU and VTOR, none
# of which the runner uses). Its closest RISC-V model to the ESP32-C3's RV32IMC is the SiFive E31
# (RV32IMAC); the firmware, built for riscv32imc, has no atomic instruction.
CPUS = {
    "cortex-m0": (*ARM, UC_CPU_ARM_CORTEX_M0),
    "cortex-m4": (*ARM, UC_CPU_ARM_CORTEX_M4),
    "cortex-m33": (*ARM, UC_CPU_ARM_CORTEX_M33),
    "riscv32": (UC_ARCH_RISCV, UC_MODE_RISCV32, UC_CPU_RISCV32_SIFIVE_E31),
}

# System control space (CPACR, SCB...): plain memory, as Unicorn does not model it.
SCS = (0xE000_0000, 1024 * 1024)
PATTERN = 0xA5C3_5AE1
MAGIC = int.from_bytes(b"NGCS", "little")
MAX_OUTPUTS = 16
TIMEOUT_S = 60

# Status values of the firmware (`Status` in src/main.rs).
DONE, ERROR, PANIC, FAULT = 1, 2, 3, 4

# Instructions of the Armv7E-M / Armv8-M DSP extension, absent from the base instruction set.
DSP_MNEMONICS = frozenset(
    """
    pkhbt pkhtb qadd qadd16 qadd8 qasx qdadd qdsub qsax qsub qsub16 qsub8 sadd16 sadd8 sasx sel
    shadd16 shadd8 shasx shsax shsub16 shsub8 smlabb smlabt smlatb smlatt smlad smladx smlalbb
    smlalbt smlaltb smlaltt smlald smlaldx smlawb smlawt smlsd smlsdx smlsld smlsldx smmla smmlar
    smmls smmlsr smmul smmulr smuad smuadx smulbb smulbt smultb smultt smulwb smulwt smusd smusdx
    ssat16 ssax ssub16 ssub8 sxtab sxtab16 sxtah sxtb16 uadd16 uadd8 uasx uhadd16 uhadd8 uhasx
    uhsax uhsub16 uhsub8 umaal uqadd16 uqadd8 uqasx uqsax uqsub16 uqsub8 usad8 usada8 usat16 usax
    usub16 usub8 uxtab uxtab16 uxtah uxtb16
    """.split()
)
# The packed-SIMD ones CMSIS-NN uses through its intrinsics (arm_nn_math_types.h, `ARM_MATH_DSP`).
# GCC does not emit them from the portable C code, unlike `smlabb` or `smulbb`, so their execution
# in an `arm_*` function shows that CMSIS-NN's DSP code paths run.
PACKED_SIMD = frozenset("smlad smladx sxtb16 sxtab16 pkhbt pkhtb sadd16".split())
assert PACKED_SIMD <= DSP_MNEMONICS
# Bits of `Report.truncated` (src/main.rs).
TRUNCATED_KERNELS, TRUNCATED_MESSAGE = 1, 2


def align(value: int, to: int) -> int:
    return (value + to - 1) // to * to


@dataclass
class Firmware:
    """The parts of the ELF a run needs."""

    path: Path
    symbols: dict[str, int]
    segments: list[tuple[int, bytes]]
    entry: int
    # Address -> (function, mnemonic) of each DSP instruction, with --expect-dsp.
    dsp: dict[int, tuple[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Firmware":
        with open(path, "rb") as f:
            elf = ELFFile(f)
            table = elf.get_section_by_name(".symtab")
            symbols = {s.name: s["st_value"] for s in table.iter_symbols() if s.name}
            segments = [
                (seg["p_paddr"], seg.data())
                for seg in elf.iter_segments()
                if seg["p_type"] == "PT_LOAD" and seg["p_filesz"] > 0
            ]
            entry = elf.header["e_entry"]
        return cls(path, symbols, segments, entry)

    def region(self, name: str) -> tuple[int, int]:
        start = self.symbols[f"_{name}_start"]
        return start, self.symbols[f"_{name}_end"] - start

    def find_dsp(self) -> None:
        """Lists the DSP instructions of the ELF with arm-none-eabi-objdump."""
        objdump = shutil.which("arm-none-eabi-objdump")
        if objdump is None:
            sys.exit("--expect-dsp and --expect-no-dsp need arm-none-eabi-objdump (package gcc-arm-none-eabi)")
        text = subprocess.run(
            [objdump, "-d", "--no-show-raw-insn", str(self.path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        function = "?"
        for line in text.splitlines():
            if m := re.match(r"^[0-9a-f]+ <(.+)>:$", line):
                function = m.group(1)
            elif m := re.match(r"^\s+([0-9a-f]+):\s+([a-z0-9]+)(?:\.[nw])?\s", line):
                mnemonic = m.group(2)
                # IT blocks add a condition suffix (`smladeq`).
                base = re.sub(r"(eq|ne|cs|cc|hs|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al)$", "", mnemonic)
                if mnemonic in DSP_MNEMONICS or base in DSP_MNEMONICS:
                    self.dsp[int(m.group(1), 16)] = (function, base if base in DSP_MNEMONICS else mnemonic)


@dataclass
class Run:
    status: int
    message: str
    outputs: list[bytes]
    kernels: list[str]
    arena_used: int
    stack_bytes: int
    dsp_executed: set[int]
    # The kernel names did not fit in the report.
    kernels_truncated: bool = False


def case_image(model: bytes, inputs: list[bytes]) -> bytes:
    """The case region: header, then the model and the inputs, each 16-byte aligned."""
    blobs = [model, *inputs]
    offset = align(16 + 8 * len(inputs), 16)
    spans = []
    for blob in blobs:
        spans.append((offset, len(blob)))
        offset = align(offset + len(blob), 16)
    image = bytearray(offset)
    words = [MAGIC, *spans[0], len(inputs)]
    for span in spans[1:]:
        words.extend(span)
    for i, w in enumerate(words):
        image[4 * i : 4 * i + 4] = w.to_bytes(4, "little")
    for (start, length), blob in zip(spans, blobs):
        image[start : start + length] = blob
    return bytes(image)


def run(fw: Firmware, cpu: tuple[int, int, int], case: bytes, seen: set[int] = frozenset()) -> Run:
    """Runs one case; hooks the DSP instructions of `fw.dsp` not in `seen` (executed earlier)."""
    sym = fw.symbols
    flash, ram, arena, region = (fw.region(n) for n in ("emulator_flash", "emulator_ram", "arena", "case"))
    if len(case) > region[1]:
        return Run(0, f"case of {len(case)} bytes, region of {region[1]}", [], [], 0, 0, set())

    arch, mode, model = cpu
    arm = arch == UC_ARCH_ARM
    uc = Uc(arch, mode, cpu=model)
    if uc.ctl_get_cpu_model() != model:
        raise RuntimeError(f"Unicorn runs CPU model {uc.ctl_get_cpu_model()}, not {model}")
    for base, size in (flash, ram, arena, region, *([SCS] if arm else [])):
        uc.mem_map(base, size)
    pattern = PATTERN.to_bytes(4, "little")
    uc.mem_write(ram[0], pattern * (ram[1] // 4))
    uc.mem_write(arena[0], pattern * (arena[1] // 4))
    for address, data in fw.segments:
        uc.mem_write(address, data)
    uc.mem_write(region[0], case)

    status_addr = sym["EMULATION_STATUS"]
    finished = {}

    def on_status(uc, access, address, size, value, user):
        # The start-up code zeroes .bss, EMULATION_STATUS included: only a non-zero value ends it.
        if value != 0:
            finished["status"] = value
            uc.emu_stop()

    uc.hook_add(UC_HOOK_MEM_WRITE, on_status, begin=status_addr, end=status_addr + 3)

    executed: set[int] = set()
    hooks = {}

    def on_dsp(uc, address, size, user):
        # Once per instruction: the hook is deleted after its first hit.
        executed.add(address)
        if address in hooks:
            uc.hook_del(hooks.pop(address))

    # Only the instructions no earlier case executed: a hook costs on every execution of a
    # translated block that holds it, even once deleted, and each case starts a fresh emulator.
    for address in fw.dsp.keys() - seen:
        hooks[address] = uc.hook_add(UC_HOOK_CODE, on_dsp, begin=address, end=address)

    if arm:
        # Unicorn does not take the reset exception: load SP and PC from the vector table.
        vectors = uc.mem_read(flash[0], 8)
        uc.reg_write(UC_ARM_REG_SP, int.from_bytes(vectors[0:4], "little"))
        start = int.from_bytes(vectors[4:8], "little") | 1
    else:
        # riscv-rt's `_start` sets up the stack itself.
        start = fw.entry
    pc = lambda: uc.reg_read(UC_ARM_REG_PC if arm else UC_RISCV_REG_PC)
    error = ""
    try:
        uc.emu_start(start, 0xFFFF_FFFF, timeout=TIMEOUT_S * 1_000_000)
    except UcError as e:
        error = f"emulator: {e} at pc=0x{pc():08x}"
    status = finished.get("status", 0)
    if not error and not status:
        error = f"emulator: stopped at pc=0x{pc():08x} (timeout?)"

    # The report: see `Report` in src/main.rs (#[repr(C)], u32 fields).
    base = sym["EMULATION_RESULT"]
    word = lambda offset: int.from_bytes(uc.mem_read(base + offset, 4), "little")
    count = word(0)
    outputs_at = 4
    arena_used_at = outputs_at + 8 * MAX_OUTPUTS
    kernels_len_at = arena_used_at + 4
    kernels_at = kernels_len_at + 4
    message_len_at = kernels_at + 4096
    message_at = message_len_at + 4
    truncated_at = message_at + 256
    truncated = word(truncated_at)
    kernels = bytes(uc.mem_read(base + kernels_at, min(word(kernels_len_at), 4096))).decode(errors="replace")
    message = bytes(uc.mem_read(base + message_at, min(word(message_len_at), 256))).decode(errors="replace")
    if truncated & TRUNCATED_MESSAGE:
        message += " (truncated)"
    outputs = []
    if status == DONE:
        for i in range(min(count, MAX_OUTPUTS)):
            address, length = word(outputs_at + 8 * i), word(outputs_at + 8 * i + 4)
            outputs.append(bytes(uc.mem_read(address, length)))

    # Stack painting: the lowest word above the static data that the run wrote.
    top = sym["_stack_start"]
    bottom = align(sym["__sheap"], 4)
    painted = bytes(uc.mem_read(bottom, top - bottom))
    lowest = top
    for offset in range(0, len(painted), 4):
        if painted[offset : offset + 4] != pattern:
            lowest = bottom + offset
            break
    if lowest == bottom:
        error = error or "the stack reached the static data (overflow)"
    return Run(
        status if not error else 0,
        error or message,
        outputs,
        kernels.splitlines(),
        word(arena_used_at),
        top - lowest,
        executed,
        bool(truncated & TRUNCATED_KERNELS),
    )


# --- The comparison of crates/nanoglia/tests/golden.rs, kept in step with it. ---

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


def check(name: str, spec: dict, result: Run, expected: list[bytes]) -> Failure | None:
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


def self_check(canary: tuple[str, dict, Run, list[bytes]]) -> bool:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", choices=sorted(CPUS), required=True)
    dsp = parser.add_mutually_exclusive_group()
    dsp.add_argument(
        "--expect-dsp", action="store_true", help="fail unless packed-SIMD DSP instructions of CMSIS-NN ran"
    )
    dsp.add_argument(
        "--expect-no-dsp",
        action="store_true",
        help="fail if the ELF holds an instruction of the Arm DSP extension, or if one executes",
    )
    parser.add_argument("--filter", default="", help="run only the cases whose name contains this")
    parser.add_argument("elf", type=Path)
    args = parser.parse_args()

    fw = Firmware.load(args.elf)
    if args.expect_dsp or args.expect_no_dsp:
        if CPUS[args.cpu][0] != UC_ARCH_ARM:
            parser.error("--expect-dsp and --expect-no-dsp are for the Arm CPUs")
        fw.find_dsp()
    custom = os.environ.get("NANOGLIA_GOLDEN_DIR")
    root = Path(custom) if custom else REPO / "testdata"
    cases = sorted(p.parent for p in root.rglob("case.json"))
    pending_reasons = load_pending()

    start = time.monotonic()
    failures, pending, passed = [], 0, 0
    max_stack, max_arena, slowest = 0, 0, (0.0, "")
    kernels_run: Counter[str] = Counter()
    dsp_executed: set[int] = set()
    # Passing cases for the self-check: the first with an integer output and a tolerance of 0,
    # where flipping one unit must fail, else the first float one.
    canaries = {}
    for directory in cases:
        name = directory.relative_to(root).as_posix()
        if args.filter not in name:
            continue
        spec = json.loads((directory / "case.json").read_text())
        model = (REPO / spec["model"]).read_bytes()
        inputs = [(directory / f"input_{i}.bin").read_bytes() for i in range(len(spec["inputs"]))]
        expected = [(directory / f"output_{i}.bin").read_bytes() for i in range(len(spec["outputs"]))]
        t0 = time.monotonic()
        result = run(fw, CPUS[args.cpu], case_image(model, inputs), dsp_executed)
        elapsed = time.monotonic() - t0
        slowest = max(slowest, (elapsed, name))
        max_stack = max(max_stack, result.stack_bytes)
        max_arena = max(max_arena, result.arena_used)
        kernels_run.update(result.kernels)
        dsp_executed |= result.dsp_executed
        error = check(name, spec, result, expected)
        reason = pending_reasons.get(name)
        if error is None and reason is None:
            print(f"ok       {name}")
            passed += 1
            exact = float(spec.get("tolerance", 0.0)) == 0 and all(
                o["dtype"] != "float32" for o in spec["outputs"]
            )
            if any(expected):
                canaries.setdefault(exact, (name, spec, result, expected))
        elif error is not None and reason is not None and error.comparison:
            pending += 1
            print(f"pending  {name}: {error.message} ({reason})")
        elif error is not None and reason is not None:
            print(f"FAIL     {name}: {error.message} (pending, but it must fail on the comparison only)")
            failures.append(name)
        elif error is None:
            print(f"FIXED    {name}: remove it from testdata/pending.json")
            failures.append(name)
        else:
            print(f"FAIL     {name}: {error.message}")
            failures.append(name)
    wall = time.monotonic() - start
    total = passed + pending + len(failures)
    # An entry of pending.json that names no case hides nothing, but it is stale.
    if not args.filter and not custom:
        names = {str(c.relative_to(root)) for c in cases}
        for stale in sorted(set(pending_reasons) - names):
            print(f"FAIL     {stale}: in testdata/pending.json, but there is no such case")
            failures.append(stale)

    print(f"{total} cases, {pending} pending, {len(failures)} failures")
    print(
        f"{args.cpu}: {wall:.1f} s wall time, slowest case {slowest[1]} ({slowest[0]:.2f} s), "
        f"peak stack {max_stack} B, peak arena {max_arena} B"
    )
    cmsis = sum(n for k, n in kernels_run.items() if k.startswith("cmsis-nn/"))
    print(f"operators run: {sum(kernels_run.values())}, {cmsis} by CMSIS-NN kernels")

    ok = not failures
    if total == 0:
        print(f"FAIL     no case{f' matches --filter {args.filter!r}' if args.filter else ''} in {root}")
        ok = False
    elif not args.filter and not custom and total <= 100:
        print(f"FAIL     test data not found: {total} cases in {root}")
        ok = False

    # Negative check: an expected output moved beyond the tolerance of a passing case must fail
    # the comparison, or the run proves nothing.
    canary = canaries.get(True) or canaries.get(False)
    if canary is not None:
        ok &= self_check(canary)
    elif passed:
        print("FAIL     self-check: no passing case has a non-zero expected output")
        ok = False

    if args.expect_dsp:
        cmsis_dsp = {a for a, (f, _) in fw.dsp.items() if f.startswith("arm_")}
        ran = dsp_executed & cmsis_dsp
        mnemonics = Counter(fw.dsp[a][1] for a in ran)
        functions = Counter(fw.dsp[a][0] for a in ran)
        packed = {a for a in ran if fw.dsp[a][1] in PACKED_SIMD}
        print(
            f"DSP instructions: {len(fw.dsp)} in the ELF, {len(cmsis_dsp)} in CMSIS-NN functions, "
            f"{len(ran)} of those executed in {len(functions)} functions, {len(packed)} packed-SIMD"
        )
        print("  executed mnemonics: " + ", ".join(f"{m} {n}" for m, n in mnemonics.most_common()))
        print("  functions: " + ", ".join(f for f, _ in functions.most_common(12)))
        if not packed:
            print(f"FAIL     no packed-SIMD instruction ({', '.join(sorted(PACKED_SIMD))}) of CMSIS-NN executed")
            ok = False
    if args.expect_no_dsp:
        print(f"DSP instructions: {len(fw.dsp)} in the ELF, {len(dsp_executed)} executed")
        for address, (function, mnemonic) in sorted(fw.dsp.items())[:12]:
            ran = "executed" if address in dsp_executed else "not executed"
            print(f"  0x{address:08x} {mnemonic} in {function} ({ran})")
        if fw.dsp:
            print("FAIL     a build without the DSP extension holds DSP instructions")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
