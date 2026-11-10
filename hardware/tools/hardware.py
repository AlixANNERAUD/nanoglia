"""Runs the golden cases of testdata/ on an Arduino Nano 33 BLE over its USB serial port, compares
the outputs as crates/nanoglia/tests/golden.rs does, and writes a results file with the time of
each operator. documentation/how-to-guides/run-the-golden-cases-on-a-board.md is the guide.

    uv run --locked --project tools python -I tools/hardware.py --image <image-1.elf> --image <image-2.elf>

flashes each image through the bootloader of the board (bossac) and runs the cases its firmware
can run: every single-operator case with the first, each MLPerf Tiny model with the image it is
linked into. Without --image, the firmware found on the board is used.

Other devices, with the same code on the host: `--device emulated --elf <firmware.elf>` runs the
emulated runner (hardware/emulated) under Unicorn; `--device simulated` (or `--dry-run`) a
simulated device that answers with the expected outputs, to check the tool itself.

`--list-ports` lists the serial ports and what answers on each; `--info` prints what the firmware
says of itself; `--bootloader` resets the board into its bootloader and `--leave-bootloader`
starts its application again; `--self-test` makes the firmware panic, fault and hang, and checks
that the board comes back each time.

Exit codes: 0, every case passed (cases skipped for lack of memory are listed); 1, a case failed
or a check of the run did; 2, the host lacks something (a port, bossac, an image); 3, the board
stopped answering and could not be brought back: it needs a double tap on its reset button; 4,
the firmware does not work on the board, which went back to its bootloader by itself: nothing to
do by hand, the firmware is to fix.
`NANOGLIA_GOLDEN_DIR` (an absolute path) runs the cases of that directory instead of testdata/.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nanoglia_hardware import protocol as p  # noqa: E402
from nanoglia_hardware.board import (  # noqa: E402
    EmulatedBoard,
    FirmwareDead,
    Image,
    SerialBoard,
    SetupError,
    SimulatedBoard,
    Unreachable,
    list_ports,
)
from nanoglia_hardware.cases import results_json, run_cases  # noqa: E402
from nanoglia_hardware.shared import REPO, golden  # noqa: E402
from nanoglia_hardware.simulated import SimulatedDevice  # noqa: E402

# The models linked into each image of the board firmware (the `model-*` features the task
# `build-hardware-nano33ble` of mise.toml gives each), for the simulated device.
IMAGE_MODELS = [
    ["kws_ref_model.tflite", "pretrainedResnet_quant.tflite", "ad01_int8.tflite"],
    ["vww_96_int8.tflite", "str_ww_ref_model.tflite"],
]


def golden_oracle(root: Path):
    """The oracle of the simulated device: the expected outputs of the golden case that has this
    model and these inputs."""
    known = {}
    pending = golden.load_pending()
    for name, directory in golden.find_cases(root):
        spec = json.loads((directory / "case.json").read_text())
        model = zlib.crc32((REPO / spec["model"]).read_bytes())
        inputs = tuple(zlib.crc32((directory / f"input_{i}.bin").read_bytes()) for i in range(len(spec["inputs"])))
        outputs = [(directory / f"output_{i}.bin").read_bytes() for i in range(len(spec["outputs"]))]
        if name in pending:
            # A pending case differs from LiteRT on the comparison only: so does the simulation.
            tolerance = float(spec.get("tolerance", 0.0))
            outputs[0] = golden.corrupt(spec["outputs"][0]["dtype"], outputs[0], tolerance) or outputs[0]
        known[(model, inputs)] = (outputs, [golden.expected_kernel(name) or "simulated/none"])

    def oracle(model: bytes, inputs: list[bytes]):
        key = (zlib.crc32(model), tuple(zlib.crc32(data) for data in inputs))
        return known.get(key, (1, "invalid encoding: not a golden case"))

    return oracle


def simulated_images() -> list[list[tuple[str, bytes]]]:
    directory = REPO / "testdata" / "mlperf-tiny"
    return [[(name, (directory / name).read_bytes()) for name in names] for names in IMAGE_MODELS]


def self_test(board, log) -> bool:
    """Makes the firmware crash in each way it can and checks that the board comes back and says
    what happened."""
    ok = True
    steps = [
        (p.PANIC, "panic", "panic"),
        (p.HARD_FAULT, "hard fault", "fault"),
        (p.HANG, "hang (the watchdog of the firmware ends it, after about 20 s on a board)", "watchdog"),
    ]
    for fault, label, expected in steps:
        log(f"self-test: {label}")
        device = board.open()
        try:
            device.fault(fault)
        except (p.NoAnswer, p.Disconnected):
            # The answer may be lost with the port.
            pass
        info = board.recover().info()
        if info.crash == expected:
            log(f"  ok: {info.describe_crash()}")
        else:
            log(f"  FAIL: expected a restart after a {expected}, the firmware reports {info.crash!r} {info.crash_message!r}")
            ok = False
    log("self-test: reset into the firmware")
    try:
        board.open().reset(p.APPLICATION)
    except (p.NoAnswer, p.Disconnected):
        pass
    info = board.recover().info()
    if info.crash is None:
        log("  ok: the firmware answers again, without a crash to report")
    else:
        log(f"  FAIL: the firmware reports a {info.crash} after a reset the host asked for")
        ok = False
    return ok


def revision() -> str:
    try:
        done = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=12", "--exclude=*"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["serial", "emulated", "simulated"], default="serial")
    parser.add_argument("--dry-run", action="store_true", help="the same as --device simulated")
    parser.add_argument("--port", help="serial port of the board, when several are plugged in")
    parser.add_argument("--bossac", help=f"Arduino's bossac ({'$BOSSAC'}, the path and the Arduino tools by default)")
    parser.add_argument("--image", action="append", default=[], type=Path, help="firmware image (ELF or .bin) to flash and run; repeat for several")
    parser.add_argument("--elf", type=Path, help="the emulated runner, for --device emulated")
    parser.add_argument("--filter", default="", help="run only the cases whose name contains this")
    parser.add_argument("--repeat", type=int, default=1, help="run each case this many times and keep the fastest (default 1)")
    parser.add_argument("--results", type=Path, help="write the results to this JSON file")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list-ports", action="store_true", help="list the serial ports and exit")
    action.add_argument("--info", action="store_true", help="print what the firmware says of itself and exit")
    action.add_argument("--bootloader", action="store_true", help="reset the board into its bootloader and exit")
    action.add_argument("--leave-bootloader", action="store_true", help="start the application of a board that sits in its bootloader")
    action.add_argument("--flash-only", action="store_true", help="flash the last --image and exit")
    action.add_argument("--self-test", action="store_true", help="check that the board comes back from a panic, a fault and a hang")
    args = parser.parse_args(argv)
    if args.dry_run:
        args.device = "simulated"
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    log = lambda message: print(message, flush=True)  # noqa: E731

    if args.list_ports:
        ports = list_ports()
        for port in ports:
            identifiers = f"{port.vid:04x}:{port.pid:04x}" if port.vid is not None else "no USB id"
            log(f"{port.device}  {identifiers}  {port.role}  {port.product} {port.serial_number}".rstrip())
        if not ports:
            log("no serial port")
        return 0

    custom = os.environ.get("NANOGLIA_GOLDEN_DIR")
    root = Path(custom) if custom else REPO / "testdata"
    try:
        if args.device == "serial":
            board = SerialBoard(args.port, args.bossac, log)
            images = [Image.load(path) for path in args.image]
        elif args.device == "emulated":
            if args.elf is None:
                parser.error("--device emulated needs --elf")
            board = EmulatedBoard(args.elf, log)
            images = []
        else:
            images = simulated_images()
            board = SimulatedBoard(SimulatedDevice(golden_oracle(root), images[0]), log)
            log("simulated device: its outputs are the expected ones, this run checks the host tool only")

        if (args.bootloader or args.leave_bootloader) and args.device != "serial":
            parser.error("--bootloader and --leave-bootloader are for a board (--device serial)")
        if args.bootloader:
            port = board.enter_bootloader()
            log(f"the board is in its bootloader on {port.device}")
            return 0
        if args.leave_bootloader:
            if board.leave_bootloader():
                log(f"the firmware answers: {board.open().info().firmware}")
                return 0
            log("the bootloader was told to start the application, and no runner firmware answers: flash one")
            return 1
        if args.flash_only:
            if not images:
                parser.error("--flash-only needs --image")
            board.flash(images[-1])
            log(f"flashed: {board.open().info().firmware}")
            return 0
        if args.info:
            info = board.open().info()
            for key, value in vars(info).items():
                log(f"{key}: {value}")
            return 0
        if args.self_test:
            if images:
                board.flash(images[0])
            return 0 if self_test(board, log) else 1

        report = run_cases(board, images, root, args.filter, args.repeat, log)
    except SetupError as error:
        log(f"error: {error}")
        return 2
    except Unreachable as error:
        log(f"error: {error}")
        return 3
    except FirmwareDead as error:
        log(f"error: {error}")
        return 4
    except (p.NoAnswer, p.Disconnected, p.ProtocolError) as error:
        log(f"error: the device stopped answering: {error}")
        return 3
    finally:
        if "board" in locals():
            board.close()

    counts = report.counts()
    log(
        f"{counts['cases']} cases: {counts['ok']} ok, {counts['pending']} pending, {counts['skipped']} skipped, "
        f"{counts['failed']} failed; {counts['timed']} timed with their model in flash"
    )
    stack = max((result.stack_peak or 0 for result in report.results), default=0)
    arena = max((result.arena_used or 0 for result in report.results), default=0)
    log(f"peak stack {stack} B, largest arena {arena} B")
    if args.results:
        date = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(results_json(report, args.device, date, revision()), indent=1) + "\n")
        log(f"results written to {args.results}")
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
