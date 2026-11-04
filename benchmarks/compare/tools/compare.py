"""Runs the whole engine comparison of benchmarks/compare and prints Markdown tables.

1. Builds the host benchmarks (one binary per engine, release profile of the workspace).
2. Runs every binary ROUNDS times (default 3), in a different order each round, and keeps the
   median inference time of each run.
3. Builds the keyword-spotting firmware for thumbv7em-none-eabihf (profile `firmware`), reads
   their sizes with arm-none-eabi-size, and runs each under the Cortex-M4 emulator of
   emulate.py: output check against the golden sample, peak stack by stack painting.

    mise run bench-compare                 # from anywhere in the repository
    ROUNDS=5 BENCH_MS=5000 mise run bench-compare

Raw results are also written to target/compare/.
"""

import os
import platform
import shutil
import statistics
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emulate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # benchmarks/compare
REPO = ROOT.parent.parent
OUT = ROOT / "target" / "compare"
TARGET = "thumbv7em-none-eabihf"

ENGINES = ["nanoglia", "tract", "MicroFlow", "embedded-nn", "ember-rs"]
BINARIES = {
    "nanoglia": "bench-nanoglia",
    "tract": "bench-tract",
    "MicroFlow": "bench-microflow",
    "embedded-nn": "bench-embedded-nn",
    "ember-rs": "bench-ember",
}
MODELS = ["kws_ref_model", "vww_96_int8", "pretrainedResnet_quant", "ad01_int8"]
FIRMWARE = [
    ("(no engine)", "fw-baseline", []),
    ("nanoglia, int8 only", "fw-nanoglia", []),
    ("nanoglia, default features", "fw-nanoglia", ["--features", "fw-nanoglia/full"]),
    ("MicroFlow", "fw-microflow", []),
    ("embedded-nn", "fw-embedded-nn", []),
    ("ember-rs", "fw-ember", []),
]
KWS_MODEL = REPO / "testdata/mlperf-tiny/kws_ref_model.tflite"
KWS_EXPECTED = REPO / "testdata/mlperf-tiny/golden/kws_ref_model/sample_0/output_0.bin"


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def sh(*cmd, **kw) -> str:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout.strip()


def cargo(*args):
    log("$ cargo", *args)
    subprocess.run(["cargo", *args], check=True, cwd=ROOT)


def environment() -> list[tuple[str, str]]:
    commit = sh("git", "rev-parse", "--short", "HEAD", cwd=REPO)
    if sh("git", "status", "--porcelain", "--untracked-files=no", cwd=REPO):
        commit += " (with uncommitted changes)"
    cpu = ""
    try:
        info = dict(
            line.split(":", 1) for line in sh("lscpu", env={**os.environ, "LC_ALL": "C"}).splitlines()
            if ":" in line
        )
        cpu = f"{info.get('Vendor ID', '?').strip()} {info.get('Model name', '').strip()}".strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    gcc = sh("arm-none-eabi-gcc", "--version").splitlines()[0]
    return [
        ("Date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
        ("Commit", f"`{commit}`"),
        ("OS", f"{platform.system()} {platform.release()} {platform.machine()}"),
        ("CPU", f"{cpu}, {os.cpu_count()} logical CPUs"),
        ("Rust", sh("rustc", "-V")),
        ("C compiler (firmware)", gcc),
        ("Emulator", f"Unicorn {emulate.unicorn_version()} (Cortex-M4)"),
        ("Time budget", f"{os.environ.get('BENCH_MS', '2000')} ms per model and run"),
    ]


def versions() -> list[tuple[str, str]]:
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    wanted = [
        "nanoglia",
        "tract-tflite",
        "tract-core",
        "microflow",
        "microflow-macros",
        "nalgebra",
        "embedded-nn",
        "embedded-nn-macros",
        "ember-infer-core",
        "ember-infer-macros",
        "ember-infer-ref",
        "embedded-alloc",
        "cortex-m-rt",
    ]
    rows = []
    for package in lock["package"]:
        if package["name"] in wanted:
            source = package.get("source", "path (this repository)")
            if source.startswith("registry+"):
                source = "crates.io"
            elif source.startswith("git+"):
                source = source.removeprefix("git+")
            rows.append((package["name"], f"{package['version']} ({source})"))
    return sorted(rows)


def host(rounds: int) -> tuple[dict, dict]:
    """Returns {(engine, model): [RESULT fields per round]} and {(engine, model): reason}."""
    cargo("build", "--release")
    results, skips = {}, {}
    raw = []
    for r in range(rounds):
        order = ENGINES[r % len(ENGINES) :] + ENGINES[: r % len(ENGINES)]
        for engine in order:
            binary = ROOT / "target" / "release" / BINARIES[engine]
            log(f"round {r + 1}/{rounds}: {engine}")
            out = subprocess.run(
                [str(binary)], check=True, text=True, capture_output=True, cwd=ROOT
            ).stdout
            for line in out.splitlines():
                raw.append(f"{r + 1}\t{line}")
                fields = line.split("\t")
                if fields[0] == "RESULT":
                    _, eng, model, samples, maxdiff, differing, total, top1, median, runs = fields
                    results.setdefault((eng, model), []).append(
                        dict(
                            samples=int(samples),
                            maxdiff=int(maxdiff),
                            differing=int(differing),
                            total=int(total),
                            top1=int(top1),
                            median=float(median),
                            runs=int(runs),
                        )
                    )
                elif fields[0] == "SKIP":
                    skips[(fields[1], fields[2])] = fields[3]
    (OUT / "host.tsv").write_text("\n".join(raw) + "\n")
    return results, skips


def size(elf: Path) -> tuple[int, int, int]:
    line = sh("arm-none-eabi-size", "-B", str(elf)).splitlines()[1].split()
    return int(line[0]), int(line[1]), int(line[2])


def firmware() -> list[dict]:
    rows = []
    for name, package, features in FIRMWARE:
        cargo("build", "--profile", "firmware", "--target", TARGET, "-p", package, *features)
        elf = OUT / f"{package}{'-full' if features else ''}.elf"
        shutil.copy(ROOT / "target" / TARGET / "firmware" / package, elf)
        text, data, bss = size(elf)
        log(f"emulating {elf.name}")
        run = emulate.run(str(elf))
        expected = KWS_EXPECTED.read_bytes()
        maxdiff = max(
            abs(int.from_bytes([a], "little", signed=True) - int.from_bytes([b], "little", signed=True))
            for a, b in zip(run.output, expected)
        )
        rows.append(
            dict(
                name=name,
                elf=elf.name,
                text=text,
                data=data,
                bss=bss,
                status=run.status,
                stack=run.stack_bytes,
                info=run.info,
                maxdiff=maxdiff if run.status == 1 and package != "fw-baseline" else None,
                error=run.error,
            )
        )
    return rows


def table(header: list[str], rows: list[list[str]], align: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def report(env, vers, results, skips, fw, rounds):
    print("### Environment\n")
    print(table(["", ""], [[k, v] for k, v in env], ["---", "---"]))
    print("\n### Versions\n")
    print(table(["Crate", "Version"], [[k, v] for k, v in vers], ["---", "---"]))

    print("\n### Correctness on the golden samples\n")
    print(
        "Max |difference| from the LiteRT reference output, in int8 units; elements that "
        "differ / all output elements of the 3 samples; samples whose top-1 class matches.\n"
    )
    rows = []
    for engine in ENGINES:
        row = [engine]
        for model in MODELS:
            runs = results.get((engine, model))
            if not runs:
                row.append(f"— ({skips.get((engine, model), 'not run')})")
                continue
            r = runs[0]
            top1 = "" if model == "ad01_int8" else f", top-1 {r['top1']}/{r['samples']}"
            row.append(f"{r['maxdiff']} ({r['differing']}/{r['total']}{top1})")
        rows.append(row)
    print(table(["Engine", *MODELS], rows, ["---"] + ["---:"] * len(MODELS)))

    print(f"\n### Host time per inference\n")
    print(
        f"Median of the {rounds} run medians, and the lowest and highest run median. "
        "A time is marked with its max |difference| when the outputs are not bit-exact.\n"
    )
    rows = []
    for engine in ENGINES:
        row = [engine]
        for model in MODELS:
            runs = results.get((engine, model))
            if not runs:
                row.append("—")
                continue
            medians = [r["median"] for r in runs]
            cell = f"{statistics.median(medians):.1f} µs ({min(medians):.0f}–{max(medians):.0f})"
            if runs[0]["maxdiff"]:
                cell += f" ⚠ Δ{runs[0]['maxdiff']}"
            row.append(cell)
        rows.append(row)
    print(table(["Engine", *MODELS], rows, ["---"] + ["---:"] * len(MODELS)))

    model_size = KWS_MODEL.stat().st_size
    print(f"\n### Keyword-spotting firmware (thumbv7em-none-eabihf)\n")
    print(
        f"arm-none-eabi-size (text includes the model: the .tflite file is {model_size} B; "
        "the ahead-of-time engines embed their own encoding of the weights). Peak stack by "
        "stack painting under the emulator. Output: max |difference| from the golden output "
        "of sample 0, computed by the firmware under the emulator.\n"
    )
    rows = []
    for r in fw:
        static = r["data"] + r["bss"]
        ok = r["status"] == 1
        rows.append(
            [
                r["name"],
                str(r["text"]),
                str(r["data"]),
                str(r["bss"]),
                str(static),
                str(r["stack"]) if ok else f"not measured ({r['error'] or 'did not finish'})",
                str(static + r["stack"]) if ok else "—",
                "—" if r["maxdiff"] is None else str(r["maxdiff"]),
            ]
        )
    print(
        table(
            ["Firmware", "text", "data", "bss", "Static RAM", "Peak stack", "RAM total", "Output max |Δ|"],
            rows,
            ["---"] + ["---:"] * 7,
        )
    )
    for r in fw:
        if r["info"]:
            print(f"\n{r['name']}: BENCH_INFO = {r['info']}")


def main():
    rounds = int(os.environ.get("ROUNDS", "3"))
    OUT.mkdir(parents=True, exist_ok=True)
    env, vers = environment(), versions()
    results, skips = host(rounds)
    fw = firmware()
    report(env, vers, results, skips, fw, rounds)


if __name__ == "__main__":
    main()
