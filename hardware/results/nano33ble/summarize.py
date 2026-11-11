"""Prints, as Markdown tables, the figures that documentation/reference/performance.md and
memory.md quote from a results file of the Nano 33 BLE.

    python3 -I hardware/results/nano33ble/summarize.py \
        hardware/results/nano33ble/nano33ble-repeat-5.json \
        hardware/results/nano33ble/nano33ble.json

The first file gives every figure; the second, optional, another run of the same images, to
show how far the cycle counts move from one run to the next. Only the cases whose "timing" is
true are used: their model is in flash, read in place. Standard library only.
"""

import collections
import json
import sys

MODELS = [
    "kws_ref_model",
    "vww_96_int8",
    "pretrainedResnet_quant",
    "ad01_int8",
    "str_ww_ref_model",
]


def load(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def timed(results):
    """{model: [case, ...]} of the cases timed with their model in flash, by sample."""
    by_model = collections.defaultdict(list)
    for case in results["cases"]:
        if case["timing"] and case["model_location"] == "flash":
            by_model[case["name"].split("/")[2]].append(case)
    for cases in by_model.values():
        cases.sort(key=lambda case: case["name"])
    return by_model


def ms(cycles, hz):
    return cycles * 1000 / hz


def main(paths):
    results = load(paths[0])
    images = results["firmware_images"]
    hz = images[0]["counter_hz"]
    keys = ("revision", "nanoglia", "build", "opt_level", "rustc", "clock", "settings")
    for key in keys:
        values = {image[key] for image in images}
        print(f"{key}: {' | '.join(sorted(values))}")
    print(f"counter_hz: {hz}, date: {results['date']}, summary: {results['summary']}")
    by_model = timed(results)
    other = timed(load(paths[1])) if len(paths) > 1 else {}

    print("\n## Inference and load, model in flash\n")
    print("| Model | Operators | Inference | Cycles | Load (`Graph::load_in`) |"
          " Spread of the 5 runs | Spread of the 3 samples | Other run |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in MODELS:
        cases = by_model[model]
        fastest = [case["compute_cycles"] for case in cases]
        runs = [max(c["compute_cycles_runs"]) - min(c["compute_cycles_runs"]) for c in cases]
        loads = [case["load_cycles"] for case in cases]
        sample_0 = cases[0]
        if model in other:
            moved = max(
                abs(a["compute_cycles"] - b["compute_cycles"])
                for a, b in zip(cases, other[model], strict=True)
            )
            other_text = f"{moved} cycles"
        else:
            other_text = "—"
        print(
            f"| `{model}` | {len(sample_0['operators'])} "
            f"| {ms(sample_0['compute_cycles'], hz):.2f} ms "
            f"| {sample_0['compute_cycles']} "
            f"| {ms(min(loads), hz):.1f} ms "
            f"| {max(runs)} cycles "
            f"| {max(fastest) - min(fastest)} cycles "
            f"({(max(fastest) - min(fastest)) / min(fastest) * 100:.3f} %) "
            f"| {other_text} |"
        )

    print("\n## Time by kernel, sample 0\n")
    print("| Model | Kernel | Operators | Time | Share |")
    print("| --- | --- | ---: | ---: | ---: |")
    for model in MODELS:
        case = by_model[model][0]
        total = sum(op["cycles"] for op in case["operators"])
        assert total == case["compute_cycles"], model
        cycles = collections.Counter()
        count = collections.Counter()
        for op in case["operators"]:
            cycles[op["kernel"]] += op["cycles"]
            count[op["kernel"]] += 1
        for kernel, kernel_cycles in cycles.most_common():
            print(
                f"| `{model}` | `{kernel}` | {count[kernel]} "
                f"| {ms(kernel_cycles, hz):.2f} ms | {kernel_cycles / total * 100:.1f} % |"
            )

    print("\n## Memory on the board, bytes\n")
    print("| Model | Graph block | Context block | `arena_used` | Inputs | Peak stack |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for model in MODELS:
        cases = by_model[model]
        for key in ("graph_bytes", "context_bytes", "arena_used", "inputs_bytes"):
            assert len({case[key] for case in cases}) == 1, (model, key)
        case = cases[0]
        stack = max(c["stack_peak"] for c in cases)
        print(
            f"| `{model}` | {case['graph_bytes']} | {case['context_bytes']} "
            f"| {case['arena_used']} | {case['inputs_bytes']} | {stack} |"
        )
    every = results["cases"]
    print(
        f"\nevery case: peak stack {max(c['stack_peak'] for c in every)}, "
        f"largest arena_used {max(c['arena_used'] for c in every)}, "
        f"arena_available {sorted({c['arena_available'] for c in every})[-1]} at most"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
