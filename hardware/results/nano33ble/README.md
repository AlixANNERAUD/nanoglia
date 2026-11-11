# Results of the first run on an Arduino Nano 33 BLE

The files of the first session of `hardware/` on a real board, an Arduino Nano 33 BLE
(nRF52840, Cortex-M4F) on the USB port of a Linux x86_64 computer, without a debug probe, on
2026-11-10 (UTC). The procedure is
[Run the golden cases on a board](../../../documentation/how-to-guides/run-the-golden-cases-on-a-board.md).

| File | What it is |
| --- | --- |
| `nano33ble.json` | Every golden case of `testdata/`, each run once: 373 ok, 3 pending (`ad01_int8`, issue #1), 0 failed, 0 skipped. |
| `nano33ble-repeat-5.json` | The same cases, each run 5 times, the fastest inference kept: the figures of `documentation/reference/performance.md` and `memory.md`. |
| `first-case.json` | The first inference of the session: `kws_ref_model/sample_0`. |
| `run-full.log`, `run-repeat-5.log` | The output of the host tool for the two runs. |
| `host.txt` | The kernel of the computer, the USB identifiers of the board, the version of `bossac` and the SHA-256 of the two firmware images. |
| `summarize.py` | Prints the tables the documentation quotes from a results file. |

The format of the results files is described in the guide; only the cases whose `timing` is
true (their model is linked into the firmware and read in place from flash) give times.

## How they were produced

The two images are those CI built at commit `2fd26df` (job "Board runner", artifact
`hardware-nano33ble-images`); their SHA-256 are in `host.txt`. From `hardware/`:

```sh
mise run hardware -- --image target/images/nano33ble-image-1.elf --image target/images/nano33ble-image-2.elf --results target/results/nano33ble.json
mise run hardware -- --image target/images/nano33ble-image-1.elf --image target/images/nano33ble-image-2.elf --repeat 5 --results target/results/nano33ble-repeat-5.json
```

The path of the home directory in the logs was replaced by `~`.

## The figures of the documentation

```sh
python3 -I hardware/results/nano33ble/summarize.py \
    hardware/results/nano33ble/nano33ble-repeat-5.json \
    hardware/results/nano33ble/nano33ble.json
```
