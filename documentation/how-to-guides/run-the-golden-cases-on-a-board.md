# Run the golden cases on a board

This guide runs every golden case of `testdata/` on an Arduino Nano 33
BLE (nRF52840, Arm Cortex-M4F) plugged in over USB, compares the outputs
with those of LiteRT, and records the time of each operator of the
MLPerf Tiny models. It needs no debug probe and, after the first flash,
no hand on the board: one command builds the firmware, flashes it
through the Arduino bootloader, runs the cases and writes a results
file.

> **Status: never run on a board.** Everything here was written and
> tested without the board: the code that runs a case and speaks the
> protocol runs in CI on the host and on an emulated Cortex-M4F, the host
> tool runs against that emulated firmware and against a simulated
> device, the firmware of the board builds and fits. What only the board
> can execute (its USB port, its clock, its counters, its resets) has not
> run anywhere: [what has never run](#what-has-never-run) lists it, and
> the [first session](#first-session-on-a-new-board) is written to find
> out, one step at a time. This is the tool for
> [issue #2](https://github.com/AlixANNERAUD/nanoglia/issues/2), not its
> result: no number in this repository comes from a board yet.

## What it measures

Two things, kept apart in the output and in the results file:

- **Correctness, for every case**: the outputs of the board against
  those LiteRT's reference kernels computed, with the tolerances, the
  pending list (`testdata/pending.json`) and the kernel checks of
  `crates/nanoglia/tests/golden.rs`
  ([bit-exactness](../explanation/bit-exactness.md)).
- **Time, only for a model that lives in flash**. An application keeps
  its model in flash and nanoglia reads the weights in place, without
  copying them. The five MLPerf Tiny models are therefore linked into
  the firmware (`include_bytes!`, aligned on 16 bytes) and run where the
  linker put them: only their cases are timed. The single-operator cases
  of `testdata/ops/` are several hundred small models, which the host
  sends into the RAM of the board: they check the results, and their
  cycle counts are recorded but marked `model in RAM, not a performance
  figure`, since a kernel that reads its weights from RAM is not timed
  as it would be in an application.

The five models and the code do not fit together in the 960 KiB of
flash after the bootloader, so the firmware is built as two images, and
the host tool flashes each in turn:

| Image | Models in flash | Cases it runs |
| --- | --- | --- |
| 1 | `kws_ref_model`, `pretrainedResnet_quant`, `ad01_int8` | every case of `testdata/ops/` (model in RAM), and the golden samples of its three models |
| 2 | `vww_96_int8`, `str_ww_ref_model` | the golden samples of its two models |

Their sizes, and the RAM each model needs on the board, are in
[Memory](../reference/memory.md#the-nano-33-ble-test-bench): with the
models in flash, all five run in the 256 KiB of RAM.

## What you need

On the computer the board is plugged into (Linux, or macOS):

- a USB cable that carries data;
- this repository, with its submodules, and [mise](https://mise.jdx.dev),
  which installs the rest (Rust, uv, and Arduino's `bossac`):

  ```sh
  git clone --recurse-submodules https://github.com/AlixANNERAUD/nanoglia
  cd nanoglia
  mise trust && mise install
  ```

- to build the firmware there: the C cross compiler and the newlib
  headers. On Debian or Ubuntu, `mise bootstrap packages apply` (or `sudo
  apt-get install gcc-arm-none-eabi libnewlib-arm-none-eabi`). On macOS,
  or to skip the build, take the two images CI built instead
  ([below](#use-the-images-ci-built));
- on Linux, the right to open serial ports: `sudo usermod -aG dialout
  $USER`, then log in again. ModemManager, if it runs, opens every new
  serial port for a few seconds and sends it modem commands: the
  protocol ignores them, but the port can be busy when the tool wants
  it. `sudo systemctl stop ModemManager` for the session if the first
  connection after a flash fails;
- on a Mac with Apple silicon, Rosetta (`softwareupdate
  --install-rosetta`): Arduino publishes `bossac` for Intel only.

### bossac

The board is flashed through its Arduino bootloader, which speaks the
SAM-BA protocol to `bossac`. It must be **Arduino's fork, version
`1.9.1-arduino2`**, the one the Arduino core for this board uses
(`tools.bossac.path` in `platform.txt` of ArduinoCore-mbed): the
mainline BOSSA does not know the nRF52840 and stops with `Device
unsupported`. The mise tasks below install it by themselves, from
Arduino's download server:

| Computer | Archive (under `https://downloads.arduino.cc/tools/`) |
| --- | --- |
| Linux x86_64 | `bossac-1.9.1-arduino2-linux64.tar.gz` |
| Linux aarch64 | `bossac-1.9.1-arduino2-linuxaarch64.tar.gz` |
| macOS | `bossac-1.9.1-arduino2-osx.tar.gz` |

Without mise, unpack the archive and put `bin/bossac` on your path, or
name it with `--bossac` or `$BOSSAC`; an Arduino IDE or `arduino-cli`
with the "Arduino Mbed OS Nano Boards" core has it too, in
`~/.arduino15/packages/arduino/tools/bossac/1.9.1-arduino2/` (Linux) or
`~/Library/Arduino15/...` (macOS), where the tool looks. It prints
`Version 1.9.1-17-g89f3556` with `--help`.

The tool runs it as the Arduino core does (`tools.bossac.upload.pattern`):

```sh
bossac -d --port=ttyACM0 -U -i -e -w nano33ble-image-1.elf.bin -R
```

`-e` erases the application, `-w` writes the image, `-R` resets into it;
the port is named without `/dev/`. The image is the firmware as a raw
binary, which the tool makes from the ELF after checking that it is
linked at `0x10000`, where the bootloader starts the application, and
that it fits in the 960 KiB that follow.

## Run everything

Plug the board in, then:

```sh
mise run test-hardware-nano33ble
```

This builds the two images (`mise run build-hardware-nano33ble`), and
for each: brings the board into its bootloader, flashes the image, waits
for the firmware to answer, and runs its cases. It prints one line per
case and ends with a summary; the results are in
`hardware/target/results/nano33ble.json`.

The board enters its bootloader without help when it runs this firmware
or a sketch of the Arduino core, the factory one included (see
[how the bootloader is entered](#how-the-bootloader-is-entered)). **When
it runs anything else, double-tap its reset button once**: the orange
LED then pulses, and the bootloader waits.

Arguments after `--` go to the host tool:

| Argument | Effect |
| --- | --- |
| `--filter kws_ref_model` | Run only the cases whose name contains this. |
| `--repeat 5` | Run each case 5 times and keep the fastest inference; every total is recorded. |
| `--port /dev/ttyACM1` | Choose the board when several are plugged in. |
| `--results file.json` | Write the results elsewhere. |

The exit code says how it went:

| Code | Meaning |
| --- | --- |
| 0 | Every case passed. Cases skipped because they do not fit in the RAM of the board are listed, and counted in the summary. |
| 1 | A case failed (a wrong output, an error of nanoglia, a crash of the firmware), a pending case passed, or a check of the run itself failed. |
| 2 | The computer lacks something: no board, no `bossac`, an image that is not one. |
| 3 | The board stopped answering and could not be brought back: it needs a double tap on its reset button. The cases run so far are in the results. |

### Use the images CI built

The job "Board runner" of `ci.yml` uploads the two images as the
artifact `hardware-nano33ble-images`. With them, the computer of the
board needs neither Rust nor a C compiler:

```sh
gh run download <run-id> --name hardware-nano33ble-images --dir hardware/target/images
mise run hardware -- \
  --image target/images/nano33ble-image-1.elf \
  --image target/images/nano33ble-image-2.elf \
  --results target/results/nano33ble.json
```

`mise run hardware -- <arguments>` runs the host tool
(`hardware/tools/hardware.py`) from `hardware/`, with `bossac` on the
path. Check out the commit the images were built from: the tool reads
the cases from `testdata/`.

## The results file

`hardware/target/results/nano33ble.json` holds the date, the revision
of the host tool, what each firmware image says of itself (firmware,
revision of its sources, version of nanoglia, kernel families and
optimization level, board, clock, size of its RAM store, models in
flash), a summary, and one entry per case:

| Field | Meaning |
| --- | --- |
| `status` | `ok`; `pending` (listed in `testdata/pending.json`, and differs on the comparison only); `skipped` (does not fit, with the sizes in `message`); `FAIL`; `FIXED` (a pending case that passes: remove it from the list). |
| `model_location` | `flash`: linked into the firmware, read in place. `ram`: received over USB. |
| `timing` | `true` only when the model is in flash and the counter of the device is a clock. Use the times of these cases, and no others. |
| `timing_note` | Why not, otherwise: `model in RAM, not a performance figure`. |
| `operators` | For each operator, its kernel, its `cycles` and, when `timing` is true, its `ms`. |
| `load_cycles`, `context_cycles` | Cycles of `Graph::load_in` and of the creation of the execution context. |
| `compute_cycles`, `compute_ms`, `compute_cycles_runs` | Cycles of the inference: the fastest of the repeats, in milliseconds when `timing` is true, then each repeat. |
| `graph_bytes`, `context_bytes`, `arena_used`, `arena_available`, `inputs_bytes`, `stack_peak` | The memory of the case on the board, in bytes: the two blocks of nanoglia, what the arena handed out and what it had, the inputs, and the deepest the stack went. |

The cycles are those of the cycle counter of the core (`DWT.CYCCNT`),
which counts the core clock: 64 MHz, from the 32 MHz crystal of the
board, which the firmware starts because USB needs it. They are read
around each operator by an [`Observer`](run-long-inferences.md) of
`compute_with`; between two operators the firmware serves its USB port,
outside the count. The firmware is built for speed (`opt-level = 3`, fat
LTO, CMSIS-NN with its DSP code paths), which its `build` line reports,
and leaves the instruction cache of the flash off, as the chip has it
after a reset.

Before quoting a time from this file, read the
[status](#what-has-never-run) above, and add the measurements to
[Performance](../reference/performance.md) with the board, the date and
the commit.

## First session on a new board

Each step has one thing to see. Stop at the first that fails.

**1. The port.** Plug the board in.

```sh
mise run hardware -- --list-ports
```

Expect one line for the board:

```text
/dev/ttyACM0  2341:805a  arduino sketch  Nano 33 BLE ...
```

`arduino sketch` is a board that runs a sketch of the Arduino core, as
from the factory; `bootloader` (`2341:005a`) one in its bootloader;
`runner` (`1209:0001`) one that already runs this firmware. No line:
the cable carries no data, or the board runs something without a USB
port; double-tap the reset button (the orange LED pulses) and list
again, to see `bootloader`. On macOS the port is `/dev/cu.usbmodem...`.

**2. The firmware.** Build the images, or download them, then flash the
first:

```sh
mise run build-hardware-nano33ble
mise run hardware -- --image target/images/nano33ble-image-1.elf --flash-only
```

Expect the `bossac` command, then:

```text
flashed: nanoglia-hardware-nano33ble 0.0.0
```

- `the board did not enter its bootloader`: double-tap the reset
  button, run again.
- `bossac failed`, with `Device unsupported` in its output: this is not
  Arduino's `bossac`. With `No device found`: the port went away;
  list the ports and run again.
- `was written but the firmware does not answer`: the firmware does not
  start on the board, or its USB port does not work: the first thing
  this session can find. List the ports: no `runner` line means that the
  port never came up. Wait: a firmware that hears no host for 5 minutes
  resets into the bootloader by itself (a `bootloader` line appears);
  otherwise double-tap. Then the board is back where it was, and the
  problem is in `hardware/nano33ble/src/main.rs`.

**3. What the firmware says.**

```sh
mise run hardware -- --info
```

Expect, among other lines:

```text
ram_bytes: 237232
counter_hz: 64000000
firmware: nanoglia-hardware-nano33ble 0.0.0
board: Arduino Nano 33 BLE (nRF52840, Cortex-M4F)
clock: DWT cycle counter, core clock of 64 MHz from the 32 MHz crystal (HFXO)
crash: None
models: [('kws_ref_model.tflite', 53936), ('pretrainedResnet_quant.tflite', 98496), ('ad01_int8.tflite', 276976)]
```

`ram_bytes` may differ by a few bytes. `clock: DWT cycle counter not
running: no timing` (and `counter_hz: 0`) means that the counter does
not count without a debugger on this chip: the cases still run and are
compared, without times. `reset_reason` is the `RESETREAS` register of
the nRF52840: 4 after a reset asked by software, as `bossac -R` does.

**4. One case.**

```sh
mise run hardware -- --filter kws_ref_model/sample_0
```

Expect:

```text
ok       mlperf-tiny/golden/kws_ref_model/sample_0 [flash, <time> ms]
self-check: a corrupted output 0 of mlperf-tiny/golden/kws_ref_model/sample_0 fails, as it must
1 cases: 1 ok, 0 pending, 0 skipped, 0 failed; 1 timed with their model in flash
```

This is the first inference on the chip: CMSIS-NN's DSP kernels, read
from flash, against LiteRT's bytes. `FAIL ... max diff`: a kernel gives
other bytes on the chip than under emulation, which is what issue #2
looks for; keep the results file. `FAIL ... the device stopped
answering`: the line says how the firmware restarted (a panic with its
message, a fault with its address, the watchdog).

**5. The ways out.** About one and a half minute, most of it waiting
for the watchdog:

```sh
mise run hardware -- --self-test
```

The firmware is made to panic, to take a fault, then to hang; each time
the board must come back and say what happened:

```text
self-test: panic
  ok: the firmware restarted after a panic: panicked at nano33ble/src/main.rs:...
self-test: hard fault
  ok: the firmware restarted after a fault: hard fault at pc=0x...
self-test: hang (the watchdog of the firmware ends it, after about 20 s on a board)
  ok: the firmware restarted after a watchdog: watchdog: nothing moved for 20 s
self-test: reset into the firmware
  ok: the firmware answers again, without a crash to report
```

A restart reported without its message (`restarted after a panic`
alone) means that the bootloader used the RAM where the message was
left: the kind of the crash is kept in a register, the text is lost. A
`FAIL: ... the firmware reports None` means that the register
(`GPREGRET2`) does not survive the reset through the bootloader: crashes
are then seen (the case fails, the board comes back) but not named, and
the count that sends a firmware crashing at start into the bootloader
does not work. Exit code 3: the board did not come back; double-tap,
flash again, and do not rely on the recovery.

**6. The bootloader, from software.**

```sh
mise run hardware -- --bootloader
mise run hardware -- --list-ports
```

Expect `the board is in its bootloader on ...`, then a `bootloader`
line. This is what makes every later flash hands-free.

**7. Everything.**

```sh
mise run test-hardware-nano33ble
```

Expect the two images flashed in turn, about 380 lines, and at the end
the counts the emulated run gives: every case `ok` except the three
`pending` ones of `ad01_int8` (LiteRT and TensorFlow Lite Micro disagree
there, see [bit-exactness](../explanation/bit-exactness.md)), none
failed, none skipped, 15 timed. Keep `hardware/target/results/nano33ble.json`.

## How the bootloader is entered

The Arduino bootloader of the Nano 33 BLE occupies the first 64 KiB of
the flash and starts the application at `0x10000`. Its source is not
published
([ArduinoCore-nRF528x-mbedos issue 23](https://github.com/arduino/ArduinoCore-nRF528x-mbedos/issues/23));
what follows is read from the code that talks to it.

- **From software**: a sketch of the Arduino core writes `0xB0`
  (`DFU_MAGIC_SERIAL_ONLY_RESET`) into the `GPREGRET` register of the
  chip, which survives a system reset, then resets
  (`_ontouch1200bps_()` in
  [`variants/ARDUINO_NANO33BLE/variant.cpp`](https://github.com/arduino/ArduinoCore-mbed/blob/b4ad4e7218a691989acf6c4479112eae183a5a50/variants/ARDUINO_NANO33BLE/variant.cpp#L212-L216),
  the value in
  [`pins_arduino.h`](https://github.com/arduino/ArduinoCore-mbed/blob/b4ad4e7218a691989acf6c4479112eae183a5a50/variants/ARDUINO_NANO33BLE/pins_arduino.h#L166));
  the bootloader then stays, waiting for `bossac`. The core does it when
  the host sets the serial port to 1200 baud and closes it, the "1200
  baud touch"
  ([`cores/arduino/USB/USBSerial.cpp`](https://github.com/arduino/ArduinoCore-mbed/blob/b4ad4e7218a691989acf6c4479112eae183a5a50/cores/arduino/USB/USBSerial.cpp#L31-L48),
  `nano33ble.upload.use_1200bps_touch=true` in `boards.txt`).
  MicroPython enters it the same way
  (`ports/nrf/boards/ARDUINO_NANO_33_BLE_SENSE/board.c`).
- **This firmware** does the same on three occasions: the `RESET`
  command of the host tool, the 1200 baud touch (so that the Arduino IDE
  or a plain `stty -F /dev/ttyACM0 1200` works too), and by itself when
  it crashed three times in a row without hearing from a host, or heard
  from none for 5 minutes after it started.
- **By hand**: a double tap on the reset button.

`bossac` cannot write the bootloader: Arduino's fork addresses the flash
of this chip from 0
([`src/Device.cpp`](https://github.com/arduino/BOSSA/blob/1.9.1-arduino2/src/Device.cpp#L58-L60)),
and the image the Arduino toolchain gives it there is linked at
`0x10000`: the bootloader places what it receives after itself. Tock's
guide for this board, which replaces the bootloader, says the same and
has to go through a second-stage loader to do so ("the bossa bootloader
does not have a mechanism for updating itself",
[`boards/nano33ble/README.md`](https://github.com/tock/tock/blob/master/boards/nano33ble/README.md)).
The firmware itself has no code that writes or erases flash: nothing a
host sends can change the firmware or the bootloader.

## When the board stops answering

The host tool waits 30 seconds for a run. Past that, or when the port
goes away, the case fails, and the tool waits for the firmware to come
back, asks it why it restarted, adds the answer to the failure, and goes
on with the next case; if the board comes back in its bootloader, the
image is flashed again. Three cases lost in a row, or a board that does
not come back in a minute, stop the run with exit code 3.

What brings the firmware back by itself:

| What happens | What the firmware does |
| --- | --- |
| A panic (an overflow, a bug in a kernel) | The handler leaves the message in RAM and the kind of crash in the `GPREGRET2` register, and resets. |
| A fault of the processor (a bad address, an undefined instruction) | The same, with the address of the instruction. |
| A kernel that never returns | A timer interrupt fires when nothing fed it for 20 seconds (the firmware feeds it between operators and while it waits for the host), and does the same. |
| Three crashes in a row without a request from the host (a firmware that crashes while it starts) | Resets into the bootloader. |
| No byte from a host for 5 minutes after the start (a USB port that does not work) | Resets into the bootloader. |
| A frame cut short | Dropped after a second without a byte; the host sends it again. |

The watchdog is a timer of the firmware, not the watchdog of the chip
(WDT): once started, that one cannot be stopped and keeps counting
through a reset into the bootloader, which does not feed it; it would
reset the board in the middle of a flash.

**What still needs a double tap on the reset button:**

- the first flash, when the board runs something that does not answer
  the 1200 baud touch;
- a firmware that hangs with interrupts masked, or that corrupts the
  timer or the interrupt controller: the timer interrupt cannot fire;
- a firmware that hangs before `main` starts the timer (the start-up
  code of the runtime, before any code of this firmware);
- a firmware whose USB port works but which the host tool cannot talk
  to, and which does not react to the 1200 baud touch;
- a flash interrupted half-way (the cable pulled during `bossac`): the
  application is erased or partial;
- anything that happens while the board is in its bootloader, which this
  repository does not control.

## What runs without the board

`mise run test-hardware` (the job "Board runner" of CI) runs everything
that does not need the board:

- the runner (`hardware/runner`: the protocol, the loading and running
  of a case, the timing, the count of crashes) on the host, behind a
  link held in memory: whole cases with the model in RAM and linked in,
  corrupted and truncated frames, pauses, bad arguments;
- the same runner built for `thumbv7em-none-eabihf` and run under the
  Unicorn engine (`hardware/emulated`), behind memory-mapped registers
  in place of the USB port, with the RAM of the board (256 KiB, 24 KiB
  of stack) and the five models in its flash: the host tool drives it
  through every golden case, then through a panic, a fault, a hang and a
  firmware that crashes while starting;
- the host tool against a simulated device that corrupts and loses
  answers, hangs, panics, comes back in its bootloader or never: the
  retries, the recovery, the exit codes, the results file; and its
  `bossac` command against a stand-in for `bossac`.

`mise run hardware -- --dry-run` runs the host tool against the
simulated device alone: it answers with the expected outputs, so this
checks the tool, never nanoglia.

## What has never run

Only the board can execute `hardware/nano33ble/src/main.rs`, and it has
not:

- the start of the clocks and of the USB device (`embassy_nrf::init`
  with the crystal, the embassy USB driver, its VBUS detection), and the
  loop that polls the USB futures by hand;
- `read` and `write` on the CDC-ACM port: packets of 64 bytes, the empty
  packet that ends a transfer, the timeouts;
- the cycle counter (`DWT.CYCCNT`) without a debugger, and the 1 MHz
  timer (TIMER1) that gives the time and the watchdog;
- the watchdog interrupt, the panic and HardFault handlers on the chip,
  `GPREGRET2` and the message in RAM across a reset that goes through
  the bootloader;
- the reset into the bootloader (`GPREGRET = 0xB0`), the 1200 baud
  touch, the return to the bootloader after 5 minutes of silence;
- `bossac` on the real bootloader, and the detection of the ports by
  their USB identifiers.

And, as everywhere in this repository until issue #2 is closed: CMSIS-NN's
DSP kernels have run under an instruction-set emulator only, which
models neither timing nor alignment faults on a Cortex-M4.
