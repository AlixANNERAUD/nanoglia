# Run the golden cases on a board

This guide runs every golden case of `testdata/` on an Arduino Nano 33
BLE (nRF52840, Arm Cortex-M4F) plugged in over USB, compares the outputs
with those of LiteRT, and records the time of each operator of the
MLPerf Tiny models. It needs no debug probe and, after the first flash,
no hand on the board: one command builds the firmware, flashes it
through the Arduino bootloader, runs the cases and writes a results
file.

> **Status: run once, on one board.** On 2026-11-10 the
> [first session](#first-session-on-a-new-board) ran from start to end
> on one Arduino Nano 33 BLE, from a Linux x86_64 computer, with the two
> images CI built at commit `2fd26df`: every golden case passed except
> the three known `pending` ones, and the MLPerf Tiny models were timed.
> The results files are in `hardware/results/nano33ble/`
> ([README](../../hardware/results/nano33ble/README.md)), the times in
> [Performance](../reference/performance.md#on-an-arduino-nano-33-ble).
> Several paths of the firmware and of the tool still have not run on a
> board: [what has run, and what has not](#what-has-run-on-a-board)
> lists them, with what this session corrected (the
> [instruction cache](#the-instruction-cache) was on). This is one
> board and one session of
> [issue #2](https://github.com/AlixANNERAUD/nanoglia/issues/2).

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
  $USER`, then log in again. A process started before that (a
  long-running service, a terminal or editor server that spawns your
  shells) keeps the groups it started with, and does not get `dialout`
  until it restarts. A udev rule gives the ports of the board to every
  user instead, whatever their groups; in
  `/etc/udev/rules.d/99-nano33ble.rules`:

  ```text
  # Arduino (2341): the sketches of the Arduino core and the bootloader.
  SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", MODE="0666"
  # The runner firmware of nanoglia (1209:0001).
  SUBSYSTEM=="tty", ATTRS{idVendor}=="1209", ATTRS{idProduct}=="0001", MODE="0666"
  ```

  then `sudo udevadm control --reload-rules && sudo udevadm trigger`,
  and unplug the board and plug it in again. `MODE="0666"` lets any
  local user open these ports: use it on a computer you do not share,
  or name a group with `GROUP=` and a stricter mode;
- on Linux, no ModemManager on the port: if it runs, it opens every new
  serial port for a few seconds and sends it modem commands. The
  protocol ignores them, but the port can be busy when the tool wants
  it. The first session stopped it (`sudo systemctl stop ModemManager`)
  before starting: how the tool fares with it running has not been
  seen on a board;
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
`Version 1.9.1-17-g89f3556` with `--help`; the tool checks this before
it touches the board and refuses any other `bossac`.

The tool runs it as the Arduino core does
(`tools.bossac.upload.pattern`), plus a verification:

```sh
bossac -d --port=ttyACM0 -U -i -e -w -v nano33ble-image-1.elf.bin -R
```

`-w` writes the image, `-v` asks the bootloader for the checksum of what
it wrote and compares, `-R` resets into the application; the port is
named without `/dev/`. `-e` is part of Arduino's pattern but erases
nothing here: this bootloader only acknowledges the erase command, and
erases each page as it writes it. The image is the firmware as a raw
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
LED then pulses, and the bootloader waits. On a board that has never
run this firmware, go through the
[first session](#first-session-on-a-new-board) instead of this command.

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
| 4 | The firmware does not work on the board, which went back to its bootloader by itself and waits there. Nothing to do by hand: the firmware is to fix. |

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
the cases from `testdata/`. The first session did this: the computer
the board was plugged into had no Arm toolchain, and ran the images of
the run of `ci.yml` at commit `2fd26df`.

## The results file

`hardware/target/results/nano33ble.json` holds the date, the revision
of the host tool, what each firmware image says of itself (firmware,
revision of its sources, version of nanoglia, kernel families and
target in `build`, `opt_level`, `rustc`, board, clock, the state of the
instruction cache in `settings`, size of its RAM store, models in
flash), a summary, and one entry per case:

| Field | Meaning |
| --- | --- |
| `status` | `ok`; `pending` (listed in `testdata/pending.json`, and differs on the comparison only); `skipped` (does not fit, with the sizes in `message`); `FAIL`; `FIXED` (a pending case that passes: remove it from the list). |
| `model_location` | `flash`: linked into the firmware, read in place. `ram`: received over USB. |
| `timing` | `true` only when the model is in flash and the counter of the device is a clock. Use the times of these cases, and no others. |
| `timing_note` | Why not, otherwise: `model in RAM, not a performance figure`. |
| `operators` | For each operator, its kernel, its `cycles` and, when `timing` is true, its `ms`. |
| `load_cycles`, `context_cycles` | Cycles of `Graph::load_in` and of the creation of the execution context. |
| `compute_cycles`, `compute_ms`, `compute_cycles_runs` | Cycles of the inference, the sum of those of its operators, on 64 bits (what the firmware does between two operators, serving its USB port, is not counted): the fastest of the repeats, in milliseconds when `timing` is true, then each repeat. |
| `graph_bytes`, `context_bytes`, `arena_used`, `arena_available`, `inputs_bytes`, `stack_peak` | The memory of the case on the board, in bytes: the two blocks of nanoglia, what the arena handed out and what it had, the inputs, and the deepest the stack went. |

The cycles are those of the cycle counter of the core (`DWT.CYCCNT`),
which counts the core clock: 64 MHz, from the 32 MHz crystal of the
board, which the firmware starts because USB needs it. They are read
around each operator by an [`Observer`](run-long-inferences.md) of
`compute_with`; between two operators the firmware serves its USB port,
outside the count. The count of an operator is on 32 bits and would
wrap after 67 seconds; the watchdog ends an operator after 20.

Three settings decide the times, and the results file records each:
the firmware is built for speed (`opt_level` 3, fat LTO, CMSIS-NN with
its DSP code paths), by the compiler named in `rustc`, and runs with
the instruction cache of the flash (`ICACHECNF` of the NVMC) in the
state `settings` reports, read when the firmware starts.

### The instruction cache

On the first session, the images were built without the feature
`icache`, and the firmware reported `instruction cache of the flash on
(NVMC ICACHECNF)`. The firmware, its HAL and this guide assumed the
cache off, as the chip has it after a reset. What is known, and what is
inferred:

- **Known, from the board**: bit 0 (`CACHEEN`) of `ICACHECNF` was 1
  when the firmware read it, after `embassy_nrf::init`, in every run of
  the session (`settings` of each image in each results file).
- **Known, from the code**: without the feature `icache`, the firmware
  never writes `ICACHECNF` (`hardware/nano33ble/src/main.rs`), and
  embassy-nrf 0.11.0, its HAL, has no code that does (no
  `icachecnf` in its sources).
- **Known, from the bootloader Arduino publishes**: in
  [`bootloader.elf`](https://github.com/arduino/ArduinoCore-mbed/blob/b4ad4e7218a691989acf6c4479112eae183a5a50/bootloaders/nano33ble/bootloader.elf)
  (sha256 `d3bdf577960943eb29887a64f0191de9930ddd1b54b352c394a330613cbc8cd7`),
  the start-up function `nordicsemi_nrf52_init` of Zephyr writes 1 to
  `0x4001e540`, `ICACHECNF`, after `SystemInit`
  (`arm-none-eabi-objdump -d`, at `0xa90`), and the image is built with
  `CONFIG_NRF_ENABLE_ICACHE=1` (a symbol of the ELF). It also writes 1
  to `0x40000578`, `DCDCEN` of POWER: the DC/DC regulator is on too.
  Starting the application does not undo either.
- **Inferred**: the bootloader of this board is that one (it behaves as
  its disassembly says everywhere else in this guide, but its flash has
  not been read back), so every application started by it, this
  firmware included, starts with the cache on.

So every time measured so far is a time **with the cache on**, and the
feature `icache` changes nothing on a board started by this bootloader.
No time with the cache off has been measured: that needs a firmware
that writes `ICACHECNF = 0` when it is built without `icache`, which it
does not do yet.

Before quoting a time from this file, read
[what has run](#what-has-run-on-a-board), and add the measurements to
[Performance](../reference/performance.md) with the board, the date and
the commit.

## First session on a new board

Each step has one thing to see, and the order matters: the way back to
the bootloader is proven before anything that could need it. Stop at
the first step that fails. `mise run hardware -- <arguments>` runs the
host tool from `hardware/`.

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

**2. The bootloader, from the factory sketch, before flashing
anything.**

```sh
mise run hardware -- --bootloader
mise run hardware -- --list-ports
```

Expect `1200 baud touch on ... (arduino sketch)`, then `the board is in
its bootloader on ...`, then a `bootloader` line in the list. This
proves the touch and the detection of the ports with a firmware that is
known to work. `the board did not enter its bootloader`: double-tap the
reset button; the rest of the session then tells whether the touch
works with this firmware (step 5). On the first session the board ran
a firmware that ignores the touch, so this step needed the double tap,
and no other step did.

**3. The firmware.** Build the images, or
[download them](#use-the-images-ci-built), then flash the first:

```sh
mise run build-hardware-nano33ble
mise run hardware -- --image target/images/nano33ble-image-1.elf --flash-only
```

Expect the `bossac` command, then, within a few seconds:

```text
flashed: nanoglia-hardware-nano33ble 0.0.0
```

On the first session the flash of image 1 took 36 s, its verification
included, and the `runner` port appeared within 4 s of the end.

- `is version ..., not Arduino's bossac`: another `bossac` is on the
  path; use the one mise installs.
- `bossac failed`, with `No device found`: the port went away; list the
  ports and run again.
- `the firmware does not answer yet: waiting up to 330 s for it, or for
  the bootloader`: no `runner` port appeared, or nothing answers on it.
  **Wait, and leave the board alone.** A firmware that hears no request
  resets into the bootloader by itself after 5 minutes, and one that
  crashes while it starts does so at its third crash: the tool waits up
  to 6 minutes for either and ends with exit code 4 and `does not work
  on this board`, the board back in its bootloader, as before the flash.
  The problem is then in `hardware/nano33ble/src/main.rs`. Only if the
  tool ends with exit code 3 (`did not return to its bootloader`) is a
  double tap needed.

**4. What the firmware says.**

```sh
mise run hardware -- --info
```

Expect, among other lines:

```text
ram_bytes: 237152
counter_hz: 64000000
crash: None
failed_boots: 0
firmware: nanoglia-hardware-nano33ble 0.0.0
opt_level: 3
board: Arduino Nano 33 BLE (nRF52840, Cortex-M4F)
clock: DWT cycle counter, core clock of 64 MHz from the 32 MHz crystal (HFXO)
settings: instruction cache of the flash on (NVMC ICACHECNF)
models: [('kws_ref_model.tflite', 53936), ('pretrainedResnet_quant.tflite', 98496), ('ad01_int8.tflite', 276976)]
```

On the first session the board answered `ram_bytes: 237152`,
`counter_hz: 64000000`, `crash: None` and `failed_boots: 0`, and the
firmware described itself as above (the `firmware_images` of the
results files), `settings` reading `on` although the images were built
without `icache` ([the instruction cache](#the-instruction-cache)).
`counter_hz: 0` with `clock: DWT cycle counter not running: no timing`
means that the counter does not count without a debugger on this chip:
the cases still run and are compared, without times. `crash:` anything
but `None` means that the firmware crashed and restarted before it
answered; `crash_message` says where. `reset_reason` is the `RESETREAS`
register of the nRF52840: 4 after a reset asked by software, as
`bossac -R` does. On the first session it was 1 (the reset pin) after
the first flash, which followed a double tap, and 4 after the later
ones.

**5. The way back, from this firmware.**

```sh
mise run hardware -- --bootloader
mise run hardware -- --list-ports
mise run hardware -- --image target/images/nano33ble-image-1.elf --flash-only
```

Expect `the board is in its bootloader on ...`, a `bootloader` line,
then `flashed:` again. From here on every flash is hands-free, and so is
the recovery of the next steps. On the first session `--bootloader`
took 1 s (the tool asks the firmware with its `RESET` command first, and
falls back on the 1200 baud touch only when that fails: the touch
itself was not needed), and the flash that followed succeeded. If `--bootloader` fails while step 2
worked, the firmware's own reset into the bootloader (`GPREGRET`) does
not work: double-tap, and do not go further before it is fixed.
(`mise run hardware -- --leave-bootloader` starts the application of a
board that sits in its bootloader, without flashing.)

**6. The ways out of a crash.** 25 s on the first session, 20 of them
waiting for the watchdog:

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

On the first session all four passed: the panic came back with its
message (file and line), the hard fault with its `pc` and `lr`, the hang
was ended by the watchdog after 20 s. The crash record therefore
survives a reset through the bootloader. (The lines above are the
format of the tool, not a copy of that session's output.)

A restart reported without its message (`restarted after a panic`
alone) means that the message, left in the last 256 bytes of the RAM,
did not survive the reset; the kind of the crash is kept in a register.
A `FAIL: ... the firmware reports None` means that the register
(`GPREGRET2`) does not survive the reset through the bootloader: crashes
are then seen (the case fails, the board comes back) but not named, and
a firmware that crashes while it starts is no longer sent to the
bootloader at its third crash (it still goes there after 5 minutes of
silence, if it gets as far as its USB loop). Exit code 3: the board did
not come back in 90 seconds; double-tap, flash again, and do not rely
on the recovery.

**7. One case.**

```sh
mise run hardware -- --filter kws_ref_model/sample_0
```

Expect, as on the first session (`hardware/results/nano33ble/first-case.json`,
10700515 cycles):

```text
ok       mlperf-tiny/golden/kws_ref_model/sample_0 [flash, 167.196 ms]
self-check: a corrupted output 0 of mlperf-tiny/golden/kws_ref_model/sample_0 fails, as it must
1 cases: 1 ok, 0 pending, 0 skipped, 0 failed; 1 timed with their model in flash
```

The time is that of the first session's firmware, with the instruction
cache on; another build gives another one.

This is the first inference on the chip: CMSIS-NN's DSP kernels, read
from flash, against LiteRT's bytes. `FAIL ... max diff`: a kernel gives
other bytes on the chip than under emulation, which is what issue #2
looks for; keep the results file. `FAIL ... the device stopped
answering`: the line says how the firmware restarted (a panic with its
message, a fault with its address, the watchdog).

**8. Everything, then again for the times.**

```sh
mise run test-hardware-nano33ble
mise run test-hardware-nano33ble -- --repeat 5 --results target/results/nano33ble-repeat-5.json
```

Expect the two images flashed in turn, about 380 lines, and at the end,
as on the first session:

```text
376 cases: 373 ok, 3 pending, 0 skipped, 0 failed; 15 timed with their model in flash
peak stack 10500 B, largest arena 94816 B
```

Every case `ok` except the three `pending` ones of `ad01_int8` (LiteRT
and TensorFlow Lite Micro disagree there, see
[bit-exactness](../explanation/bit-exactness.md)), none failed, none
skipped, 15 timed: the counts of the emulated run. On the first session
the first run took 1 min 22 s, both flashes included, and the second,
which keeps the fastest of five inferences of each case, 1 min 48 s.
Keep the two files of `hardware/target/results/`.

## How the bootloader is entered

The Arduino bootloader of the Nano 33 BLE occupies the first 64 KiB of
the flash and starts the application at `0x10000`. Its source is not
published
([ArduinoCore-nRF528x-mbedos issue 23](https://github.com/arduino/ArduinoCore-nRF528x-mbedos/issues/23)),
but its ELF file is, with its symbols:
[`bootloaders/nano33ble/bootloader.elf`](https://github.com/arduino/ArduinoCore-mbed/blob/b4ad4e7218a691989acf6c4479112eae183a5a50/bootloaders/nano33ble/bootloader.elf)
in ArduinoCore-mbed, a Zephyr program. Disassembled
(`arm-none-eabi-objdump -d`), its `main` settles what follows:

- **It stays, waiting for `bossac`**, in three cases: the `GPREGRET`
  register of the chip holds `0xB0`; the reset button was tapped twice
  (it leaves a mark at `0x20007ffc` and waits 500 ms for a second
  reset to find it); or the first words of the application, at
  `0x10000`, are erased. It clears `GPREGRET` when it stays.
- **Otherwise it starts the application** (`mbed_start_application`):
  it disables and clears every interrupt line, points the vector table
  at `0x10000`, loads the stack pointer and jumps. It does not reset the
  peripherals: its clocks and timers keep running, which is why the
  firmware turns them off first thing.
- **It cannot be written through itself**: it adds `0x10000` to every
  address `bossac` gives it, for the writes and for the checksums.
  Arduino's `bossac` addresses the flash of this chip from 0
  ([`src/Device.cpp`](https://github.com/arduino/BOSSA/blob/1.9.1-arduino2/src/Device.cpp#L58-L60)).
  Tock's guide for this board, which replaces the bootloader, has to go
  through a second-stage loader to do so ("the bossa bootloader does not
  have a mechanism for updating itself",
  [`boards/nano33ble/README.md`](https://github.com/tock/tock/blob/master/boards/nano33ble/README.md)).
- **It uses the RAM from `0x20000000` to `0x20009e14`** (its `bss`,
  `noinit` and data sections) at every reset, the first 27976 bytes of
  it cleared. Its stacks are in there too: the initial stack pointer of
  its vector table is `0x20008d48`, the top of an 8 KiB main stack in
  `noinit`, followed by the idle, interrupt and work-queue stacks up to
  `0x20009a88`. Nothing of it is at the top of the RAM, where the
  firmware keeps the message of a crash in the last 256 bytes.

`GPREGRET` survives a system reset, and that is how software asks for
the bootloader:

- A sketch of the Arduino core writes `0xB0`
  (`DFU_MAGIC_SERIAL_ONLY_RESET`) into it, then resets
  (`_ontouch1200bps_()` in
  [`variants/ARDUINO_NANO33BLE/variant.cpp`](https://github.com/arduino/ArduinoCore-mbed/blob/b4ad4e7218a691989acf6c4479112eae183a5a50/variants/ARDUINO_NANO33BLE/variant.cpp#L212-L216),
  the value in
  [`pins_arduino.h`](https://github.com/arduino/ArduinoCore-mbed/blob/b4ad4e7218a691989acf6c4479112eae183a5a50/variants/ARDUINO_NANO33BLE/pins_arduino.h#L166)),
  when the host sets the serial port to 1200 baud and closes it, the
  "1200 baud touch"
  ([`cores/arduino/USB/USBSerial.cpp`](https://github.com/arduino/ArduinoCore-mbed/blob/b4ad4e7218a691989acf6c4479112eae183a5a50/cores/arduino/USB/USBSerial.cpp#L31-L48),
  `nano33ble.upload.use_1200bps_touch=true` in `boards.txt`).
  MicroPython enters it the same way
  (`ports/nrf/boards/ARDUINO_NANO_33_BLE_SENSE/board.c`).
- **This firmware** does the same on four occasions: the `RESET`
  command of the host tool; the 1200 baud touch (so that the Arduino IDE
  or a plain `stty -F /dev/ttyACM0 1200` works too); in the handler of
  its third crash in a row without a request from a host; and when no
  request came for 5 minutes after it started.

The way out of the bootloader without flashing is `bossac -R`
(`mise run hardware -- --leave-bootloader`), which starts the
application if there is one.

What the firmware can write: it has no code that erases or writes a
page of flash, so nothing a host sends can change the firmware or the
bootloader. The start-up code of its HAL (`embassy_nrf::init`) holds a
routine that programs words of the UICR, the configuration registers
of the chip; the firmware is configured so that no call of it writes:
the reset pin is declared a GPIO to the HAL (which then wants
`PSELRESET` erased, and leaves a programmed one alone: the reset button
keeps working as the board has it), the NFC pins are left to the NFC
(same reasoning, the other way), and neither the debug port protection
nor the regulator voltage is configured.

## When the board stops answering

The host tool waits 30 seconds for a run. Past that, or when the port
goes away, the case fails, and the tool waits up to 90 seconds for the
board to settle: either the firmware answers again, and is asked why it
restarted (the answer is added to the failure), or the board is in its
bootloader, and the image is flashed again. Then the run goes on with
the next case. Three cases lost in a row, or a board that shows neither
port in 90 seconds, stop the run with exit code 3. After a flash, the
tool waits up to 330 seconds, for the same two outcomes.

These waits follow from what the firmware does by itself:

| What happens | What the firmware does |
| --- | --- |
| A panic (an overflow, a bug in a kernel) | The handler leaves the message in the last bytes of the RAM, counts the crash in the `GPREGRET2` register, and resets. |
| A fault of the processor (a bad address, an undefined instruction), an interrupt nothing handles | The same, with the address of the instruction or the number of the interrupt. |
| A kernel that never returns, a start-up that waits for a clock that never starts | A timer interrupt, alone at the highest priority, fires when nothing fed it for 20 seconds (the firmware feeds it between operators and while it waits for the host), and does the same. The timer starts first thing in `main`. |
| The third crash in a row without a request from the host | Its handler resets into the bootloader instead: at most 3 × 20 seconds, plus the resets, for a firmware that hangs at each start. The count is made in the handler, so a crash in the first lines of `main` counts. |
| No request from a host for 5 minutes after the start (a USB port that does not work, or works and hears only noise) | Resets into the bootloader. |
| A frame cut short | Dropped after a second without a byte; the host sends it again. |

The watchdog is a timer of the firmware, not the watchdog of the chip
(WDT): once started, that one cannot be stopped and would keep counting
through a reset into the bootloader, which does not feed it; it would
reset the board in the middle of a flash.

**What still needs a double tap on the reset button:**

- the first flash, when the board runs something that does not answer
  the 1200 baud touch (not the factory sketch, which does);
- a firmware that hangs with interrupts masked, or after it corrupted
  the timer or the interrupt controller: the timer interrupt cannot
  fire;
- a firmware that hangs before `main` starts the timer: the start-up
  code of the runtime, a few instructions that copy and clear memory;
- a fault inside the crash handler itself: the core locks up and the
  chip resets without the crash being counted, so a loop of those never
  reaches the bootloader (unlikely: the handler writes one static and
  two registers);
- a firmware whose crash handler itself does not work on the chip (it
  resets without counting, forever) and which crashes before its USB
  loop: every recovery rests on the handler writing `GPREGRET2` and on
  that register surviving the reset, which step 6 checks;
- a flash interrupted half-way (the cable pulled during `bossac`) that
  leaves the first page written and the rest not: the bootloader starts
  what is there. (Interrupted before the first page, the bootloader
  stays by itself.)
- anything that happens while the board is in its bootloader, which this
  repository does not control.

A firmware whose USB port works but which the host cannot talk to is
not in this list: it hears no request, and resets into the bootloader
after 5 minutes.

## What runs without the board

`mise run test-hardware` (the job "Board runner" of CI) runs everything
that does not need the board:

- the runner (`hardware/runner`: the protocol, the loading and running
  of a case, the timing, the count of crashes) on the host, behind a
  link held in memory: whole cases with the model in RAM and linked in,
  corrupted and truncated frames, pauses, bad arguments, crashes counted
  without the runner ever starting;
- the same runner built for `thumbv7em-none-eabihf` and run under the
  Unicorn engine (`hardware/emulated`), behind memory-mapped registers
  in place of the USB port, with the RAM of the board (256 KiB, the
  crash record in its last 256 bytes, 24 KiB of stack below) and the
  five models in its flash: the host tool drives it through every golden
  case, then through a panic, a fault, a hang and a firmware that
  crashes while starting, before its runner exists;
- the host tool against a simulated device that corrupts and loses
  answers, hangs, panics, comes back in its bootloader or never: the
  retries, the recovery, the exit codes, the results file; its waits
  for a board that restarts, lands in its bootloader or disappears,
  against scripted lists of ports; and its `bossac` commands against a
  stand-in for `bossac`.

`mise run hardware -- --dry-run` runs the host tool against the
simulated device alone: it answers with the expected outputs, so this
checks the tool, never nanoglia.

## What has run on a board

One session, on 2026-11-10, on one Arduino Nano 33 BLE plugged into a
Linux x86_64 computer, without a debug probe, with the two images CI
built at commit `2fd26df` (artifact `hardware-nano33ble-images`),
flashed with Arduino's `bossac` 1.9.1-17-g89f3556. Its results files and
logs are in `hardware/results/nano33ble/`
([README](../../hardware/results/nano33ble/README.md)). Every step of
the [first session](#first-session-on-a-new-board) passed, and with them
what `hardware/nano33ble/src/main.rs` does on the chip:

- the clean-up of what the bootloader leaves on, the start of the clocks
  and of the USB device (`embassy_nrf::init` with the crystal, the
  embassy USB driver, its VBUS detection), and the loop that polls the
  USB futures by hand: the `runner` port appeared within 4 s of each
  flash, and answered;
- `read` and `write` on the CDC-ACM port: 376 cases, the single-operator
  models sent into the RAM, with no case lost;
- the cycle counter (`DWT.CYCCNT`) without a debugger (`counter_hz:
  64000000`, and 15 cases timed), and the 1 MHz timer (TIMER1) that
  gives the time and the watchdog;
- the panic and HardFault handlers, and the watchdog interrupt ending a
  hang after 20 s; `GPREGRET2` and the message at the top of the RAM
  across a reset through the bootloader (`--self-test`: the panic with
  its file and line, the fault with its `pc` and `lr`);
- the reset into the bootloader asked with the `RESET` command
  (`GPREGRET = 0xB0`; `--bootloader` in 1 s, and between the two images
  of each run);
- `bossac` on the real bootloader, its `-v` and `-R` included, and the
  detection of the ports by their USB identifiers;
- CMSIS-NN's DSP kernels on a real Cortex-M4F, reading their weights
  from flash: 373 of the 376 golden cases matched LiteRT (bit for bit,
  the float32 ones within their tolerance of 1e-5), the other 3 the known `ad01_int8` difference
  (issue #1).

It also corrected one assumption: the instruction cache of the flash
was on, not off ([the instruction cache](#the-instruction-cache)).

## What has never run on a board

- **The instruction cache off**: no time has been measured without it;
  the firmware cannot turn it off yet. The feature `icache` has not been
  built for the board either (on this bootloader it would change
  nothing).
- **The 1200 baud touch of this firmware** (`--bootloader` used the
  `RESET` command, which worked), and the touch of the factory sketch
  (step 2): the board ran a firmware that ignores it, and needed a
  double tap instead.
- **The return to the bootloader after 5 minutes without a request.**
- **The third crash in a row sending the board to its bootloader**, on a
  real crash loop, and exit code 4: the self-test crashes once per kind,
  with a host present.
- **The unexpected-interrupt handler**, and the watchdog interrupt
  preempting a USB interrupt.
- **The recovery in the middle of a run** (a case lost, the firmware
  asked why, or flashed again) and exit code 3: no case failed.
- **macOS** (the port names, Rosetta and the Intel `bossac`), a Linux
  aarch64 host (its `bossac` archive), and a `bossac` found in an
  Arduino IDE or `arduino-cli` installation: the session used the one
  mise installs, on Linux x86_64.
- **A second board**, another Nano 33 BLE or any other chip, and a
  second session on the same board.
- **Building the images on the computer of the board**
  (`mise run build-hardware-nano33ble` there): the session used CI's.

Elsewhere in the repository, as listed in
[issue #2](https://github.com/AlixANNERAUD/nanoglia/issues/2):
CMSIS-NN's Helium paths and ESP-NN's ESP32-S3 kernels have run nowhere,
and the other Cortex-M cores and RISC-V run under an instruction-set
emulator only, which models no timing.
