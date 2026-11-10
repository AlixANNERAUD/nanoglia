# Run on a microcontroller

This guide runs a model on a microcontroller without a heap, starting
from the example firmware for the Arduino Nano 33 BLE (Arm Cortex-M4F),
then adapting it to your own firmware, on Arm Cortex-M or on the
ESP32-S3.

> **Status: not yet validated on hardware.** The firmware builds in CI
> (`mise run example-nano33ble`, and `mise run cross-esp32s3` for the
> ESP32-S3 libraries) and the same code runs bit-exact on the host, where
> CMSIS-NN and ESP-NN compile their portable C code. The golden cases
> also pass under an instruction-set emulator for Cortex-M4F, M33 (with
> CMSIS-NN's DSP code paths), M0+ and riscv32imc (`mise run
> test-emulated`). Nothing has been run and compared with the golden
> outputs on a board yet, and CMSIS-NN's Helium paths and ESP-NN's
> ESP32-S3 kernels not even in an emulator: this is tracked in issue #2.
> Check the results on your device against the host before relying on
> them.

## The Nano 33 BLE example

`examples/nano33ble/` runs the MLPerf Tiny keyword spotting model on the
nRF52840 (Cortex-M4F, 1 MB flash, 256 KB RAM) of an Arduino Nano 33 BLE.
It loads the model from flash into a static 32 KiB `Arena`, prints over
RTT the memory used and the load time, then runs an inference every
second on a test pattern and prints the scores and the cycle count.

You need:

- the C cross compiler and the newlib headers: on Debian or Ubuntu, the
  packages `gcc-arm-none-eabi` and `libnewlib-arm-none-eabi` (`mise
  bootstrap packages apply` installs them);
- the Rust target `thumbv7em-none-eabihf`;
- [probe-rs](https://probe.rs) and a debug probe it supports, wired to
  the SWD pads on the bottom of the board.

Build it and print the firmware size:

```sh
mise run example-nano33ble
```

Flash and run it through the probe:

```sh
cd examples/nano33ble
cargo run --release
```

`cargo run` calls `probe-rs run --chip nRF52840_xxAA` (set in
`examples/nano33ble/.cargo/config.toml`), which flashes the firmware and
prints its `defmt` log. To see the kernel chosen for each operator, raise
the log level:

```sh
DEFMT_LOG=debug cargo run --release
```

The application is linked at address `0x10000`, after the 64 KiB of the
Arduino bootloader (`examples/nano33ble/memory.x`), so the bootloader
stays intact and usable.

## Use nanoglia in your firmware

### 1. Choose the features

For an int8 model, leave out the float32 and int16 kernels to save flash,
and keep `kernels` for the operators CMSIS-NN does not have (such as
`RESHAPE`). Without the default features, name the operators too:
`all-ops` for all of them,

```toml
[dependencies]
nanoglia = { git = "https://github.com/AlixANNERAUD/nanoglia", default-features = false, features = ["cmsis-nn", "kernels", "all-ops"] }
```

or, to link only the kernels your models need, the `op-*` feature of
each of their operators in place of `all-ops`, as the Nano 33 BLE example
does ([Reduce the code size](reduce-code-size.md)). With neither, no
model loads.

Add `defmt` to format nanoglia's errors with `defmt`. Leave out `alloc`:
nothing then needs a global allocator. [Cargo features](../reference/features.md)
lists what each feature brings and what stays without it; check that
your model's operators stay supported
([Supported operators](../reference/supported-operators.md)).

### 2. Give CMSIS-NN a C compiler

CMSIS-NN is compiled by the [`cc`](https://crates.io/crates/cc) crate.
It needs a C compiler for the target and the target's C headers
(`stdint.h`, `string.h`...); nothing links a C library, and CMSIS-NN
calls no `malloc`. Set the usual `cc` variables, for example in
`.cargo/config.toml`:

```toml
[env]
CC_thumbv7em_none_eabihf = "arm-none-eabi-gcc"
AR_thumbv7em_none_eabihf = "arm-none-eabi-ar"
CFLAGS_thumbv7em_none_eabihf = "-isystem /usr/include/newlib"
```

`/usr/include/newlib` is where Debian and Ubuntu install newlib's
headers. The build script adds the DSP flags for Armv7E-M (Cortex-M4,
M7), where the DSP extension is always present. On Armv8-M Mainline
(Cortex-M33) and Armv8.1-M (Cortex-M55, M85) it is optional, so the
build script uses it, and Helium, only when the Rust target enables it:
build with `-C target-cpu=cortex-m33` (or `-C target-feature=+dsp`, and
`+mve` for Helium), otherwise CMSIS-NN runs its portable C code. The same variables, with the target name
changed, work for `thumbv6m-none-eabi` (Cortex-M0+, no DSP).

For `thumbv8m.main-none-eabihf`, whose name contains a dot, `cc` also
reads `CC_thumbv8m.main-none-eabihf`; if a wrapper of your build drops
such variable names, use `TARGET_CC`, `TARGET_AR` and `TARGET_CFLAGS`
instead, as `mise run cross-arm` does.

### 3. Place the model and the memory

```rust
use nanoglia::{Arena, Encoding, Graph, Target};

/// The model, in flash, on a 16-byte boundary.
#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);

static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!("model.tflite"));

/// The graph and one execution context.
static mut MEMORY: [u8; 32 * 1024] = [0; 32 * 1024];
```

The model is borrowed, never copied: it stays in flash. The alignment
keeps its constant buffers aligned for their types. Size `MEMORY` with
[Size the memory](size-memory.md).

### 4. Load once, compute in a loop

```rust
// SAFETY: this runs once, and nothing else uses `MEMORY`.
let arena = Arena::new(unsafe { &mut *core::ptr::addr_of_mut!(MEMORY) });
let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
let mut ctx = graph.init_execution_context_in(&arena)?;

loop {
    // Write the inputs before every compute: the memory plan reuses
    // their memory once the first operators have consumed them.
    ctx.input_mut(0)?.as_mut_slice::<i8>()?.copy_from_slice(&features);
    ctx.compute()?; // allocates nothing
    let scores = ctx.output(0)?;
    let scores = scores.as_slice::<i8>()?;
    // ...
}
```

`load_in` and `init_execution_context_in` make one allocation each from
the arena; `compute` makes none. Keep the graph and the context for the
life of the program, and write the inputs again before each `compute`,
even when they did not change.

`Arena` needs no global allocator and no critical section: it lives on
the stack of the function that creates it, and the graph and context
borrow it.

### 5. Check the load errors

If loading fails, the error says why and where
([Errors](../reference/errors.md)). With `defmt`:

```rust
let graph = match Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena) {
    Ok(graph) => graph,
    Err(error) => defmt::panic!("load failed: {}", error),
};
```

`Detail::UnsupportedOp { op_index, opcode }` names the first operator no
enabled backend supports, or whose `op-*` feature is off;
`ErrorCode::TooLarge` means `MEMORY` is too small.

## On an ESP32-S3

ESP-NN, Espressif's kernel library, has optimized kernels for the
ESP32-S3. Enable the `esp-nn` feature; keep `cmsis-nn` for the
operators ESP-NN refuses (dilated convolutions, for example) and
`kernels` for the rest:

```toml
[dependencies]
nanoglia = { git = "https://github.com/AlixANNERAUD/nanoglia", default-features = false, features = ["esp-nn", "cmsis-nn", "kernels", "all-ops"] }
```

The target, `xtensa-esp32s3-none-elf`, needs Espressif's Rust toolchain
and C compiler, both installed by
[espup](https://github.com/esp-rs/espup):

```sh
espup install --targets esp32s3
. $HOME/export-esp.sh
export CC_xtensa_esp32s3_none_elf=xtensa-esp32s3-elf-gcc
export AR_xtensa_esp32s3_none_elf=xtensa-esp32s3-elf-ar
cargo +esp build --release --target xtensa-esp32s3-none-elf -Zbuild-std=core
```

This is what `mise run install-esp` and `mise run cross-esp32s3` do in
nanoglia's repository. ESP-NN is compiled without ESP-IDF: it needs no
FreeRTOS, no heap and no C library. The rest of this guide applies
unchanged (static model, `Arena`, `load_in`, `compute` in a loop).

ESP-NN's scratch buffers are global, so only one context runs it at a
time: when two tasks or the two cores compute at once, one of them runs
the CMSIS-NN kernels for those operators, with the same results
([Supported operators](../reference/supported-operators.md#esp-nn)). The
graph keeps the parameters of both, so it needs more memory than with
CMSIS-NN alone ([Memory](../reference/memory.md#mlperf-tiny-footprints)).

## See also

- [Your first inference](../tutorials/first-inference.md), the same
  model on the host.
- [Development tasks](../reference/development-tasks.md):
  `cross-arm`, `cross-riscv32`, `no-malloc`.
