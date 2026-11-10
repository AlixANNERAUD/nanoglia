# nanoglia-tools

**Check on your computer whether [nanoglia](https://github.com/AlixANNERAUD/nanoglia) runs a TensorFlow Lite model, before flashing anything.** The `nanoglia` command lists every operator of a `.tflite` file with the kernel that runs it, or the reason no kernel does, the memory the model needs, and the Cargo features of a firmware that runs it. Loading a model stops at the first operator it cannot run; the check reports them all.

```console
$ nanoglia check testdata/mlperf-tiny/kws_ref_model.tflite --ram 256k
model: testdata/mlperf-tiny/kws_ref_model.tflite (53936 bytes)
checked with: backends cmsis-nn, kernels; features float, int16, all-ops

 #  operator              inputs                    outputs          kernel
 0  CONV_2D v3            int8[1,49,10,1] +2 const  int8[1,25,5,64]  cmsis-nn/conv2d_s8
 1  DEPTHWISE_CONV_2D v3  int8[1,25,5,64] +2 const  int8[1,25,5,64]  cmsis-nn/depthwise_conv2d_s8
[...]
10  RESHAPE v1            int8[1,1,1,64] +1 const   int8[1,64]       rust/reshape
11  FULLY_CONNECTED v4    int8[1,64] +2 const       int8[1,12]       cmsis-nn/fully_connected_s8
12  SOFTMAX v2            int8[1,12]                int8[1,12]       cmsis-nn/softmax_s8

RUNS: 13 of 13 operators have a kernel.

memory (bytes):
  graph block             7712  of which prepared kernel parameters 5984
  execution context      16160  activations 16000 + scratch 160
  one arena for both     23887  MemoryRequirements::arena_bytes(1)
  RAM budget            262144  fits, 238257 bytes left
  model                  53936  borrowed where it is stored, never copied
[...]
Cargo features of a firmware that runs this model with these kernels, for its Cargo.toml:
nanoglia = { version = "0.2.0", default-features = false, features = ["cmsis-nn", "kernels", "op-average-pool-2d", "op-conv-2d", "op-depthwise-conv-2d", "op-fully-connected", "op-reshape", "op-softmax"] }
```

(MLPerf Tiny keyword spotting, on a Linux aarch64 computer, 2026-11-10, on top of commit `d98b750`; the lines cut here and the closing notes are in the guide.)

Run it from a clone of the repository (`mise run submodules` first) with `cargo run -p nanoglia-tools -- check <MODEL.tflite>`, or, from the tool's first release on crates.io, install the `nanoglia` command with `cargo install nanoglia-tools` (a C compiler is needed: the tool builds CMSIS-NN and ESP-NN).

## Usage

```text
nanoglia check [OPTIONS] <MODEL.tflite>...
```

With several models, the models of one firmware, each gets its report, then the output ends with the Cargo features of a firmware that runs them all.

| Option | Meaning |
| --- | --- |
| `--backends <LIST>` | The backends of your firmware: a comma-separated list of `cmsis-nn`, `esp-nn` and `kernels`, the Cargo features of `nanoglia` with the same names. Default: `cmsis-nn,kernels`, the default features. They are tried in the order of `Registry::default()` (ESP-NN, CMSIS-NN, the Rust kernels), whatever the order of the list. |
| `--ram <SIZE>` | RAM budget for the graph and one execution context of each model, in bytes or with a binary suffix (`256k`, `1M`). The check fails when `MemoryRequirements::arena_bytes(1)` is larger. The model bytes are not counted: nanoglia borrows them where they are. |
| `--json` | One JSON document instead of the table. |
| `-h`, `--help`; `-V`, `--version` | Help; version. |

| Exit status | Meaning |
| ---: | --- |
| 0 | The model runs, and fits the budget if one is given; with several models, every one. |
| 1 | It does not (one of them does not): an operator has no kernel, the model cannot be loaded (not a TFLite model, several subgraphs...), or it needs more than the budget. |
| 2 | Wrong arguments, or a file cannot be read (message on the standard error). |

## What is checked, and with what

The tool calls `nanoglia::ModelReport::check_with`, which selects, plans and prepares the kernel of every operator as `Graph::load_with` does: a model that passes loads, given the memory. The verdict depends on what the firmware is built with, and the first lines of the output say what the tool used:

- **backends**: chosen at run time with `--backends`;
- **features**: `float`, `int16`, `single-rounding`, `all-ops` and the `op-*` features (one per operator) are Cargo features of the tool, as they are features of `nanoglia`. `float`, `int16` and `all-ops` are on by default, like in `nanoglia`; for an int8-only firmware, run `cargo run -p nanoglia-tools --no-default-features --features all-ops -- check <MODEL.tflite>`. For a firmware built with some operators only, name its `op-*` features in place of `all-ops`: the other operators are then reported without a kernel, with the reason `built without op-...`.

The check runs on your computer, not on the microcontroller:

- The kernels are those the firmware selects: selection is Rust code that does not depend on the target. One exception: ESP-NN claims no operator on targets without atomic compare-and-swap (Cortex-M0+, `riscv32imc`), where CMSIS-NN or the Rust kernels run its operators.
- Inside a kernel, the code differs: on the computer CMSIS-NN runs its portable C code and ESP-NN its ANSI C code; the Arm DSP and Helium paths and ESP-NN's ESP32-S3 kernels are built for those targets only.
- The sizes are those of the computer's build. On a 32-bit target the graph block is a little smaller (it holds pointers), and CMSIS-NN with DSP or Helium, or ESP-NN on an ESP32-S3, may ask for another scratch size: confirm the sizes on the device ([Size the memory](https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/how-to-guides/size-memory.md)).

## Cargo features of the firmware

When the model runs, the output ends with the dependency of a firmware that selects the same kernels, `nanoglia = { version = "...", default-features = false, features = [...] }`, version that of the `nanoglia` the tool was built on. It names:

- the backends of the selected kernels: `cmsis-nn`, `esp-nn`, and `kernels` for the `rust/...` ones;
- `float` when a selected kernel is one of the float32 kernels that only this feature adds (the convolutions, pooling, fully-connected, softmax, activations, `PRELU`, `RESIZE_BILINEAR` and the unary operators on f32), `int16` when it is one of the 16x8 kernels it adds (CMSIS-NN's `_s16` kernels, and the Rust 16x8 fully-connected, softmax, `LOGISTIC` and `TANH`). Other kernels run f32 or int16 tensors without them, the Rust `ADD` or `RESHAPE` for instance: the type of the tensors alone does not tell;
- `single-rounding`, when the tool was built with it;
- the `op-*` feature of each operator, in alphabetical order.

Without one of these features, the firmware would select other kernels or none. Turning off `float` or `int16` while another kernel could take over is not tried: the line is the one that reproduces this selection. With `esp-nn`, the line holds for targets with atomic compare-and-swap only: elsewhere ESP-NN claims no operator, and a check without `esp-nn` gives the features needed.

## JSON output

| Field | Value |
| --- | --- |
| `model` | `path` and size in `bytes` of the file. |
| `backends` | The backends, in the order they are tried. |
| `features` | `float`, `int16`, `single-rounding`: whether the tool was built with each. |
| `pointer_bits` | Pointer width of the build the sizes come from. |
| `passes` | Whether the exit status is 0. |
| `runs` | Whether the model loads. |
| `unsupported` | Number of operators without a kernel; `null` when the operators could not be listed. |
| `error` | Why the model does not load when no operator is to blame, else `null`: `code` (the wasi-nn name of the `ErrorCode`) and a `message`. |
| `memory` | When the model runs: `graph_bytes`, `context_bytes`, `persistent_bytes` (part of the graph block), `activation_bytes` and `scratch_bytes` (the two parts of a context), `arena_bytes` (`arena_bytes(1)`). Else `null`. |
| `ram_budget_bytes`, `fits` | The budget and whether `arena_bytes` fits it; `null` without `--ram` (and `fits` when the model does not run). |
| `firmware` | When the model runs: `nanoglia_version`, `features` (the list of [Cargo features of the firmware](#cargo-features-of-the-firmware)) and `dependency` (the line for `Cargo.toml`). Else `null`. |
| `operators` | One object per operator: `index`, `opcode` (TFLite builtin code, 32 for a custom operator), `name`, `custom`, `version`, `inputs` and `outputs` (`tensor` index, `type`, `shape`, `constant`; `null` for an omitted optional input), then `kernel`, `backend`, `persistent_bytes` and `scratch_bytes`, or `null` for these four and an `error` as above. |

With several models, the document is `{"models": [...], "passes": ..., "firmware": ...}`: the document of each model as above, whether every model passes, and the `firmware` of them all (`null` unless every model runs).

The guide [Check whether a model runs](https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/how-to-guides/check-whether-a-model-runs.md) shows the outputs and how to act on them.

## Dependencies

Only the nanoglia crates: the arguments (one command, a few options) and the JSON document are handled by hand. The three backends are always built, into a registry of the tool's own, so building the workspace with the tool does not change the features of the library the other crates are tested with.

## Versions

The tool is released with the libraries, under its own version number, and checks with the kernels of the `nanoglia` release it was built on (its `nanoglia` dependency on crates.io). Use a release of the tool built on the `nanoglia` version of your firmware.
