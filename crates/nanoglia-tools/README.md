# nanoglia-tools

**Check on your computer whether [nanoglia](https://github.com/AlixANNERAUD/nanoglia) runs a TensorFlow Lite model, before flashing anything.** The `nanoglia` command lists every operator of a `.tflite` file with the kernel that runs it, or the reason no kernel does, and the memory the model needs. Loading a model stops at the first operator it cannot run; the check reports them all.

```console
$ nanoglia check testdata/mlperf-tiny/kws_ref_model.tflite --ram 256k
model: testdata/mlperf-tiny/kws_ref_model.tflite (53936 bytes)
checked with: backends cmsis-nn, kernels; features float, int16

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
```

(MLPerf Tiny keyword spotting, on a Linux aarch64 computer, 2026-11-10; the lines cut here and the closing notes are in the guide.)

Install the `nanoglia` command with `cargo install nanoglia-tools` (a C compiler is needed: the tool builds CMSIS-NN and ESP-NN), or run it from a clone of the repository (`mise run submodules` first) with `cargo run -p nanoglia-tools -- check <MODEL.tflite>`.

## Usage

```text
nanoglia check [OPTIONS] <MODEL.tflite>
```

| Option | Meaning |
| --- | --- |
| `--backends <LIST>` | The backends of your firmware: a comma-separated list of `cmsis-nn`, `esp-nn` and `kernels`, the Cargo features of `nanoglia` with the same names. Default: `cmsis-nn,kernels`, the default features. They are tried in the order of `Registry::default()` (ESP-NN, CMSIS-NN, the Rust kernels), whatever the order of the list. |
| `--ram <SIZE>` | RAM budget for the graph and one execution context, in bytes or with a binary suffix (`256k`, `1M`). The check fails when `MemoryRequirements::arena_bytes(1)` is larger. The model bytes are not counted: nanoglia borrows them where they are. |
| `--json` | One JSON document instead of the table. |
| `-h`, `--help`; `-V`, `--version` | Help; version. |

| Exit status | Meaning |
| ---: | --- |
| 0 | The model runs, and fits the budget if one is given. |
| 1 | It does not: an operator has no kernel, the model cannot be loaded (not a TFLite model, several subgraphs...), or it needs more than the budget. |
| 2 | Wrong arguments, or the file cannot be read (message on the standard error). |

## What is checked, and with what

The tool calls `nanoglia::ModelReport::check_with`, which selects, plans and prepares the kernel of every operator as `Graph::load_with` does: a model that passes loads, given the memory. The verdict depends on what the firmware is built with, and the first lines of the output say what the tool used:

- **backends**: chosen at run time with `--backends`;
- **features**: `float`, `int16` and `single-rounding` are Cargo features of the tool, as they are features of `nanoglia`. Both are on by default, like in `nanoglia`; for an int8-only firmware, run `cargo run -p nanoglia-tools --no-default-features -- check <MODEL.tflite>`.

The check runs on your computer, not on the microcontroller:

- The kernels are those the firmware selects: selection is Rust code that does not depend on the target. One exception: ESP-NN claims no operator on targets without atomic compare-and-swap (Cortex-M0+, `riscv32imc`), where CMSIS-NN or the Rust kernels run its operators.
- Inside a kernel, the code differs: on the computer CMSIS-NN runs its portable C code and ESP-NN its ANSI C code; the Arm DSP and Helium paths and ESP-NN's ESP32-S3 kernels are built for those targets only.
- The sizes are those of the computer's build. On a 32-bit target the graph block is a little smaller (it holds pointers), and CMSIS-NN with DSP or Helium, or ESP-NN on an ESP32-S3, may ask for another scratch size: confirm the sizes on the device ([Size the memory](https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/how-to-guides/size-memory.md)).

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
| `error` | Why the model does not load when no operator is to blame, else `null`: `code` (the wasi-nn name of the `ErrorCode`), `detail` (the `Detail`) and a `message`. |
| `memory` | When the model runs: `graph_bytes`, `context_bytes`, `persistent_bytes` (part of the graph block), `activation_bytes` and `scratch_bytes` (the two parts of a context), `arena_bytes` (`arena_bytes(1)`). Else `null`. |
| `ram_budget_bytes`, `fits` | The budget and whether `arena_bytes` fits it; `null` without `--ram` (and `fits` when the model does not run). |
| `operators` | One object per operator: `index`, `opcode` (TFLite builtin code, 32 for a custom operator), `name`, `custom`, `version`, `inputs` and `outputs` (`tensor` index, `type`, `shape`, `constant`; `null` for an omitted optional input), then `kernel`, `backend`, `persistent_bytes` and `scratch_bytes`, or `null` for these four and an `error` as above. |

The guide [Check whether a model runs](https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/how-to-guides/check-whether-a-model-runs.md) shows the outputs and how to act on them.

## Dependencies

Only the nanoglia crates: the arguments (one command, three options) and the JSON document are handled by hand. The three backends are always built, into a registry of the tool's own, so building the workspace with the tool does not change the features of the library the other crates are tested with.

## Versions

The tool is released with the libraries and checks with the kernels of its own version: install the version of `nanoglia-tools` that matches the `nanoglia` your firmware uses.
