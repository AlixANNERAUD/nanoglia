# Reduce the code size

nanoglia loads models at run time, so the linker cannot know which
kernels your firmware needs and keeps them all. This guide links only
the kernels of the operators your models use, with the `op-*` Cargo
features. The sizes this saves on a Cortex-M4F firmware are in
[Memory](../reference/memory.md#code-size).

You need the `.tflite` files of every model the firmware will load. A
model that uses an operator you left out fails to load.

## 1. Get the features from `nanoglia check`

Run `nanoglia check` on every model the firmware loads, in one command
([Check whether a model runs](check-whether-a-model-runs.md)); from a
clone of the repository, for the keyword spotting model alone:

```sh
cargo run -p nanoglia-tools -- check testdata/mlperf-tiny/kws_ref_model.tflite
```

Each line of the table is an operator of the model, with the kernel
that runs it:

```text
 #  operator              inputs                    outputs          kernel
 0  CONV_2D v3            int8[1,49,10,1] +2 const  int8[1,25,5,64]  cmsis-nn/conv2d_s8
 1  DEPTHWISE_CONV_2D v3  int8[1,25,5,64] +2 const  int8[1,25,5,64]  cmsis-nn/depthwise_conv2d_s8
...
 9  AVERAGE_POOL_2D v2    int8[1,25,5,64]           int8[1,1,1,64]   cmsis-nn/average_pool2d_s8
10  RESHAPE v1            int8[1,1,1,64] +1 const   int8[1,64]       rust/reshape
11  FULLY_CONNECTED v4    int8[1,64] +2 const       int8[1,12]       cmsis-nn/fully_connected_s8
12  SOFTMAX v2            int8[1,12]                int8[1,12]       cmsis-nn/softmax_s8
```

The output ends with the dependency of a firmware that runs the model
with these kernels, and links nothing else:

```text
Cargo features of a firmware that runs this model with these kernels, for its Cargo.toml:
nanoglia = { version = "0.2.0", default-features = false, features = ["cmsis-nn", "kernels", "op-average-pool-2d", "op-conv-2d", "op-depthwise-conv-2d", "op-fully-connected", "op-reshape", "op-softmax"] }
```

Paste it into the `[dependencies]` of your firmware, in place of your
`nanoglia` line. With several models (`check a.tflite b.tflite`), the
last line of the output is the one of a firmware that runs them all.
(Tool output from a Linux aarch64 computer on 2026-11-10, on top of
commit `d98b750`.)

## 2. Understand the features, or write them by hand

The line of step 1 turns the default features off, since they include
`all-ops`, every operator, and names the backends, the types and the
operators. The same dependency, one feature per line:

```toml
[dependencies]
nanoglia = { git = "https://github.com/AlixANNERAUD/nanoglia", default-features = false, features = [
  "cmsis-nn",
  "kernels",
  "op-average-pool-2d",
  "op-conv-2d",
  "op-depthwise-conv-2d",
  "op-fully-connected",
  "op-reshape",
  "op-softmax",
] }
```

- **Backends.** The backends of the `kernel` column: `cmsis-nn` for
  `cmsis-nn/...`, `kernels` for `rust/...`, and `esp-nn` for
  `esp-nn/...`. An operator feature reaches every backend you enable.
- **Types.** The model above is int8, so `float` and `int16` are left
  out. A float32 model needs `float` and a 16x8 model `int16`: an
  operator feature covers every type, but its float32 and int16 kernels
  are only built with these features
  ([Cargo features](../reference/features.md)). Some operators run these
  types without them (the Rust `ADD` on f32, `RESHAPE` on any type), so
  `nanoglia check` names `float` or `int16` only when a kernel it
  selected needs it.
- **Operators.** One `op-*` feature per operator of the table: its name
  in lower case, with dashes in place of underscores, after `op-`.
  `CONV_2D` gives `op-conv-2d`, `RELU_N1_TO_1` gives `op-relu-n1-to-1`;
  in a program, `BuiltinOp::feature()` returns it. The table of
  [Cargo features](../reference/features.md#operator-features) lists
  them all.

`examples/nano33ble/Cargo.toml` is this configuration, and
`examples/train-to-device/Cargo.toml` the one of the gesture classifier
of the [tutorial](../tutorials/train-and-deploy.md).

From C, the CMake project takes the operator names instead
([Use nanoglia from C](use-from-c.md)):

```sh
cmake -S crates/nanoglia-c -B build-firmware -DNANOGLIA_CARGO_TARGET=thumbv7em-none-eabihf \
      -DNANOGLIA_INT8_ONLY=ON \
      "-DNANOGLIA_OPERATORS=CONV_2D;DEPTHWISE_CONV_2D;AVERAGE_POOL_2D;RESHAPE;FULLY_CONNECTED;SOFTMAX"
```

## 3. Check that your models still load

Load every model with the new features before flashing: in a host test
of your crate, or with the tool built with the same features, which
then reports an operator you forgot without a kernel:

```sh
cargo run -p nanoglia-tools --no-default-features \
    --features op-average-pool-2d,op-conv-2d,op-depthwise-conv-2d,op-fully-connected,op-reshape,op-softmax \
    -- check testdata/mlperf-tiny/kws_ref_model.tflite
```

It must say `RUNS: 13 of 13 operators have a kernel.` An operator you
left out is reported with the feature to add; here the ResNet model of
`testdata/mlperf-tiny/`, checked with the same build:

```text
 3  ADD v2              int8[1,32,32,16] int8[1,32,32,16]  int8[1,32,32,16]  NONE: built without `op-add`
[...]
DOES NOT RUN: 3 of 16 operators have no kernel (ADD v2 x3).
```

On the device, a model with an operator that was left out fails in
`Graph::load_in` with `Detail::UnsupportedOp { op_index, opcode }`,
which names the operator ([Errors](../reference/errors.md)); it never
panics. `BuiltinOp(opcode).name()` and `.feature()` give its name and
the feature to add.

## 4. Measure

Compare the sizes before and after:

```sh
cargo build --release
arm-none-eabi-size target/thumbv7em-none-eabihf/release/your-firmware
```

`text` is the flash used by code and constants, your model included if
it is a constant of the program.

## What stays

- Code that does not depend on the operators: the model verifier and
  reader, the memory planner, the interpreter.
- Code shared by several operators, when one of them is on: for
  example the Rust `ADD` kernel also computes `SUB`, `MUL` and `DIV`.
  The backend still refuses the operators whose feature is off.
- When a firmware must run models it does not know yet, keep `all-ops`:
  a model received later may use any operator.

Other savings: `opt-level = "s"` or `"z"`, `lto = "fat"` and
`codegen-units = 1` in the release profile, as in
`examples/nano33ble/Cargo.toml`.
