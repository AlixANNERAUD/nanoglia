# Check whether a model runs

`Graph::load_in` stops at the first operator nanoglia cannot run. This
guide answers the question before you flash anything, for the whole
model at once: which kernel runs each operator, which operators block
and why, and how much memory the model needs. It uses the `nanoglia`
command-line tool on your computer, then the same check from Rust.

The outputs below come from a Linux aarch64 computer on 2026-11-10, with
the change that added the tool (on top of commit `136ee56`), on the
MLPerf Tiny keyword spotting model of `testdata/mlperf-tiny/`.

## 1. Run the check

The tool is the crate `nanoglia-tools`. Install its `nanoglia` command
from crates.io (a C compiler is needed: the tool builds CMSIS-NN and
ESP-NN):

```sh
cargo install nanoglia-tools
nanoglia check model.tflite
```

Use a release of the tool built on the `nanoglia` version of your
firmware ([Versions](../../crates/nanoglia-tools/README.md#versions)).
With the installed tool, replace `cargo run -p nanoglia-tools --` in the
commands below with `nanoglia`, and give the build options to
`cargo install` (`cargo install nanoglia-tools --no-default-features
--features all-ops`). The commands below run the tool from a clone of the
repository (with the submodules: `mise run submodules`):

```sh
cargo run -p nanoglia-tools -- check testdata/mlperf-tiny/kws_ref_model.tflite
```

```text
model: testdata/mlperf-tiny/kws_ref_model.tflite (53936 bytes)
checked with: backends cmsis-nn, kernels; features float, int16, all-ops

 #  operator              inputs                    outputs          kernel
 0  CONV_2D v3            int8[1,49,10,1] +2 const  int8[1,25,5,64]  cmsis-nn/conv2d_s8
 1  DEPTHWISE_CONV_2D v3  int8[1,25,5,64] +2 const  int8[1,25,5,64]  cmsis-nn/depthwise_conv2d_s8
 2  CONV_2D v3            int8[1,25,5,64] +2 const  int8[1,25,5,64]  cmsis-nn/conv2d_s8
 3  DEPTHWISE_CONV_2D v3  int8[1,25,5,64] +2 const  int8[1,25,5,64]  cmsis-nn/depthwise_conv2d_s8
 4  CONV_2D v3            int8[1,25,5,64] +2 const  int8[1,25,5,64]  cmsis-nn/conv2d_s8
 5  DEPTHWISE_CONV_2D v3  int8[1,25,5,64] +2 const  int8[1,25,5,64]  cmsis-nn/depthwise_conv2d_s8
 6  CONV_2D v3            int8[1,25,5,64] +2 const  int8[1,25,5,64]  cmsis-nn/conv2d_s8
 7  DEPTHWISE_CONV_2D v3  int8[1,25,5,64] +2 const  int8[1,25,5,64]  cmsis-nn/depthwise_conv2d_s8
 8  CONV_2D v3            int8[1,25,5,64] +2 const  int8[1,25,5,64]  cmsis-nn/conv2d_s8
 9  AVERAGE_POOL_2D v2    int8[1,25,5,64]           int8[1,1,1,64]   cmsis-nn/average_pool2d_s8
10  RESHAPE v1            int8[1,1,1,64] +1 const   int8[1,64]       rust/reshape
11  FULLY_CONNECTED v4    int8[1,64] +2 const       int8[1,12]       cmsis-nn/fully_connected_s8
12  SOFTMAX v2            int8[1,12]                int8[1,12]       cmsis-nn/softmax_s8

RUNS: 13 of 13 operators have a kernel.

memory (bytes):
  graph block             7712  of which prepared kernel parameters 5984
  execution context      16160  activations 16000 + scratch 160
  one arena for both     23887  MemoryRequirements::arena_bytes(1)
  model                  53936  borrowed where it is stored, never copied

Kernel selection does not depend on the target: a firmware built with these backends and
features selects these kernels.
The sizes are those of this 64-bit build: on a 32-bit target the graph block is a little
smaller, and CMSIS-NN with DSP or Helium, or ESP-NN on an ESP32-S3, may ask for another
scratch size.
```

Each row is an operator, in execution order: its name and version in the
model, the type and shape of the tensors it computes on (constants such
as weights and biases are only counted), and the kernel that runs it.
The name of a kernel starts with its backend: `cmsis-nn`, `esp-nn`, or
`rust` for the Rust kernels.

The exit status is 0 when the model runs, 1 when it does not, 2 when the
arguments are wrong or the file cannot be read.

## 2. Check with the backends and features of your firmware

Whether an operator has a kernel depends on what your firmware is built
with ([Cargo features](../reference/features.md)). The second line of
the output says what the check used; make it match:

- **Backends**, at run time: `--backends` takes the Cargo features of
  `nanoglia` that add a backend, `cmsis-nn`, `esp-nn` and `kernels`
  (default: `cmsis-nn,kernels`, as in `nanoglia`). For an ESP32-S3
  firmware with feature `esp-nn`:

  ```sh
  cargo run -p nanoglia-tools -- check model.tflite --backends esp-nn,cmsis-nn,kernels
  ```

  The int8 convolutions, pooling, fully-connected, softmax, `ADD` and
  `MUL` then show `esp-nn/...` kernels, and the graph block grows (13664
  bytes instead of 7712 for keyword spotting), since each of these
  operators also keeps the parameters of its CMSIS-NN fallback
  ([Memory](../reference/memory.md#mlperf-tiny-footprints)).

- **`float`, `int16`, `single-rounding`**, at build time: they are
  features of the tool too, with the defaults of `nanoglia` (`float` and
  `int16` on). For an int8-only firmware (`default-features = false,
  features = ["cmsis-nn", "kernels", "all-ops"]`):

  ```sh
  cargo run -p nanoglia-tools --no-default-features --features all-ops -- check model.tflite
  ```

- **`all-ops` and the `op-*` features**, at build time too: the tool
  checks with every operator by default. Built with the `op-*` features
  of a firmware in place of `all-ops`, it reports the operators that
  firmware leaves out without a kernel
  ([Reduce the code size](reduce-code-size.md)).

## 3. Read a model that does not run

Every operator without a kernel is listed, not only the first. Here the
same model checked for a firmware without CMSIS-NN (`--backends
kernels`), which the Rust kernels cannot run alone:

```text
model: testdata/mlperf-tiny/kws_ref_model.tflite (53936 bytes)
checked with: backends kernels; features float, int16, all-ops

 #  operator              inputs                    outputs          kernel
 0  CONV_2D v3            int8[1,49,10,1] +2 const  int8[1,25,5,64]  NONE: no kernel for these types, shapes and options
 1  DEPTHWISE_CONV_2D v3  int8[1,25,5,64] +2 const  int8[1,25,5,64]  NONE: no kernel for these types, shapes and options
[...]
 9  AVERAGE_POOL_2D v2    int8[1,25,5,64]           int8[1,1,1,64]   NONE: no kernel for these types, shapes and options
10  RESHAPE v1            int8[1,1,1,64] +1 const   int8[1,64]       rust/reshape
11  FULLY_CONNECTED v4    int8[1,64] +2 const       int8[1,12]       NONE: no kernel for these types, shapes and options
12  SOFTMAX v2            int8[1,12]                int8[1,12]       NONE: no kernel for these types, shapes and options

DOES NOT RUN: 12 of 13 operators have no kernel (CONV_2D v3 x5, DEPTHWISE_CONV_2D v3 x4, AVERAGE_POOL_2D v2 x1, FULLY_CONNECTED v4 x1, SOFTMAX v2 x1).
What each backend runs, with which types and restrictions:
https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/reference/supported-operators.md
```

For each blocking operator, in this order:

1. Look it up in [Supported operators](../reference/supported-operators.md)
   with the types and shapes of the row. If a backend or a feature you
   left out runs it, enable it (here `cmsis-nn`).
2. If nanoglia runs the operator but not this variant (a type, a
   dilation, a broadcast, an option), change the export: quantize to
   int8, use static shapes, or replace the layer
   ([Convert a PyTorch model](convert-a-pytorch-model.md)).
3. If nanoglia does not have the operator, replace it in the model, or
   provide a kernel for it ([Write a backend](write-a-backend.md)).

Other reasons appear in the same column: a tensor type nanoglia cannot
represent (strings, sparse tensors...), or an unsupported structure
(variable tensors of stateful layers, more than 16 inputs or 8
outputs). When the model cannot be walked at all, the table is replaced
by the reason: not a TFLite model, a truncated file, or several
subgraphs (control flow). [Errors](../reference/errors.md#checking-modelreportcheck-check_in-check_with)
lists them.

## 4. Give a RAM budget

`--ram` fails the check when the graph and one execution context need
more than the RAM you can give them (bytes, or `k` and `M` for multiples
of 1024):

```sh
cargo run -p nanoglia-tools -- check testdata/mlperf-tiny/kws_ref_model.tflite --ram 16k
```

```text
DOES NOT FIT: every operator has a kernel, but the graph and one execution context need 23887 bytes, 7503 more than the budget of 16384.

memory (bytes):
  graph block             7712  of which prepared kernel parameters 5984
  execution context      16160  activations 16000 + scratch 160
  one arena for both     23887  MemoryRequirements::arena_bytes(1)
  RAM budget             16384  7503 bytes short
  model                  53936  borrowed where it is stored, never copied
```

The amount compared is `MemoryRequirements::arena_bytes(1)`: the size of
one `Arena` buffer for the graph block and one execution context
([Size the memory](size-memory.md)). The model bytes are not part of
it: nanoglia reads them where your application keeps them (flash, or
RAM if it receives the model at run time), so count them yourself in
the second case. The sizes are those of your computer's build, see
[the last section](#what-the-check-cannot-tell).

## 5. Use it in a script or in CI

`--json` prints one JSON document with the same content: `runs`,
`passes` (runs and fits the budget), `unsupported`, `memory`, and one
object per operator with its tensors and its `kernel` or its `error`.
For example, the operators that block, with [jq](https://jqlang.org):

```sh
cargo run -q -p nanoglia-tools -- check model.tflite --json --backends kernels \
  | jq -r '.operators[] | select(.error) | "\(.index) \(.name): \(.error.message)"'
```

```text
0 CONV_2D: no selected backend runs this operator with these types, shapes and options
1 DEPTHWISE_CONV_2D: no selected backend runs this operator with these types, shapes and options
[...]
```

The fields are listed in the
[README of the tool](../../crates/nanoglia-tools/README.md#json-output).
The exit status alone is enough to fail a build when a new export of
the model stops running or outgrows its budget.

## 6. Run the same check from Rust

The tool is a thin layer over `nanoglia::ModelReport`, which you can
call from a test of your own crate. This is the most faithful check: it
uses exactly the nanoglia version, features and `Registry` of your
firmware, your own backends included. With the features of your
firmware plus `alloc`:

```rust
use nanoglia::{Encoding, ModelReport, OpCode, Target};

#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);

static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!("../kws_ref_model.tflite"));

#[test]
fn model_runs() {
    let report = ModelReport::check(&[&MODEL.0], Encoding::Autodetect, Target::Auto, |op| {
        let name = match op.opcode() {
            OpCode::Builtin(builtin) => builtin.name().unwrap_or("?"),
            OpCode::Custom(name) => name,
        };
        match op.kernel() {
            Ok(kernel) => println!("{:>2} {name}: {}", op.index(), kernel.name),
            Err(error) => println!("{:>2} {name}: no kernel ({error})", op.index()),
        }
    })
    .unwrap();
    assert_eq!(report.unsupported, 0, "operators without a kernel");
    println!("{:?}", report.memory.unwrap());
}
```

```text
 0 CONV_2D: cmsis-nn/conv2d_s8
 1 DEPTHWISE_CONV_2D: cmsis-nn/depthwise_conv2d_s8
[...]
12 SOFTMAX: cmsis-nn/softmax_s8
MemoryBreakdown { graph_bytes: 7712, context_bytes: 16160, persistent_bytes: 5984, activation_bytes: 16000, scratch_bytes: 160 }
```

The closure receives an `OperatorReport` per operator: `index()`,
`opcode()`, `version()`, the descriptions of its tensors (`input(i)`,
`output(i)`: type, shape, quantization) and `kernel()`, which is the
selected kernel (name, backend, the persistent and scratch bytes it
asks for) or the error loading would return for this operator.
`ModelReport` then holds the number of operators, the number without a
kernel, and `memory`: the `MemoryBreakdown` of a model that loads, or
the error `Graph::load_in` returns.

Like the rest of the library it needs no heap: `ModelReport::check_in`
takes its working memory from any allocator, such as an `Arena` (one
block of 20 bytes per tensor plus the largest prepared parameters of an
operator, freed before returning), and `check_with` takes a `Registry`
of your own. So the check also runs on the device, for a model received
at run time, before loading it.

## What the check cannot tell

The check runs on your computer. It tells exactly whether the model
loads, and with which kernels, with one exception: ESP-NN claims no
operator on targets without atomic compare-and-swap (Cortex-M0+,
`riscv32imc`), where the next backend runs its operators
([Supported operators](../reference/supported-operators.md#esp-nn)).
It does not tell:

- **The sizes on the device.** The graph block holds pointers, so a
  32-bit target needs a little less than a 64-bit computer (7616 bytes
  instead of 7712 for keyword spotting). The scratch size is the one of
  the code the computer runs, CMSIS-NN's portable C and ESP-NN's ANSI C:
  the Arm DSP and Helium paths of CMSIS-NN and ESP-NN's ESP32-S3 kernels
  may ask for another one. Leave a margin on a tight `--ram` budget and
  confirm on the device ([Size the memory](size-memory.md#3-account-for-the-target)).
- **The stack**, which the kernels use besides the two blocks.
- **The speed** ([Performance](../reference/performance.md) has host and
  wasmtime timings, and those of the five MLPerf Tiny models on one
  Arduino Nano 33 BLE).
- **That the outputs are right on your chip.** The outputs are checked
  bit for bit against LiteRT on the host, under WebAssembly and under
  instruction-set emulation of Cortex-M and RISC-V cores, and the golden
  cases have run on one board (a Nano 33 BLE: Cortex-M4F, CMSIS-NN's
  DSP paths); the Helium paths, ESP-NN's ESP32-S3 kernels and every
  other chip have not run on hardware yet
  ([issue #2](https://github.com/AlixANNERAUD/nanoglia/issues/2)).

## See also

- [Supported operators](../reference/supported-operators.md): what each
  backend runs.
- [Size the memory](size-memory.md): from the sizes to a static buffer.
- [Errors](../reference/errors.md): every reason a model is refused.
