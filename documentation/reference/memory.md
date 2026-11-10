# Memory

What nanoglia allocates, when, from where, and how much the MLPerf Tiny
models need. To size a buffer for your own model, follow
[Size the memory](../how-to-guides/size-memory.md).

## The contract

nanoglia has no allocator of its own. Every byte of dynamic memory comes
from the allocator your application passes, and the number of
allocations is fixed:

| Call | Allocations | Size |
| --- | --- | --- |
| `Graph::load_in` (and `load`, `load_with`) | 1, kept until the graph is dropped | `graph_bytes()` |
| `Graph::init_execution_context_in` (and `init_execution_context`) | 1 per context, kept until the context is dropped | `context_bytes()` |
| `ExecutionContext::compute` (and `compute_with`) | none (an `Observer` may allocate on its own) | — |
| `ExecutionContext::set_input`, `input_mut`, `output`, `output_by_name` | none | — |
| `MemoryRequirements::compute_in` (and `compute`, `compute_with`) | 1, freed before returning | the planner's working memory: 20 bytes per tensor of the model |
| `ModelReport::check_in` (and `check`, `check_with`) | 1, freed before returning (the closure it calls may allocate on its own) | the planner's working memory, rounded up to 16 bytes, plus the largest prepared parameters of an operator: each kernel is prepared there in turn |

The CMSIS-NN C code calls no `malloc`: `mise run no-malloc` checks that
the C libraries built for the Arm and RISC-V targets reference no C heap
function. `crates/nanoglia/tests/memory.rs` counts the allocations of a
graph, a context and two `compute` calls; `crates/nanoglia/tests/check.rs`
those of a check, and that the sizes it reports are those a graph and a
context then allocate.

The C API (`nanoglia-c`) keeps the same contract with one buffer from
the caller: `nanoglia_load()` places its handle (the graph and one
context, 448 bytes on a 64-bit host, 224 on `thumbv7em-none-eabihf`)
there, followed by an `Arena` holding the two blocks, so it needs
`arena_bytes(1)` plus the handle
([Use nanoglia from C](../how-to-guides/use-from-c.md#the-memory)).

### Owning models and execution contexts

`OwnedGraph::load_in(model, encoding, target, allocator)` takes an aligned
`Block` containing model bytes, without copying them. It makes the same prepared-graph
allocation as `Graph::load_in`; the model allocation is supplied by the caller and is
released after the prepared graph. Model views borrow the owner, including constant outputs.
The owning loader uses the default static kernel registry; borrowed registries still use
`Graph::load_with`.

`GraphExecutionContext::new_in(handle, allocator)` retains a stable graph handle and
makes one arena allocation, just like the original borrowed `ExecutionContext`.
Handles may be borrowed graphs, graphs moved by value, `OwnedGraph`, or (`alloc`)
`alloc::sync::Arc` handles shared between contexts on targets with pointer atomics.
Custom shared pointers can implement the unsafe `GraphHandle` contract: every borrow
must refer to the same immutable prepared graph and model. Creating the `Arc` is the caller's
allocation; cloning it does not allocate. The borrowed `ExecutionContext` alias remains available
without `alloc` and adds no ownership wrapper. `graph()` is available on every context;
its graph and model views borrow the context. Unlike the previous borrowed-only accessor,
these views cannot be retained past dropping the context, even if its graph lives longer.
Borrow the original graph directly when that longer lifetime is required. The handle cannot switch graphs while
its arena exists.

For a file-loaded model, allocate one aligned `Block`, read the file into its mutable
slice, then move it into `OwnedGraph`. This avoids an intermediate file-sized buffer.
Each execution context owns independent activations and scratch space; immutable model
bytes and prepared parameters can be shared. `input_mut` and `output` access tensor
storage directly; adapters need not allocate staging tensors.

Inputs must be written before every inference: memory planning can reuse their storage.
After a failed `compute`, the outputs are not valid.

### Allocators

Any type implementing `allocator_api2::alloc::Allocator` (re-exported as
`nanoglia::Allocator`) works, and so does a shared reference to one:

- `Global`, the Rust global allocator, with feature `alloc`
  (`Graph::load`, `init_execution_context` and
  `MemoryRequirements::compute` use it);
- your own allocator, for example over a memory pool of an embedded OS;
- `Arena`, a bump allocator over a buffer you own, usually a `static`.
  Pass `&arena` to every call.

`Arena::new(&mut buffer)` (or `Arena::new_uninit` for a
`[MaybeUninit<u8>]`) hands out blocks from the start of the buffer.
Freeing the most recent block gives its memory back (stack order);
other frees are ignored until `Arena::reset`, which needs `&mut` access,
so nothing allocated from the arena can still be alive.
`Arena::used()` reports the bytes handed out so far, alignment padding
included, and `Arena::capacity()` the buffer size.

### Alignment

Every block nanoglia allocates is aligned on `nanoglia::core::ALIGN` = 16
bytes, enough for every scalar type and for the vector loads of CMSIS-NN
(Helium). Every activation tensor and the scratch area inside a context
start on a 16-byte boundary too. `graph_bytes` and `context_bytes` are
multiples of 16.

With an `Arena`, the first block may lose up to 15 bytes to reach a
16-byte boundary. `MemoryRequirements::arena_bytes(n)` counts them:

```text
arena_bytes(n) = 15 + graph_bytes + n × context_bytes
```

so a buffer of that size is enough whatever its own alignment.

### Model bytes

The graph borrows the model (`Graph<'m, A>`): weights and other constant
tensors are read where they are, in flash or RAM, and never copied. The
model bytes must outlive the graph, and each execution context borrows
its graph. A graph loaded with `Graph::load_with` also borrows the
backends of its `Registry` for the same lifetime `'m` (the default
registry is `'static`).

Constant buffers are read as typed arrays, so each must be aligned for
its element type (4 bytes for int32 biases, 8 for int64 ones); loading
fails with `Detail::Misaligned` otherwise. Their alignment in memory
depends on where the first byte of the model is: place it on a 16-byte
boundary, as the tests and examples do:

```rust
#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);

static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!("model.tflite"));
// Pass `&MODEL.0` to Graph::load_in.
```

A `Vec<u8>` filled from a file has no such guarantee.

### Inputs are rewritten before each compute

The memory plan places activations whose lifetimes do not overlap at the
same offsets. Once an operator has consumed a model input, later
operators may write their outputs over it. Write every input
(`set_input` or `input_mut`) before each `compute`, even when it has not
changed. Outputs stay valid until the next call that borrows the context
mutably (`input_mut`, `set_input`, `compute`).

### Several contexts

A graph is immutable after loading. Create one execution context per
inference that may run at the same time: each has its own
activations and scratch memory, and they share the graph block.

## What the blocks hold

The graph block, in this order (`crates/nanoglia/src/graph.rs`):

1. the operator table: for each operator, its kernel (a `&dyn Kernel`
   borrowed from its backend, two pointers), the place of its parameters
   and of its inputs and outputs: 32 bytes per operator on a 64-bit
   target, 24 on a 32-bit one;
2. the location of each tensor: constant (offset in the model bytes),
   activation (offset in the context) or unused;
3. the tensor indices of every operator's inputs and outputs;
4. the memory planner's working memory, 20 bytes per tensor: a lifetime
   interval (16 bytes) and an ordering slot (4 bytes);
5. the prepared parameters of each operator (multipliers, shifts,
   shapes, lookup tables), each rounded up to 16 bytes.

The size of items 1 to 3 depends on the pointer width, so a 32-bit
target needs a little less than a 64-bit host.

The context block holds the activation tensors, placed by a greedy
planner (largest tensors first, each at the lowest offset that does not
overlap a tensor alive at the same time, the strategy of TensorFlow Lite
Micro's `GreedyMemoryPlanner`), then the scratch area shared by all
operators: the largest scratch any kernel asks for.

`ModelReport` reports these sizes for a model without loading it, as a
`MemoryBreakdown`
([Check whether a model runs](../how-to-guides/check-whether-a-model-runs.md)):

| Field | Contents |
| --- | --- |
| `graph_bytes` | The graph block, as `Graph::graph_bytes()`. |
| `persistent_bytes` | Item 5 of the graph block: the prepared parameters. For keyword spotting, 5984 of the 7712 bytes on a 64-bit host. |
| `context_bytes` | The context block, as `Graph::context_bytes()`: `activation_bytes + scratch_bytes`. |
| `activation_bytes` | The activation tensors as the planner packs them, rounded up to 16 bytes (16000 for keyword spotting). |
| `scratch_bytes` | The scratch area, rounded up to 16 bytes (160 for keyword spotting, with CMSIS-NN's portable C code). |

`requirements()` gives the `MemoryRequirements` (`graph_bytes` and
`context_bytes`). Each operator's `SelectedKernel::plan` holds what its
kernel asks for: `persistent` (rounded up to 16 bytes in the graph
block) and `scratch`. (Numbers: `cargo run -p nanoglia-tools -- check
testdata/mlperf-tiny/kws_ref_model.tflite` on a Linux aarch64 host,
2026-11-10, with the change that added `ModelReport`, on top of commit
`136ee56`.)

## MLPerf Tiny footprints

The four int8 reference models of `testdata/mlperf-tiny/`. Graph and
context sizes measured with `Graph::graph_bytes()` and
`context_bytes()` on 2026-11-04 at commit `28c6f1a`, default features:

- 64-bit: Linux aarch64 host, printed by the test
  `requirements_match_the_loaded_graph` of
  `crates/nanoglia/tests/memory.rs`
  (`cargo test -p nanoglia --test memory -- --nocapture`);
- 32-bit: the same test built for `wasm32-wasip1` and run under
  wasmtime 50.0.0 (`cargo test -p nanoglia --test memory --target
  wasm32-wasip1 --release -- --nocapture`, with the environment of
  `mise run test-wasip1`).

| Model | File | Operators | Graph, 64-bit | Context, 64-bit | `arena_bytes(1)`, 64-bit | Graph, 32-bit | Context, 32-bit | `arena_bytes(1)`, 32-bit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kws_ref_model` (keyword spotting, DS-CNN) | 53936 B | 13 | 7712 B | 16160 B | 23887 B | 7616 B | 16160 B | 23791 B |
| `vww_96_int8` (visual wake words, MobileNetV1) | 333288 B | 31 | 29520 B | 64624 B | 94159 B | 29280 B | 64624 B | 93919 B |
| `pretrainedResnet_quant` (image classification, ResNet-8) | 98496 B | 16 | 6192 B | 51456 B | 57663 B | 6064 B | 51456 B | 57535 B |
| `ad01_int8` (anomaly detection, FC autoencoder) | 276976 B | 10 | 16288 B | 768 B | 17071 B | 16208 B | 768 B | 16991 B |

The 32-bit `arena_bytes(1)` column is computed with the formula above.
With an `Arena` whose buffer starts on a 16-byte boundary, `Arena::used()`
after loading and creating one context is `graph_bytes + context_bytes`
(23776 B for keyword spotting on 32-bit).

With feature `esp-nn`, each operator ESP-NN runs keeps the parameters of
both ESP-NN and its CMSIS-NN fallback in the graph block, and the scratch
area fits the larger of the two. On the same host, with default features
plus `esp-nn` (ESP-NN's ANSI C kernels):

| Model | Graph | Context | `arena_bytes(1)` |
| --- | ---: | ---: | ---: |
| `kws_ref_model` | 13664 B | 16160 B | 29839 B |
| `vww_96_int8` | 55056 B | 64624 B | 119695 B |
| `pretrainedResnet_quant` | 10464 B | 51456 B | 61935 B |
| `ad01_int8` | 16928 B | 768 B | 17711 B |

(`cargo test -p nanoglia --features esp-nn --test memory -- --nocapture`.)
On the ESP32-S3, ESP-NN's optimized kernels may ask for other scratch
sizes.

These sizes come from the portable C code of CMSIS-NN. On Arm targets
with the DSP or Helium extension, CMSIS-NN may ask for different scratch
sizes, so `context_bytes` can differ: compute the requirements for your
target ([Size the memory](../how-to-guides/size-memory.md)), or read
them on the device (the Nano 33 BLE example prints them).

## Code size

What nanoglia adds to the flash depends on the Cargo features
([Cargo features](features.md)): the backends, the float32 and int16
kernels (`float`, `int16`), and the operators (`all-ops`, or the `op-*`
features of your models,
[Reduce the code size](../how-to-guides/reduce-code-size.md)).

The firmware of `examples/nano33ble` (`thumbv7em-none-eabihf`,
`target-cpu=cortex-m4`, CMSIS-NN with its DSP code paths, `opt-level =
"s"`, fat LTO, one codegen unit, `defmt` over RTT), which loads one
model from flash and runs it. `text` of `arm-none-eabi-size`, in bytes:
code and constants, **the model included**; `data` is 56 bytes and `bss`
33808 bytes (the 32 KiB arena) in every row.

| `nanoglia` features, besides `cmsis-nn`, `kernels`, `defmt` | Keyword spotting (model: 53936 B) | Gesture classifier (model: 5048 B) |
| --- | ---: | ---: |
| `all-ops`, `float`, `int16` (the default operators and types) | 313688 | 264808 |
| `all-ops` (int8 only, every operator) | 274120 | 225240 |
| the operators of the model, `float`, `int16` | 169332 | 116268 |
| the operators of the model (int8 only) | 138240 | 91688 |

- The operators of the keyword-spotting model
  (`testdata/mlperf-tiny/kws_ref_model.tflite`): `op-conv-2d`,
  `op-depthwise-conv-2d`, `op-average-pool-2d`, `op-reshape`,
  `op-fully-connected`, `op-softmax`. This last row is the example as
  committed.
- The operators of the gesture classifier
  (`examples/train-to-device/reference/model.tflite`, built into the same
  firmware in place of the keyword-spotting model): `op-conv-2d`,
  `op-max-pool-2d`, `op-mean`, `op-fully-connected`, `op-softmax`.
- Listing the operators of the model saves 135880 bytes of an int8-only
  keyword-spotting firmware (274120 to 138240) and 133552 bytes of the
  gesture one. Without the model, 84304 bytes remain for keyword
  spotting: the firmware's own code (startup, `defmt`), the model
  verifier, the memory planner, and the kernels of six operators.
- The C kernels go too: the int8-only keyword-spotting firmware links 53
  CMSIS-NN functions with every operator and 38 with the operators of the
  model (`arm-none-eabi-nm` on the two files, symbols `arm_*`). Those
  that leave are the kernels of the other operators:
  `arm_elementwise_add_s8`, `arm_elementwise_mul_s8`, `arm_maximum_s8`,
  `arm_minimum_s8`, `arm_max_pool_s8`, `arm_pad_s8`,
  `arm_transpose_conv_s8` and their helpers.
- Before the `op-*` features (commit `6825380`, release 0.2.0) the same
  firmware measured 313872 bytes with `float` and `int16` and 274920
  bytes without, for keyword spotting; 264992 and 226040 bytes for the
  gesture classifier.

Measured on 2026-11-10 at commit `e607b15`, on a Linux aarch64 host with
rustc 1.99.1 and `arm-none-eabi-gcc` 14.2.1: for each cell, the
`features` of `nanoglia` in `examples/nano33ble/Cargo.toml` set to the
row's, the model of `src/main.rs` to the column's, then
`cargo build --release` in `examples/nano33ble` and `arm-none-eabi-size
target/thumbv7em-none-eabihf/release/nanoglia-example-nano33ble`
(`mise run example-nano33ble` does both for the committed example).
These are sizes of a linked firmware; it has not run on the board
([issue #2](https://github.com/AlixANNERAUD/nanoglia/issues/2)). The
sizes of a firmware compared with other engines are in
[Performance](performance.md#keyword-spotting-firmware).

## The Nano 33 BLE test bench

The firmware that runs the golden cases on an Arduino Nano 33 BLE
(`hardware/nano33ble`,
[Run the golden cases on a board](../how-to-guides/run-the-golden-cases-on-a-board.md))
links every kernel family (`cmsis-nn`, `kernels`, `float`, `int16`,
`all-ops`) and a USB stack, at `opt-level = 3` with fat LTO, and keeps
the MLPerf Tiny models in flash, where they run in place. The five
models (837216 bytes) and the code do not fit together in the 983040
bytes of flash after the bootloader: two images are built. `text` of
`arm-none-eabi-size`, in bytes, models included; `data` is 0 and `bss`
332 in each:

| Image | Models in flash | `text` | Left of 983040 |
| --- | --- | ---: | ---: |
| none | — | 433604 | 549436 |
| 1 | `kws_ref_model`, `pretrainedResnet_quant`, `ad01_int8` | 863132 | 119908 |
| 2 | `vww_96_int8`, `str_ww_ref_model` | 841524 | 141516 |

With the model in flash, a case needs RAM for its inputs and for the
arena (the graph and context blocks) only. Of the 262144 bytes of RAM,
the firmware keeps the last 256 for the message of a crash, 24576
below them for the stack, and 160 for its static data; the rest, 237152
bytes, receives the inputs and holds the arena. What each model takes
of it, measured on the same code under emulation (`hardware/emulated`:
the same memory layout, without static data):

| Model | Inputs | Graph | Context | Arena used | Peak stack | Inputs + arena, of 237152 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `kws_ref_model` | 490 | 7616 | 16256 | 23872 | 8208 | 24362 |
| `pretrainedResnet_quant` | 3072 | 6064 | 51456 | 57520 | 8208 | 60592 |
| `str_ww_ref_model` | 1200 | 9248 | 10496 | 19744 | 7936 | 20944 |
| `ad01_int8` | 640 | 16208 | 768 | 16976 | 7936 | 17616 |
| `vww_96_int8` | 27648 | 29280 | 65536 | 94816 | 8208 | 122464 |

All five run: the largest, visual wake words, takes about half of what
is free. The contexts differ from the 32-bit column of the
[footprints](#mlperf-tiny-footprints) where CMSIS-NN's DSP kernels ask
for other scratch sizes than its portable C code (keyword spotting,
visual wake words). The single-operator cases of `testdata/ops/` are
received into the same RAM with their model: the largest model is 7648
bytes, the largest arena 4352 bytes, and the deepest stack over every
golden case is 8444 bytes of the 24576.

Measured on 2026-11-10, with the change that added `hardware/`, on top
of commit `d98b750`, on a Linux aarch64 host with rustc 1.99.1,
`arm-none-eabi-gcc` 14.2.1 and Unicorn 2.1.4: the sizes with `mise run
build-hardware-nano33ble` (and `cargo build --release --target
thumbv7em-none-eabihf -p nanoglia-hardware-nano33ble` for the image
without model); the RAM with `mise run test-hardware`, from
`hardware/target/results/emulated.json` (fields `inputs_bytes`,
`graph_bytes`, `context_bytes`, `arena_used`, `stack_peak`). The stack
is measured by painting it. These are figures of a linked firmware and
of an emulated run: the firmware has not run on the board
([issue #2](https://github.com/AlixANNERAUD/nanoglia/issues/2)).
