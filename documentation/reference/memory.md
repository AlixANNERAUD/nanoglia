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
block) and `scratch`. (Numbers: `cargo run -p nanoglia-cli -- check
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
