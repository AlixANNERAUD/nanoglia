# Architecture

nanoglia loads a TensorFlow Lite model at run time, picks a kernel for
each operator among the libraries available on the chip, plans the
memory, and runs inferences without allocating. This page explains how
the pieces fit and why they are built this way.

## Crates

| Crate | Role | Depends on |
| --- | --- | --- |
| `nanoglia` | The API your application uses: `Graph`, `ExecutionContext`, `Registry`, `MemoryRequirements`, `ModelReport`, tensor views, `Observer`. Loads, plans, computes. | all the others |
| `nanoglia-core` | The shared vocabulary: tensors (`TensorDesc`, `DataType`, `Dims`, `Quantization`), operators (`OpDesc`, `BuiltinOp`, `OpOptions`), errors, the `Backend` / `Kernel` interface, TFLite-exact quantization math (`quant`), `Arena` and `Block`. | `allocator-api2` |
| `nanoglia-tflite` | Verifies a `.tflite` flatbuffer once, then reads it in place: zero-copy, allocation-free, panic-free, `#![forbid(unsafe_code)]`, its own small flatbuffer reader. | `nanoglia-core` |
| `nanoglia-cmsis-nn-sys` | CMSIS-NN v8.0.0 (integer kernels; a git submodule in the repository, its sources shipped in the package) compiled with `cc`, hand-written FFI bindings. | — (build: `cc`) |
| `nanoglia-cmsis-nn` | The `cmsis-nn` backend: the operators CMSIS-NN computes exactly like TFLite. | `nanoglia-core`, `nanoglia-cmsis-nn-sys` |
| `nanoglia-esp-nn-sys` | ESP-NN v1.4.2 (a git submodule in the repository, its sources shipped in the package) compiled with `cc` (ANSI C kernels everywhere, ESP32-S3 optimized kernels on Xtensa), hand-written FFI bindings and the lock of its global scratch buffers. | — (build: `cc`) |
| `nanoglia-esp-nn` | The `esp-nn` backend (feature `esp-nn`): the int8 operators ESP-NN computes exactly like TFLite, with CMSIS-NN as fallback. | `nanoglia-core`, `nanoglia-cmsis-nn`, `nanoglia-esp-nn-sys` |
| `nanoglia-kernels` | The `rust` backend: what CMSIS-NN lacks (shape and quantization operators, broadcasting, activations, int16 softmax, fully-connected, logistic and tanh) and the f32 operators. | `nanoglia-core`, `libm` |
| `nanoglia-tools` | The `nanoglia check` command for your computer ([Check whether a model runs](../how-to-guides/check-whether-a-model-runs.md)): prints what `ModelReport` finds. The only crate that uses `std`; not published. | `nanoglia` and the three backends |

Everything else is `#![no_std]`, with no dependency on a HAL, an RTOS or a
vendor SDK. The only C code is CMSIS-NN and ESP-NN (compiled without
ESP-IDF), which need the C headers of the target but link no C
library.

## The API, after wasi-nn

The public API follows [wasi-nn](https://github.com/WebAssembly/wasi-nn),
the WebAssembly interface for machine-learning inference:

| wasi-nn | nanoglia |
| --- | --- |
| `load(builders, encoding, target)` | `Graph::load_in(builders, Encoding, Target, alloc)` |
| `graph-encoding` | `Encoding::{TensorFlowLite, Autodetect}` |
| `execution-target` | `Target::{Cpu, Npu, Auto}` |
| `init-execution-context` | `Graph::init_execution_context_in(alloc)` |
| `set-input`, `compute`, `get-output` | `ExecutionContext::set_input` / `input_mut`, `compute`, `output` |
| `error-code` | `ErrorCode` |

The same vocabulary lets a wasi-nn host for WebAssembly modules run on
nanoglia and forward its errors unchanged. What wasi-nn leaves out,
nanoglia adds as extra parameters: the allocator of each call, and the
`Registry` of backends.

## Loading a model

`Graph::load_with(registry, builders, encoding, target, alloc)` runs
these steps (`crates/nanoglia/src/graph.rs`):

1. **Open and verify.** The encoding is checked (`TFL3` identifier), and
   `nanoglia_tflite::Model::parse` verifies the whole flatbuffer once:
   offsets, vtables, vectors, strings and indices. After this, every
   read stays in bounds, so a corrupted or hostile model gives an
   `Error`, never a panic or an out-of-bounds read. Only one subgraph is
   supported.
2. **Select kernels and size the graph.** For each operator, nanoglia
   builds an `OpDesc` on the stack (its code, options and the
   descriptions of its tensors, borrowed from the model) and asks the
   backends of the registry in order; the first `select` that returns a
   kernel wins. Loading runs this selection twice (here, then again in
   step 6), so a backend must answer the same way both times. The kernel's `plan` reports how many bytes of persistent
   parameters and of scratch memory it needs. The sum of the
   parameters, the tables and the planner's working memory gives the
   size of the graph block; the largest scratch gives the size of the
   scratch area.
3. **Allocate the graph block**: the only allocation of the graph.
4. **Plan the activations.** nanoglia computes, for each activation
   tensor, the first and last operator that uses it, checking the
   dataflow on the way (each tensor produced once, before it is
   consumed, never a constant). The greedy planner then places the
   tensors in the context arena, largest first, each at the lowest
   offset that does not overlap a tensor alive at the same time. This is
   the strategy of TensorFlow Lite Micro's `GreedyMemoryPlanner`. The
   planner works in the graph block, so it allocates nothing.
5. **Locate the tensors.** Constants are recorded as offsets in the
   model bytes (and checked to be aligned for their type), activations as
   offsets in the context arena.
6. **Prepare the kernels.** Each kernel's `prepare` writes its
   parameters into its region of the graph block: requantization
   multipliers and shifts, paddings, activation ranges, lookup tables,
   precomputed sums. After this, kernels never read the model again.

`MemoryRequirements::compute_with` runs steps 1, 2 and 4 with only the
planner's working memory, and returns the two sizes without loading
anything ([Size the memory](../how-to-guides/size-memory.md)).

## Running an inference

`ExecutionContext` owns one block: the activations and the scratch area.
`compute` walks the operator table and, for each operator, builds the
input and output byte slices from the recorded locations (constants in
the model, activations in the arena), then calls the kernel's `eval`
with them, its prepared parameters and the scratch area. `eval` reads
only these and never allocates. The operator table holds one
`&dyn Kernel` per operator: one indirect call per operator, nothing
allocated.

`compute_with(observer)` runs the same loop and calls an `Observer`
before and after each operator, with an `OpInfo`: the operator index,
its kernel name and opcode, and views of its inputs and outputs, built
only when the observer asks. `compute` is `compute_with(&mut ())`: the
observer methods of `()` are empty and inlined, so nothing remains of
them.

## Backends and the registry

A backend implements `nanoglia::core::backend::Backend`: a name, a kind
(`Cpu` or `Npu`) and `select`, which returns a kernel only if it fully
supports the operator with its types, shapes and options. A backend
never claims an operator it would compute differently from TFLite; it
leaves it to the next one.

A kernel implements `nanoglia::core::backend::Kernel`: `plan`, `prepare`
and `eval`, plus a name for diagnostics. `select` returns a
`&dyn Kernel` borrowed from the backend, and the graph keeps it in its
operator table, so the graph borrows the backend. This is what lets a
backend hold state its kernels use (a driver handle for an accelerator,
a pool of DMA buffers, counters) without globals: the kernel reaches it
through `&self`. The borrow uses the lifetime that already ties the
graph to the model bytes, so `Graph` has no extra lifetime parameter,
and the default registry, `'static`, works with any graph. Kernels are
`Sync`, because several execution contexts may run the same graph at
once. The bundled backends are stateless: their kernels are
`FnKernel`s, three plain functions (`plan`, `prepare`, `eval`) in a
`static`. [Write a backend](../how-to-guides/write-a-backend.md) shows a
stateful one.

`Registry::default()` holds the backends enabled by the Cargo features,
fastest first: ESP-NN (feature `esp-nn`, off by default), CMSIS-NN, then
the Rust kernels. `Registry::new(&[...])`
builds another list, for example to put your own backend first or to
test one backend alone, and `Graph::load_with` takes it.

CMSIS-NN computes each operator in one call into C, after its
parameters were computed at prepare time from TensorFlow Lite Micro's
code ported to Rust. On Cortex-M4, M7 and M33 it uses the DSP
extension, on Armv8.1-M Helium, and on every other target its portable
C code. ESP-NN, on the ESP32-S3, uses the chip's SIMD instructions; its
parameters are computed from the code of esp-tflite-micro, Espressif's
port of TensorFlow Lite Micro. The Rust backend ports TFLite's reference
kernels and their preparation code.

ESP-NN keeps its scratch buffers in global variables, so only one
context may run it at a time. The `esp-nn` backend takes a lock (an
atomic flag that never blocks) around each call. The lock is a global
too, not state of the backend: the C globals it guards exist once per
program, whatever the number of backend values. When another thread,
task, interrupt handler or core holds it, the operator runs the CMSIS-NN
kernel instead, which computes the same result. For this, the backend
claims an operator only if CMSIS-NN supports it too, and prepares both
kernels: their parameters both live in the graph block, and the scratch
area fits the larger of the two.

## Zero copy

Weights are the largest part of a model (the MLPerf Tiny model files
weigh 54 KB to 333 KB), often more than the RAM of a microcontroller.
nanoglia never copies them. The TFLite flatbuffer is designed to be read in place,
so the model can stay in flash (memory-mapped) or wherever your
application loaded it, and kernels read constant tensors straight from
it. Tensor descriptions are views into the model bytes too, read byte by
byte so that metadata needs no alignment. Only the constant buffers
that kernels read as typed arrays must be aligned
([Memory](../reference/memory.md#model-bytes)).

## Accelerators and delegates

`Target::Npu`, `Target::Cpu` and `BackendKind::Npu` are in the API for
neural accelerators, which would run as backends of kind `Npu` claiming
whole operators or delegate operators. No such backend exists yet:
`Target::Npu` currently fails with `ErrorCode::UnsupportedOperation`.
ESP-NN's ESP32-P4 kernels are not built yet: they need Espressif's
RISC-V compiler.
