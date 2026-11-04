# Size the memory

Firmware without a heap gives nanoglia a static buffer through an
`Arena`. This guide finds the exact size that buffer needs for your
model, checks it in a test, and confirms it on the device.

`MemoryRequirements` computes it without loading the model: it verifies
the model, selects the kernels and plans the memory, and returns two
numbers:

- `graph_bytes`: the graph block (kernel table, prepared parameters,
  memory plan);
- `context_bytes`: each execution context (activations and scratch
  memory).

`arena_bytes(n)` turns them into the size of an `Arena` buffer for the
graph and `n` contexts, whatever the alignment of the buffer
([Memory](../reference/memory.md#alignment)).

## 1. Compute the size on the host

Add a test to a crate that builds for your computer and can read the
model file, with the same nanoglia features as your firmware (they
decide which kernels are chosen, and kernels differ in the memory they
need) plus `alloc`:

```toml
[dev-dependencies]
nanoglia = { git = "https://github.com/AlixANNERAUD/nanoglia", default-features = false, features = ["cmsis-nn", "kernels", "alloc"] }
```

```rust
use nanoglia::{Encoding, MemoryRequirements, Target};

#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);

static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!("../kws_ref_model.tflite"));

/// The size of the firmware's `MEMORY` buffer.
const MEMORY_BYTES: usize = 24 * 1024;

#[test]
fn memory_fits() {
    let needs =
        MemoryRequirements::compute(&[&MODEL.0], Encoding::Autodetect, Target::Auto).unwrap();
    println!("{needs:?}, arena_bytes(1) = {}", needs.arena_bytes(1));
    assert!(needs.arena_bytes(1) <= MEMORY_BYTES);
}
```

Run it and read the numbers:

```sh
cargo test --release memory_fits -- --nocapture
```

```text
MemoryRequirements { graph_bytes: 7712, context_bytes: 16160 }, arena_bytes(1) = 23887
```

(These are the numbers of the MLPerf Tiny keyword spotting model on a
64-bit computer.)

Without the `alloc` feature, call `MemoryRequirements::compute_in` with
any allocator: it takes only the planner's working memory from it, 20
bytes per tensor, and frees it before returning. With a custom
`Registry`, use `compute_with`.

## 2. Size the static buffer

Reserve at least `arena_bytes(1)` bytes in the firmware, and keep the
same constant in the test above so that a model change that needs more
memory fails the test:

```rust
static mut MEMORY: [u8; 24 * 1024] = [0; 24 * 1024];
```

For several execution contexts on the same graph (for example two
inferences that may run at the same time), use `arena_bytes(n)`.

To put the graph and the contexts in different memories, for example
the graph in ordinary RAM and the activations in faster tightly-coupled
memory, use two arenas: give the graph's arena `graph_bytes + 15` bytes
and the context's `context_bytes + 15`.

## 3. Account for the target

The host numbers differ from the device's in two ways:

- The graph block holds pointers: a 32-bit microcontroller needs a little
  less than a 64-bit computer (7616 bytes instead of 7712 for keyword
  spotting), so the host number is a safe upper bound for this part.
- On the host, CMSIS-NN compiles its portable C code. On Arm targets
  with the DSP or Helium extension it may ask for a different scratch
  size, so `context_bytes` can differ, in either direction.

Check the numbers on the device once, at start-up, before loading:

```rust
let arena = Arena::new(unsafe { &mut *core::ptr::addr_of_mut!(MEMORY) });
let needs = MemoryRequirements::compute_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
defmt::info!("graph {} B, context {} B, arena {} B", needs.graph_bytes, needs.context_bytes, needs.arena_bytes(1));
let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
```

`compute_in` frees its working memory before returning, and an `Arena`
gives back the most recent block when it is freed, so the graph is
loaded at the start of the buffer again and `arena_bytes(1)` bytes
still suffice. After loading,
`graph.memory_requirements()` returns the same two numbers.

## If the buffer is too small

`Graph::load_in` or `init_execution_context_in` fails with
`ErrorCode::TooLarge` and `Detail::OutOfMemory { requested }`, where
`requested` is the size of the block that did not fit
([Errors](../reference/errors.md)). `MemoryRequirements` checks the model
like `Graph::load_in` does (verification, kernel selection, alignment of the
constant buffers) but does not run the kernels' `prepare`, so a kernel that
rejects its parameters is only reported by `Graph::load_in`.

## See also

- [Memory](../reference/memory.md): what the blocks hold, and the
  footprints of the MLPerf Tiny models.
- [Run on a microcontroller](run-on-a-microcontroller.md).
