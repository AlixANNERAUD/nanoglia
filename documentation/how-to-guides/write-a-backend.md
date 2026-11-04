# Write a backend

This guide adds a kernel backend of your own in front of the bundled
ones: a backend that owns state its kernels use, here the driver of a
DMA controller that runs the model's `RESHAPE` operators. Then it times
each operator with an observer, to see what the change bought.

The same steps apply to any driver: an accelerator's register block, a
pool of buffers, a hardware lock. How the pieces fit is explained in
[Architecture](../explanation/architecture.md#backends-and-the-registry).

## 1. Add the dependencies

The bundled backends are in their own crates. To list them after yours,
depend on them from the same source as `nanoglia`, so that all share one
`nanoglia-core`:

```toml
[dependencies]
nanoglia = { git = "https://github.com/AlixANNERAUD/nanoglia" }
nanoglia-cmsis-nn = { git = "https://github.com/AlixANNERAUD/nanoglia" }
nanoglia-kernels = { git = "https://github.com/AlixANNERAUD/nanoglia" }
```

## 2. Write the kernel

A kernel implements `nanoglia::core::backend::Kernel`. While the model
loads, `plan` says how many bytes of parameters and scratch memory an
operator needs and `prepare` writes the parameters; at each inference,
`eval` computes the outputs from the inputs and the parameters, without
allocating and without reading the model.

The kernel holds a reference to the driver. Execution contexts may run
the same graph at the same time, so a kernel is `Sync` and `eval` takes
`&self`: here an atomic flag guards the controller's single channel, and
the kernel copies with the CPU while another context uses it.

```rust
use core::sync::atomic::{AtomicBool, Ordering};
use nanoglia::core::backend::{Backend, EvalCtx, Kernel, KernelPlan};
use nanoglia::core::{BuiltinOp, OpDesc};
use nanoglia::{Detail, Error, Result};

/// The driver of a DMA controller, owned by your application.
pub struct Dma {
    // Registers, channel number...
}

impl Dma {
    /// Copies `src` into `dst` (same length) and waits until the transfer ends.
    pub fn copy(&self, src: &[u8], dst: &mut [u8]) {
        dst.copy_from_slice(src); // Stand-in for the hardware transfer.
    }
}

/// The number of bytes a `RESHAPE` copies, if the DMA kernel can run `op`.
fn reshape_bytes(op: &OpDesc<'_, '_>) -> Option<u32> {
    let (input, output) = (op.input(0)?, op.output(0)?);
    if op.builtin()? != BuiltinOp::RESHAPE || op.outputs.len() != 1 || input.dtype != output.dtype
    {
        return None;
    }
    let bytes = input.byte_len()?;
    if output.byte_len()? != bytes {
        return None;
    }
    u32::try_from(bytes).ok()
}

/// `RESHAPE` with the DMA controller. Several execution contexts may run at the same time, and
/// the controller has one channel: the kernel copies with the CPU while the channel is busy.
struct DmaReshape<'d> {
    dma: &'d Dma,
    busy: AtomicBool,
}

impl Kernel for DmaReshape<'_> {
    fn name(&self) -> &'static str {
        "dma/reshape"
    }

    fn plan(&self, op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
        reshape_bytes(op).ok_or_else(|| unsupported(op))?;
        // Persistent parameters: the byte count, as a little-endian u32.
        Ok(KernelPlan {
            persistent: 4,
            scratch: 0,
        })
    }

    fn prepare(&self, op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
        let bytes = reshape_bytes(op).ok_or_else(|| unsupported(op))?;
        persist.copy_from_slice(&bytes.to_le_bytes());
        Ok(())
    }

    fn eval(&self, ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
        let bytes = ctx
            .persist
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| Error::invalid_argument(Detail::None))?;
        let input = ctx.inputs.first().copied().unwrap_or_default();
        let output = ctx.output(0);
        if input.len() != bytes as usize || output.len() != bytes as usize {
            return Err(Error::invalid_argument(Detail::SizeMismatch {
                expected: bytes,
                actual: output.len() as u32,
            }));
        }
        if self
            .busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            self.dma.copy(input, output);
            self.busy.store(false, Ordering::Release);
        } else {
            output.copy_from_slice(input);
        }
        Ok(())
    }
}

fn unsupported(op: &OpDesc<'_, '_>) -> Error {
    Error::unsupported(Detail::UnsupportedOp {
        op_index: op.index,
        opcode: op.opcode_number(),
    })
}
```

A kernel must compute exactly what TensorFlow Lite computes: a copy is
trivially exact; for arithmetic, port the TFLite reference or TFLite
Micro code ([Bit-exactness](../explanation/bit-exactness.md)). A
stateless kernel needs no type of its own: a `static` `FnKernel`, three
plain functions, is enough.

## 3. Write the backend and register it

The backend owns its kernels and lends them out: `select` returns a
kernel borrowed from `self`, only for the operators the kernel fully
supports, and `None` for the others, which go to the next backend.
Loading calls `select` twice per operator, so it must answer the same
way each time.

```rust
/// The DMA backend: it owns its kernel, which borrows the driver.
pub struct DmaBackend<'d> {
    reshape: DmaReshape<'d>,
}

impl<'d> DmaBackend<'d> {
    pub fn new(dma: &'d Dma) -> Self {
        Self {
            reshape: DmaReshape {
                dma,
                busy: AtomicBool::new(false),
            },
        }
    }
}

impl Backend for DmaBackend<'_> {
    fn name(&self) -> &'static str {
        "dma"
    }

    fn select<'s>(&'s self, op: &OpDesc<'_, '_>) -> Option<&'s dyn Kernel> {
        reshape_bytes(op)?;
        Some(&self.reshape)
    }
}
```

Put it first in a `Registry`, followed by the bundled backends, and
load the model with `Graph::load_with`:

```rust
use nanoglia::{Encoding, Graph, Registry, Target};

let dma = Dma {};
let dma_backend = DmaBackend::new(&dma);
let backends: [&dyn Backend; 3] = [
    &dma_backend,
    &nanoglia_cmsis_nn::BACKEND,
    &nanoglia_kernels::BACKEND,
];
let registry = Registry::new(&backends);
let graph = Graph::load_with(&registry, &[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
for i in 0..graph.operator_count() {
    println!("operator {i}: {}", graph.kernel_name(i).unwrap_or("?"));
}
```

The graph borrows the backends, so `dma`, `dma_backend` and `backends`
must outlive it; they can live on the stack of the function that runs
the model. The compiler rejects a graph that would outlive them.

With the MLPerf Tiny keyword spotting model, operator 10 is now the DMA
kernel:

```text
operator 9: cmsis-nn/average_pool2d_s8
operator 10: dma/reshape
operator 11: cmsis-nn/fully_connected_s8
```

## 4. Time each operator

`ExecutionContext::compute_with` runs the graph like `compute`, and
calls an `Observer` before and after each operator. This one adds the
cycles of each operator to a table, read from a clock you pass (on a
Cortex-M, the DWT cycle counter):

```rust
use core::ops::ControlFlow;
use nanoglia::{Observer, OpInfo};

/// Cycles spent in each operator, summed over the inferences.
struct Profiler<F: FnMut() -> u32> {
    clock: F,
    start: u32,
    cycles: [u32; 64],
}

impl<F: FnMut() -> u32> Observer for Profiler<F> {
    fn before(&mut self, _op: &OpInfo<'_>) -> ControlFlow<()> {
        self.start = (self.clock)();
        ControlFlow::Continue(())
    }

    fn after(&mut self, op: &OpInfo<'_>, _result: &Result<()>) {
        let elapsed = (self.clock)().wrapping_sub(self.start);
        if let Some(total) = self.cycles.get_mut(op.index()) {
            *total = total.wrapping_add(elapsed);
        }
    }
}
```

```rust
let mut ctx = graph.init_execution_context_in(&arena)?;
let mut profiler = Profiler { clock, start: 0, cycles: [0; 64] };
ctx.input_mut(0)?.as_mut_slice::<i8>()?.fill(3);
ctx.compute_with(&mut profiler)?;
for (i, cycles) in profiler.cycles.iter().enumerate().take(graph.operator_count()) {
    println!("{:>28}: {cycles}", graph.kernel_name(i).unwrap_or("?"));
}
```

The `OpInfo` an observer receives also gives the operator's opcode and
read-only views of its inputs and outputs (`op.input(i)`,
`op.output(i)`), built only when called: in `after`, `op.output(0)` is
the value the kernel just computed, which is how to compare an
intermediate tensor with a reference. `compute` is `compute_with(&mut
())`, so a firmware that never observes pays nothing.

## Check it

- `graph.kernel_name(i)` names your kernel for the operators you
  expected.
- The outputs are those of the default backends: load the model a
  second time with `Graph::load_in` and compare.
- If `load_with` fails with `UnsupportedOperation` /
  `UnsupportedOp`, no backend claims that operator: your `select`
  refused it and the bundled backends you listed do not support it
  ([Errors](../reference/errors.md)).

The code of this guide was run on a Linux aarch64 computer with the
keyword spotting model (on 2026-11-04), the DMA transfer replaced by a
CPU copy and the clock by `std::time::Instant`; the outputs matched
those of the default backends.
