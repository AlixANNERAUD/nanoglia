use crate::graph::{Graph, GraphTensors, Location};
use crate::{TensorMut, TensorRef};
use nanoglia_core::backend::{EvalCtx, Kernel};
use nanoglia_core::{
    Allocator, Block, Detail, Error, MAX_OP_INPUTS, MAX_OP_OUTPUTS, OpCode, Result,
};

/// The mutable state of one inference: the arena holding the activations and the scratch memory.
///
/// Write the inputs ([`Self::set_input`] or [`Self::input_mut`]) before every
/// [`Self::compute`]: the memory plan reuses input memory once it has been consumed. Outputs
/// stay valid until the next call that borrows the context mutably.
///
/// ```
/// # #[repr(C, align(16))]
/// # struct Aligned<T: ?Sized>(T);
/// # static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
/// #     env!("CARGO_MANIFEST_DIR"),
/// #     "/doc/classifier.tflite"
/// # )));
/// # fn main() -> Result<(), nanoglia::Error> {
/// use nanoglia::{Arena, Encoding, Graph, Target};
///
/// let mut memory = [0u8; 4 * 1024];
/// let arena = Arena::new(&mut memory);
/// let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
/// let mut ctx = graph.init_execution_context_in(&arena)?;
///
/// for sample in [[1i8; 36], [-3i8; 36]] {
///     // Inputs are written again before each inference.
///     ctx.input_mut(0)?.as_mut_slice::<i8>()?.copy_from_slice(&sample);
///     ctx.compute()?;
///     let scores = ctx.output_by_name("output")?;
///     let best = scores.as_slice::<i8>()?.iter().max();
///     assert!(best.is_some());
/// }
/// # Ok(())
/// # }
/// ```
pub struct ExecutionContext<'g, 'm, A: Allocator, B: Allocator> {
    graph: &'g Graph<'m, A>,
    arena: Block<B>,
}

impl<A: Allocator, B: Allocator> core::fmt::Debug for ExecutionContext<'_, '_, A, B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("arena_bytes", &self.arena.len())
            .finish()
    }
}

impl<'g, 'm, A: Allocator, B: Allocator> ExecutionContext<'g, 'm, A, B> {
    pub(crate) fn new(graph: &'g Graph<'m, A>, alloc: B) -> Result<Self> {
        Ok(Self {
            graph,
            arena: Block::new_zeroed(graph.context_bytes(), alloc)?,
        })
    }

    /// The graph this context runs.
    pub fn graph(&self) -> &'g Graph<'m, A> {
        self.graph
    }

    /// Copies `data` (raw little-endian bytes) into input `index`. Its length must match the
    /// tensor size exactly, otherwise the error carries [`Detail::SizeMismatch`]. To fill the
    /// input in place without a copy, use [`Self::input_mut`].
    pub fn set_input(&mut self, index: usize, data: &[u8]) -> Result<()> {
        let mut input = self.input_mut(index)?;
        let bytes = input.bytes_mut();
        if bytes.len() != data.len() {
            return Err(Error::invalid_argument(Detail::SizeMismatch {
                expected: bytes.len() as u32,
                actual: data.len() as u32,
            }));
        }
        bytes.copy_from_slice(data);
        Ok(())
    }

    /// A writable view of input `index`, directly in the arena (no copy).
    pub fn input_mut(&mut self, index: usize) -> Result<TensorMut<'_>> {
        let (desc, location) = self.graph.input_tensor(index)?;
        let Location::Arena { offset, len } = location else {
            return Err(Error::invalid_argument(Detail::IndexOutOfRange {
                index: index as u32,
            }));
        };
        let start = offset as usize;
        let data = &mut self.arena.as_mut_slice()[start..start + len as usize];
        Ok(TensorMut::new(desc, data))
    }

    /// A read-only view of output `index`.
    pub fn output(&self, index: usize) -> Result<TensorRef<'_>> {
        let (desc, location) = self.graph.output_tensor(index)?;
        let data: &[u8] = match location {
            Location::Arena { offset, len } => {
                &self.arena.as_slice()[offset as usize..offset as usize + len as usize]
            }
            Location::Const { offset, len } => {
                &self.graph.model()[offset as usize..offset as usize + len as usize]
            }
            Location::Unused => &[],
        };
        Ok(TensorRef::new(desc, data))
    }

    /// A read-only view of the output named `name`.
    pub fn output_by_name(&self, name: &str) -> Result<TensorRef<'_>> {
        self.output(self.graph.output_index(name)?)
    }

    /// Runs the graph on the current inputs. Allocates nothing.
    ///
    /// The same as [`Self::compute_with`] with the observer `()`, which compiles to nothing.
    #[inline]
    pub fn compute(&mut self) -> Result<()> {
        self.compute_with(&mut ())
    }

    /// Runs the graph on the current inputs, calling `observer` before and after each operator.
    /// Allocates nothing (unless the observer does).
    ///
    /// The observer sees each operator through an [`OpInfo`]: its index, its kernel, and views of
    /// its input and output tensors, built only when asked for. If a kernel fails,
    /// [`Observer::after`] receives its error, then `compute_with` returns it.
    ///
    /// Time each operator (here with `std`; on a microcontroller, read a cycle counter):
    ///
    /// ```
    /// # #[repr(C, align(16))]
    /// # struct Aligned<T: ?Sized>(T);
    /// # static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
    /// #     env!("CARGO_MANIFEST_DIR"),
    /// #     "/doc/classifier.tflite"
    /// # )));
    /// # fn main() -> Result<(), nanoglia::Error> {
    /// use nanoglia::{Arena, Encoding, Graph, Observer, OpInfo, Result, Target};
    /// use std::time::{Duration, Instant};
    ///
    /// /// The kernel and the duration of each operator.
    /// #[derive(Default)]
    /// struct Profiler {
    ///     start: Option<Instant>,
    ///     times: Vec<(&'static str, Duration)>,
    /// }
    ///
    /// impl Observer for Profiler {
    ///     fn before(&mut self, _op: &OpInfo<'_>) {
    ///         self.start = Some(Instant::now());
    ///     }
    ///
    ///     fn after(&mut self, op: &OpInfo<'_>, _result: &Result<()>) {
    ///         if let Some(start) = self.start.take() {
    ///             self.times.push((op.kernel_name(), start.elapsed()));
    ///         }
    ///     }
    /// }
    ///
    /// let mut memory = [0u8; 4 * 1024];
    /// let arena = Arena::new(&mut memory);
    /// let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
    /// let mut ctx = graph.init_execution_context_in(&arena)?;
    ///
    /// ctx.input_mut(0)?.as_mut_slice::<i8>()?.fill(10);
    /// let mut profiler = Profiler::default();
    /// ctx.compute_with(&mut profiler)?;
    /// assert_eq!(profiler.times.len(), graph.operator_count());
    /// for (kernel, time) in &profiler.times {
    ///     println!("{kernel}: {time:?}");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute_with<O: Observer + ?Sized>(&mut self, observer: &mut O) -> Result<()> {
        let graph = self.graph;
        let model = graph.model();
        let io = graph.io_table();
        let tensors = graph.tensors();
        let (scratch_offset, scratch_len) = graph.scratch();
        let arena_len = self.arena.len();
        let base = self.arena.as_mut_ptr();

        for (index, op) in graph.ops().iter().enumerate() {
            let start = op.io_start as usize;
            let inputs = &io[start..start + op.inputs as usize];
            let outputs =
                &io[start + op.inputs as usize..start + op.inputs as usize + op.outputs as usize];
            // The arena as the observer sees it, between kernels. Built anew for each call: no
            // shared view of the arena may live while a kernel writes into it.
            let info = |arena| OpInfo {
                index,
                kernel: op.kernel,
                inputs,
                outputs,
                tensors,
                arena,
            };
            // SAFETY: no mutable view of the arena is alive: the previous kernel has returned.
            observer.before(&info(unsafe {
                core::slice::from_raw_parts(base, arena_len)
            }));

            let mut input_bytes: [&[u8]; MAX_OP_INPUTS] = [&[]; MAX_OP_INPUTS];
            for (slot, &raw) in input_bytes.iter_mut().zip(inputs) {
                *slot = match graph.location(raw) {
                    Location::Unused => &[],
                    Location::Const { offset, len } => {
                        &model[offset as usize..offset as usize + len as usize]
                    }
                    Location::Arena { offset, len } => {
                        debug_assert!(offset as usize + len as usize <= arena_len);
                        // SAFETY: the memory plan places every tensor inside the arena, and the
                        // tensors alive during this operator (its inputs and outputs) do not
                        // overlap each other nor the scratch region. Inputs are only read.
                        unsafe {
                            core::slice::from_raw_parts(base.add(offset as usize), len as usize)
                        }
                    }
                };
            }
            let mut output_bytes: [&mut [u8]; MAX_OP_OUTPUTS] =
                core::array::from_fn(|_| <&mut [u8]>::default());
            for (slot, &raw) in output_bytes.iter_mut().zip(outputs) {
                if let Location::Arena { offset, len } = graph.location(raw) {
                    debug_assert!(offset as usize + len as usize <= arena_len);
                    // SAFETY: as above; outputs of an operator are distinct tensors (checked at
                    // load) that do not overlap its inputs.
                    *slot = unsafe {
                        core::slice::from_raw_parts_mut(base.add(offset as usize), len as usize)
                    };
                }
            }
            debug_assert!(scratch_offset + scratch_len <= arena_len);
            // SAFETY: the scratch region follows every activation in the arena.
            let scratch =
                unsafe { core::slice::from_raw_parts_mut(base.add(scratch_offset), scratch_len) };

            let mut ctx = EvalCtx {
                inputs: &input_bytes[..op.inputs as usize],
                outputs: &mut output_bytes[..op.outputs as usize],
                persist: graph.persist(op),
                scratch,
            };
            let result = op.kernel.eval(&mut ctx);
            // SAFETY: the kernel has returned, and its views of the arena are no longer used.
            observer.after(
                &info(unsafe { core::slice::from_raw_parts(base, arena_len) }),
                &result,
            );
            result?;
        }
        Ok(())
    }
}

/// Watches the operators of [`ExecutionContext::compute_with`]: to time them, trace them, or check
/// intermediate tensors.
///
/// Both methods do nothing by default. `()` is the observer that watches nothing: with it,
/// `compute_with` compiles to the same code as without observer ([`ExecutionContext::compute`]).
pub trait Observer {
    /// Called before operator `op` runs. Its inputs are ready; the contents of its outputs are
    /// undefined (whatever the arena held there).
    #[inline(always)]
    fn before(&mut self, op: &OpInfo<'_>) {
        let _ = op;
    }

    /// Called after operator `op` ran, with the kernel's result. Its outputs hold the computed
    /// values if `result` is `Ok`.
    #[inline(always)]
    fn after(&mut self, op: &OpInfo<'_>, result: &Result<()>) {
        let _ = (op, result);
    }
}

/// Watches nothing.
impl Observer for () {}

impl<O: Observer + ?Sized> Observer for &mut O {
    #[inline(always)]
    fn before(&mut self, op: &OpInfo<'_>) {
        (**self).before(op);
    }

    #[inline(always)]
    fn after(&mut self, op: &OpInfo<'_>, result: &Result<()>) {
        (**self).after(op, result);
    }
}

/// An operator, as an [`Observer`] sees it during [`ExecutionContext::compute_with`].
///
/// Everything is computed only when asked for: an observer that only reads [`Self::index`]
/// costs no lookup.
pub struct OpInfo<'a> {
    index: usize,
    kernel: &'a dyn Kernel,
    inputs: &'a [i32],
    outputs: &'a [i32],
    tensors: GraphTensors<'a>,
    arena: &'a [u8],
}

impl core::fmt::Debug for OpInfo<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OpInfo")
            .field("index", &self.index)
            .field("kernel", &self.kernel.name())
            .finish()
    }
}

impl<'a> OpInfo<'a> {
    /// Index of the operator in the graph (execution order), as in
    /// [`Graph::kernel_name`](crate::Graph::kernel_name).
    pub fn index(&self) -> usize {
        self.index
    }

    /// Name of the kernel running the operator.
    pub fn kernel_name(&self) -> &'static str {
        self.kernel.name()
    }

    /// The operator code, decoded from the model at each call. The graph decoded every operator
    /// when it loaded, from the same bytes, so this does not fail on a loaded graph.
    pub fn opcode(&self) -> Result<OpCode<'a>> {
        self.tensors.opcode(self.index)
    }

    /// Number of inputs, omitted optional inputs included.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Number of outputs.
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// A read-only view of input `i`: its contents are those the kernel reads.
    ///
    /// Fails with [`Detail::IndexOutOfRange`] if `i >= self.input_count()`, and with
    /// [`Detail::TensorNotFound`] (code `NotFound`) for an omitted optional input.
    pub fn input(&self, i: usize) -> Result<TensorRef<'_>> {
        self.tensor(self.inputs, i)
    }

    /// A read-only view of output `i`. In [`Observer::before`] its contents are undefined; in
    /// [`Observer::after`] they are the computed values (if the kernel succeeded), valid until
    /// a later operator reuses their memory.
    ///
    /// Fails with [`Detail::IndexOutOfRange`] if `i >= self.output_count()`.
    pub fn output(&self, i: usize) -> Result<TensorRef<'_>> {
        self.tensor(self.outputs, i)
    }

    fn tensor(&self, list: &[i32], i: usize) -> Result<TensorRef<'_>> {
        let raw = *list
            .get(i)
            .ok_or_else(|| Error::invalid_argument(Detail::IndexOutOfRange { index: i as u32 }))?;
        self.tensors.view(raw, self.arena)
    }
}
