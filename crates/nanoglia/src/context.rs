use crate::graph::{Graph, GraphTensors, Location};
use crate::{GraphHandle, TensorMut, TensorRef};
use core::future::Future;
use core::ops::ControlFlow;
use core::pin::Pin;
use core::task::{Context, Poll};
use nanoglia_core::backend::{EvalCtx, Kernel};
use nanoglia_core::{
    Allocator, Block, Detail, Error, ErrorCode, MAX_OP_INPUTS, MAX_OP_OUTPUTS, OpCode, Result,
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
pub struct GraphExecutionContext<G: GraphHandle, B: Allocator> {
    graph: G,
    arena: Block<B>,
    /// The next operator of the inference run step by step, if one is in progress.
    next: Option<usize>,
}

impl<G: GraphHandle, B: Allocator> core::fmt::Debug for GraphExecutionContext<G, B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GraphExecutionContext")
            .field("arena_bytes", &self.arena.len())
            .finish()
    }
}

/// An execution context that borrows its graph; no ownership allocation is introduced.
///
/// [`Self::graph`] borrows the graph for as long as the context is borrowed, as for every
/// [`GraphHandle`].
pub type ExecutionContext<'g, 'm, A, B> = GraphExecutionContext<&'g Graph<'m, A>, B>;

impl<'g, 'm, A: Allocator, B: Allocator> ExecutionContext<'g, 'm, A, B> {
    pub(crate) fn new(graph: &'g Graph<'m, A>, alloc: B) -> Result<Self> {
        Self::new_in(graph, alloc)
    }
}

impl<G: GraphHandle, B: Allocator> GraphExecutionContext<G, B> {
    /// Retains `graph` and allocates one arena from `alloc`. Inference allocates nothing.
    pub fn new_in(graph: G, alloc: B) -> Result<Self> {
        let bytes = graph.graph().context_bytes();
        Ok(Self {
            graph,
            arena: Block::new_zeroed(bytes, alloc)?,
            next: None,
        })
    }

    /// Borrows the prepared graph through this context's retained handle.
    ///
    /// Graph and model views are tied to this borrow of the context, for both borrowed
    /// and owning handles. They cannot outlive an owning context's model storage.
    ///
    /// ```compile_fail
    /// # use nanoglia::{GraphExecutionContext, OwnedGraph};
    /// # use nanoglia::core::allocator_api2::alloc::Global;
    /// # use std::sync::Arc;
    /// fn cannot_escape(context: GraphExecutionContext<Arc<OwnedGraph<Global>>, Global>) {
    ///     let input = context.graph().input(0).unwrap();
    ///     drop(context);
    ///     println!("{:?}", input);
    /// }
    /// ```
    pub fn graph(&self) -> &Graph<'_, G::Allocator> {
        self.graph.graph()
    }

    /// The retained graph handle, for example to check shared-owner identity.
    pub fn graph_handle(&self) -> &G {
        &self.graph
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
        let (desc, location) = self.graph.graph().input_tensor(index)?;
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
        let (desc, location) = self.graph.graph().output_tensor(index)?;
        let data: &[u8] = match location {
            Location::Arena { offset, len } => {
                &self.arena.as_slice()[offset as usize..offset as usize + len as usize]
            }
            Location::Const { offset, len } => {
                &self.graph.graph().model()[offset as usize..offset as usize + len as usize]
            }
            Location::Unused => &[],
        };
        Ok(TensorRef::new(desc, data))
    }

    /// A read-only view of the output named `name`.
    pub fn output_by_name(&self, name: &str) -> Result<TensorRef<'_>> {
        self.output(self.graph.graph().output_index(name)?)
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
    /// use core::ops::ControlFlow;
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
    ///     fn before(&mut self, _op: &OpInfo<'_>) -> ControlFlow<()> {
    ///         self.start = Some(Instant::now());
    ///         ControlFlow::Continue(())
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
        self.next = None;
        for index in 0..self.graph.graph().operator_count() {
            self.run_operator(index, observer)?;
        }
        Ok(())
    }

    /// Runs the next operator of the current inference, starting a new inference (on the current
    /// inputs) if none is in progress, and returns whether it was the last one.
    ///
    /// Use it to run a long inference in slices, between which the application does something
    /// else: answer a radio, refresh a display, check a deadline. Stopping is free: stop calling
    /// `step`, or call [`Self::cancel`]. The granularity is one operator: a single operator
    /// (a large convolution) cannot be split.
    ///
    /// Do not change the inputs while an inference is in progress: the operators already run
    /// would not see the change, and the memory plan may already have reused the inputs' memory.
    /// Outputs hold their final values only once `step` returns [`Step::Done`]. If an operator
    /// fails, its error is returned and the inference is abandoned: the next `step` starts over.
    ///
    /// ```
    /// # #[repr(C, align(16))]
    /// # struct Aligned<T: ?Sized>(T);
    /// # static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
    /// #     env!("CARGO_MANIFEST_DIR"),
    /// #     "/doc/classifier.tflite"
    /// # )));
    /// # fn main() -> Result<(), nanoglia::Error> {
    /// use nanoglia::{Arena, Encoding, Graph, Step, Target};
    ///
    /// let mut memory = [0u8; 4 * 1024];
    /// let arena = Arena::new(&mut memory);
    /// let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
    /// let mut ctx = graph.init_execution_context_in(&arena)?;
    ///
    /// ctx.input_mut(0)?.as_mut_slice::<i8>()?.fill(10);
    /// while ctx.step()? == Step::Pending {
    ///     // Anything else the firmware has to do between two operators.
    /// }
    /// assert_eq!(ctx.output(0)?.as_slice::<i8>()?.len(), 18);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn step(&mut self) -> Result<Step> {
        self.step_with(&mut ())
    }

    /// [`Self::step`], calling `observer` around the operator it runs.
    pub fn step_with<O: Observer + ?Sized>(&mut self, observer: &mut O) -> Result<Step> {
        let count = self.graph.graph().operator_count();
        let index = self.next.take().unwrap_or(0);
        if index >= count {
            return Ok(Step::Done);
        }
        // `next` stays `None` if the operator fails: the inference is abandoned.
        self.run_operator(index, observer)?;
        if index + 1 == count {
            Ok(Step::Done)
        } else {
            self.next = Some(index + 1);
            Ok(Step::Pending)
        }
    }

    /// Abandons the inference run step by step, if any: the next [`Self::step`] starts a new one.
    pub fn cancel(&mut self) {
        self.next = None;
    }

    /// The index of the next operator [`Self::step`] will run, if an inference is in progress.
    pub fn next_operator(&self) -> Option<usize> {
        self.next
    }

    /// Runs the graph on the current inputs, yielding to the async executor between two
    /// operators so that other tasks run during a long inference. Allocates nothing and needs no
    /// particular executor (it works with embassy, RTIC or your own).
    ///
    /// Dropping the future abandons the inference. Like [`Self::step`], it cannot yield in the
    /// middle of an operator.
    pub async fn compute_async(&mut self) -> Result<()> {
        self.compute_async_with(&mut ()).await
    }

    /// [`Self::compute_async`], calling `observer` around each operator.
    pub async fn compute_async_with<O: Observer + ?Sized>(
        &mut self,
        observer: &mut O,
    ) -> Result<()> {
        self.next = None;
        loop {
            match self.step_with(observer) {
                Ok(Step::Done) => return Ok(()),
                Ok(Step::Pending) => YieldNow(false).await,
                Err(error) => return Err(error),
            }
        }
    }

    /// Runs operator `index`, with the observer around it.
    #[inline]
    fn run_operator<O: Observer + ?Sized>(&mut self, index: usize, observer: &mut O) -> Result<()> {
        let graph = self.graph.graph();
        let model = graph.model();
        let io = graph.io_table();
        let tensors = graph.tensors();
        let (scratch_offset, scratch_len) = graph.scratch();
        let arena_len = self.arena.len();
        let base = self.arena.as_mut_ptr();
        let Some(op) = graph.ops().get(index) else {
            return Err(Error::invalid_argument(Detail::IndexOutOfRange {
                index: index as u32,
            }));
        };
        {
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
            let flow = observer.before(&info(unsafe {
                core::slice::from_raw_parts(base, arena_len)
            }));
            if flow.is_break() {
                return Err(Error::new(
                    ErrorCode::Timeout,
                    Detail::Interrupted {
                        op_index: index as u32,
                    },
                ));
            }

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
            result
        }
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
    ///
    /// Return [`ControlFlow::Break`] to stop the computation before this operator, for example
    /// when a deadline has passed: `compute*` then returns an error with code
    /// [`ErrorCode::Timeout`] and [`Detail::Interrupted`].
    #[inline(always)]
    fn before(&mut self, op: &OpInfo<'_>) -> ControlFlow<()> {
        let _ = op;
        ControlFlow::Continue(())
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
    fn before(&mut self, op: &OpInfo<'_>) -> ControlFlow<()> {
        (**self).before(op)
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

/// Progress of an inference run step by step ([`ExecutionContext::step`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Step {
    /// Operators remain: call `step` again.
    Pending,
    /// The inference is complete: the outputs hold its result.
    Done,
}

/// A future that is pending once, waking itself, then ready: lets the executor run other tasks.
struct YieldNow(bool);

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}
