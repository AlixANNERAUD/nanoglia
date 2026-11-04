use crate::planner::{self, Interval};
use crate::{Encoding, ExecutionContext, Registry, Target};
use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use nanoglia_core::backend::{BackendKind, Kernel};
use nanoglia_core::le::LeSlice;
use nanoglia_core::{
    ALIGN, Allocator, Block, DataType, Detail, Dims, Error, ErrorCode, MAX_OP_INPUTS,
    MAX_OP_OUTPUTS, OpDesc, Quantization, Result, TensorDesc, align_up,
};
use nanoglia_tflite::{Model, SubGraph};

/// Where a tensor lives at execution time.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Location {
    /// Not used by any operator.
    Unused,
    /// Constant data, at this offset in the model bytes.
    Const { offset: u32, len: u32 },
    /// Activation, at this offset in the context arena.
    Arena { offset: u32, len: u32 },
}

/// A prepared operator.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OpEntry<'k> {
    /// The selected kernel, borrowed from a backend of the registry.
    pub kernel: &'k dyn Kernel,
    pub persist_offset: u32,
    pub persist_len: u32,
    /// Index of the first input in the I/O table; outputs follow the inputs.
    pub io_start: u32,
    pub inputs: u16,
    pub outputs: u16,
}

/// A typed array inside the graph block.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Section<T> {
    offset: usize,
    len: usize,
    _type: PhantomData<T>,
}

impl<T> Section<T> {
    fn new(layout: &mut Layout, len: usize) -> Result<Self> {
        let array = Layout::array::<T>(len).map_err(|_| Error::out_of_memory(usize::MAX))?;
        let (extended, offset) = layout
            .extend(array)
            .map_err(|_| Error::out_of_memory(usize::MAX))?;
        *layout = extended;
        Ok(Self {
            offset,
            len,
            _type: PhantomData,
        })
    }

    fn get<'b>(&self, block: &'b [u8]) -> &'b [T] {
        debug_assert!(self.offset + self.len * size_of::<T>() <= block.len());
        // SAFETY: the section lies within the block (laid out by `Layout::extend`), is aligned
        // for `T` (the block is `ALIGN`-aligned and `T`'s alignment is at most `ALIGN`), and was
        // fully initialised during `Graph::load` before any shared access.
        unsafe {
            core::slice::from_raw_parts(block.as_ptr().add(self.offset).cast::<T>(), self.len)
        }
    }

    fn uninit<'b>(&self, block: &'b mut [u8]) -> &'b mut [MaybeUninit<T>] {
        debug_assert!(self.offset + self.len * size_of::<T>() <= block.len());
        // SAFETY: in bounds and aligned as above; `MaybeUninit` needs no initialisation.
        unsafe {
            core::slice::from_raw_parts_mut(
                block.as_mut_ptr().add(self.offset).cast::<MaybeUninit<T>>(),
                self.len,
            )
        }
    }
}

/// A loaded model, ready to create execution contexts.
///
/// The graph borrows the model bytes and the backends of its [`Registry`] (`'m`): weights are
/// never copied, and each operator keeps a reference to the kernel its backend selected. It owns
/// one block of memory from `A` holding the operator table, the prepared parameters and the
/// memory plan. A graph is immutable: create one [`ExecutionContext`] per concurrent inference.
/// It is `Send` and `Sync` when `A` is (kernels and backends are `Sync`).
///
/// # Example
///
/// Load a model from bytes the firmware already holds (here a small int8 classifier: input
/// `[3, 12]`, output `[3, 6]`), with all memory taken from a buffer on the stack:
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
///
/// let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
/// let mut ctx = graph.init_execution_context_in(&arena)?;
///
/// ctx.input_mut(0)?.as_mut_slice::<i8>()?.fill(10);
/// ctx.compute()?;
/// let scores = ctx.output(0)?;
/// assert_eq!(scores.dims().as_slice(), &[3, 6]);
/// assert_eq!(scores.as_slice::<i8>()?.len(), 18);
/// # Ok(())
/// # }
/// ```
pub struct Graph<'m, A: Allocator> {
    model: &'m [u8],
    subgraph: SubGraph<'m>,
    block: Block<A>,
    ops: Section<OpEntry<'m>>,
    locations: Section<Location>,
    io: Section<i32>,
    arena_size: usize,
    scratch_offset: usize,
    scratch_size: usize,
}

impl<A: Allocator> core::fmt::Debug for Graph<'_, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Graph")
            .field("operators", &self.ops.len)
            .field("tensors", &self.locations.len)
            .field("graph_bytes", &self.block.len())
            .field("context_bytes", &self.arena_size)
            .finish()
    }
}

const NO_TENSOR: TensorDesc<'static> = TensorDesc {
    index: u32::MAX,
    dtype: DataType::I8,
    dims: Dims::scalar(),
    quant: Quantization::NONE,
    data: None,
    name: "",
};

/// The operator descriptions of one operator, built on the stack.
struct OpDescs<'m> {
    inputs: [Option<TensorDesc<'m>>; MAX_OP_INPUTS],
    outputs: [TensorDesc<'m>; MAX_OP_OUTPUTS],
}

fn malformed() -> Error {
    Error::new(
        ErrorCode::InvalidEncoding,
        Detail::MalformedModel { offset: 0 },
    )
}

fn tensor_index(raw: i32, count: usize) -> Result<Option<usize>> {
    match raw {
        -1 => Ok(None),
        i if i >= 0 && (i as usize) < count => Ok(Some(i as usize)),
        _ => Err(malformed()),
    }
}

impl<'m, A: Allocator> Graph<'m, A> {
    /// Loads a model with the default [`Registry`] (the backends enabled by the crate features),
    /// taking all its memory from `alloc`.
    ///
    /// `builders` holds the model bytes (a single buffer for TFLite), which the graph borrows.
    /// They must stay valid and unchanged as long as the graph lives; constant buffers must be
    /// aligned for their type, which holds when the model starts on a 16-byte boundary.
    ///
    /// The model is verified first: a malformed model returns an error, never panics.
    ///
    /// ```
    /// # #[repr(C, align(16))]
    /// # struct Aligned<T: ?Sized>(T);
    /// # static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
    /// #     env!("CARGO_MANIFEST_DIR"),
    /// #     "/doc/classifier.tflite"
    /// # )));
    /// # fn main() -> Result<(), nanoglia::Error> {
    /// use nanoglia::{Arena, Encoding, ErrorCode, Graph, Target};
    ///
    /// let mut memory = [0u8; 4 * 1024];
    /// let arena = Arena::new(&mut memory);
    /// let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
    /// assert_eq!(graph.operator_count(), 1);
    ///
    /// // Not a TFLite model: an error, not a panic.
    /// let error = Graph::load_in(&[b"not a model"], Encoding::Autodetect, Target::Auto, &arena)
    ///     .unwrap_err();
    /// assert_eq!(error.code, ErrorCode::InvalidEncoding);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_in(
        builders: &[&'m [u8]],
        encoding: Encoding,
        target: Target,
        alloc: A,
    ) -> Result<Self> {
        Self::load_with(&Registry::default(), builders, encoding, target, alloc)
    }

    /// Loads a model, picking kernels from `registry`: for each operator, the first backend of the
    /// registry that supports it. Use it to add your own backends or kernels, or to restrict
    /// the backends a firmware links.
    ///
    /// The graph borrows the backends for as long as it lives (the same lifetime `'m` as the
    /// model bytes; the default registry is `'static`), so a backend may own state its kernels
    /// use, such as a driver handle. See the guide *Write a backend* in the documentation.
    ///
    /// ```
    /// # #[repr(C, align(16))]
    /// # struct Aligned<T: ?Sized>(T);
    /// # static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
    /// #     env!("CARGO_MANIFEST_DIR"),
    /// #     "/doc/classifier.tflite"
    /// # )));
    /// # fn main() -> Result<(), nanoglia::Error> {
    /// use nanoglia::{Arena, Detail, Encoding, ErrorCode, Graph, Registry, Target};
    ///
    /// let mut memory = [0u8; 4 * 1024];
    /// let arena = Arena::new(&mut memory);
    ///
    /// // A registry without any backend: the model's FULLY_CONNECTED has no kernel.
    /// let empty = Registry::new(&[]);
    /// let error = Graph::load_with(&empty, &[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)
    ///     .unwrap_err();
    /// assert_eq!(error.code, ErrorCode::UnsupportedOperation);
    /// assert!(matches!(error.detail, Detail::UnsupportedOp { op_index: 0, .. }));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// A graph cannot outlive the backends it borrows:
    ///
    /// ```compile_fail,E0597
    /// # #[repr(C, align(16))]
    /// # struct Aligned<T: ?Sized>(T);
    /// # static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
    /// #     env!("CARGO_MANIFEST_DIR"),
    /// #     "/doc/classifier.tflite"
    /// # )));
    /// use nanoglia::core::OpDesc;
    /// use nanoglia::core::allocator_api2::alloc::Global;
    /// use nanoglia::core::backend::{Backend, Kernel};
    /// use nanoglia::{Encoding, Graph, Registry, Target};
    ///
    /// struct Local;
    /// impl Backend for Local {
    ///     fn name(&self) -> &'static str {
    ///         "local"
    ///     }
    ///     fn select<'s>(&'s self, op: &OpDesc<'_, '_>) -> Option<&'s dyn Kernel> {
    ///         Registry::default().backends().iter().find_map(|b| b.select(op))
    ///     }
    /// }
    ///
    /// let graph = {
    ///     let local = Local;
    ///     let backends: [&dyn Backend; 1] = [&local];
    ///     let registry = Registry::new(&backends);
    ///     Graph::load_with(&registry, &[&MODEL.0], Encoding::Autodetect, Target::Auto, Global)
    /// }; // error: `local` dropped while the graph still borrows it
    /// # drop(graph);
    /// ```
    pub fn load_with(
        registry: &Registry<'m>,
        builders: &[&'m [u8]],
        encoding: Encoding,
        target: Target,
        alloc: A,
    ) -> Result<Self> {
        let (bytes, subgraph) = open(builders, encoding)?;
        let sizing = Sizing::compute(registry, target, &subgraph)?;
        let tensor_count = subgraph.tensor_count();
        let mut block = Block::new_zeroed(sizing.block_size, alloc)?;
        let memory = block.as_mut_slice();

        // Memory plan, in the working sections of the block.
        let (intervals, order) = sizing.sections.working_memory(memory);
        let activations = plan_activations(&subgraph, intervals, order)?;

        // Tensor locations.
        for t in 0..tensor_count {
            let desc = subgraph.tensor(t)?;
            let location = if let Some(data) = desc.data {
                let offset = data.as_ptr() as usize - bytes.as_ptr() as usize;
                let align = desc.dtype.size().unwrap_or(1);
                if data.as_ptr().align_offset(align) != 0 {
                    return Err(Error::invalid_argument(Detail::Misaligned {
                        tensor: t as u32,
                    }));
                }
                Location::Const {
                    offset: offset as u32,
                    len: data.len() as u32,
                }
            } else {
                let (intervals, _) = sizing.sections.working_memory(memory);
                let iv = intervals[t];
                if iv.first == u32::MAX {
                    Location::Unused
                } else {
                    Location::Arena {
                        offset: iv.offset,
                        len: iv.size,
                    }
                }
            };
            sizing.sections.locations.uninit(memory)[t].write(location);
        }

        // Kernels: prepare their parameters and fill the operator and I/O tables.
        let mut persist_offset = sizing.persist_start;
        let mut io_next = 0usize;
        for index in 0..subgraph.operator_count() {
            let mut descs = OpDescs {
                inputs: [None; MAX_OP_INPUTS],
                outputs: [NO_TENSOR; MAX_OP_OUTPUTS],
            };
            let (desc, input_ids, output_ids) = describe(&subgraph, index, &mut descs)?;
            let (kernel, _) = select(registry, target, &desc)?;
            let plan = kernel.plan(&desc)?;
            if plan.scratch > sizing.scratch_size {
                // The backend selected or planned differently in the first pass.
                return Err(Error::new(ErrorCode::Unknown, Detail::None));
            }
            let persist_len = plan.persistent;
            let region = persist_offset
                .checked_add(persist_len)
                .and_then(|end| memory.get_mut(persist_offset..end))
                .ok_or_else(|| Error::new(ErrorCode::Unknown, Detail::None))?;
            kernel.prepare(&desc, region)?;

            let io_slots = sizing.sections.io.uninit(memory);
            let io_start = io_next;
            for raw in input_ids.iter().chain(output_ids.iter()) {
                io_slots[io_next].write(raw);
                io_next += 1;
            }
            sizing.sections.ops.uninit(memory)[index].write(OpEntry {
                kernel,
                persist_offset: persist_offset as u32,
                persist_len: persist_len as u32,
                io_start: io_start as u32,
                inputs: input_ids.len() as u16,
                outputs: output_ids.len() as u16,
            });
            persist_offset = align_up(persist_len)
                .and_then(|len| persist_offset.checked_add(len))
                .ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        }

        let scratch_offset =
            align_up(activations).ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        let scratch_size =
            align_up(sizing.scratch_size).ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        let arena_size = scratch_offset
            .checked_add(scratch_size)
            .ok_or_else(|| Error::out_of_memory(usize::MAX))?;

        Ok(Self {
            model: bytes,
            subgraph,
            block,
            ops: sizing.sections.ops,
            locations: sizing.sections.locations,
            io: sizing.sections.io,
            arena_size,
            scratch_offset,
            scratch_size,
        })
    }

    /// The memory this graph uses: its own block and the arena of each execution context.
    pub fn memory_requirements(&self) -> MemoryRequirements {
        MemoryRequirements {
            graph_bytes: self.block.len(),
            context_bytes: self.arena_size,
        }
    }

    /// Creates an execution context, taking its arena (activations and scratch memory) from
    /// `alloc`.
    pub fn init_execution_context_in<B: Allocator>(
        &self,
        alloc: B,
    ) -> Result<ExecutionContext<'_, 'm, A, B>> {
        ExecutionContext::new(self, alloc)
    }

    /// Number of inputs.
    pub fn input_count(&self) -> usize {
        self.subgraph.inputs().len()
    }

    /// Number of outputs.
    pub fn output_count(&self) -> usize {
        self.subgraph.outputs().len()
    }

    /// Description of input `index`.
    pub fn input(&self, index: usize) -> Result<TensorDesc<'m>> {
        self.io_desc(self.subgraph.inputs(), index)
    }

    /// Description of output `index`.
    pub fn output(&self, index: usize) -> Result<TensorDesc<'m>> {
        self.io_desc(self.subgraph.outputs(), index)
    }

    /// Index of the input named `name`.
    pub fn input_index(&self, name: &str) -> Result<usize> {
        self.find(self.subgraph.inputs(), name)
    }

    /// Index of the output named `name`.
    pub fn output_index(&self, name: &str) -> Result<usize> {
        self.find(self.subgraph.outputs(), name)
    }

    /// Bytes of the graph block (kernel table, prepared parameters, memory plan).
    pub fn graph_bytes(&self) -> usize {
        self.block.len()
    }

    /// Bytes each execution context allocates (activations and scratch memory).
    pub fn context_bytes(&self) -> usize {
        self.arena_size
    }

    /// Number of operators.
    pub fn operator_count(&self) -> usize {
        self.ops.len
    }

    /// Name of the kernel selected for operator `index`, for diagnostics.
    pub fn kernel_name(&self, index: usize) -> Option<&'static str> {
        self.ops().get(index).map(|op| op.kernel.name())
    }

    fn io_desc(&self, list: LeSlice<'m, i32>, index: usize) -> Result<TensorDesc<'m>> {
        let raw = list.get(index).ok_or_else(|| {
            Error::invalid_argument(Detail::IndexOutOfRange {
                index: index as u32,
            })
        })?;
        let t = tensor_index(raw, self.locations.len)?.ok_or_else(malformed)?;
        self.subgraph.tensor(t)
    }

    fn find(&self, list: LeSlice<'m, i32>, name: &str) -> Result<usize> {
        for (i, raw) in list.iter().enumerate() {
            if let Some(t) = tensor_index(raw, self.locations.len)?
                && self.subgraph.tensor(t)?.name == name
            {
                return Ok(i);
            }
        }
        Err(Error::new(ErrorCode::NotFound, Detail::TensorNotFound))
    }

    pub(crate) fn ops(&self) -> &[OpEntry<'m>] {
        self.ops.get(self.block.as_slice())
    }

    pub(crate) fn io_table(&self) -> &[i32] {
        self.io.get(self.block.as_slice())
    }

    pub(crate) fn location(&self, raw: i32) -> Location {
        match tensor_index(raw, self.locations.len) {
            Ok(Some(t)) => self.locations.get(self.block.as_slice())[t],
            _ => Location::Unused,
        }
    }

    pub(crate) fn persist(&self, op: &OpEntry<'_>) -> &[u8] {
        let start = op.persist_offset as usize;
        &self.block.as_slice()[start..start + op.persist_len as usize]
    }

    pub(crate) fn model(&self) -> &'m [u8] {
        self.model
    }

    pub(crate) fn scratch(&self) -> (usize, usize) {
        (self.scratch_offset, self.scratch_size)
    }

    pub(crate) fn input_tensor(&self, index: usize) -> Result<(TensorDesc<'m>, Location)> {
        let raw = self.subgraph.inputs().get(index).ok_or_else(|| {
            Error::invalid_argument(Detail::IndexOutOfRange {
                index: index as u32,
            })
        })?;
        Ok((
            self.io_desc(self.subgraph.inputs(), index)?,
            self.location(raw),
        ))
    }

    pub(crate) fn output_tensor(&self, index: usize) -> Result<(TensorDesc<'m>, Location)> {
        let raw = self.subgraph.outputs().get(index).ok_or_else(|| {
            Error::invalid_argument(Detail::IndexOutOfRange {
                index: index as u32,
            })
        })?;
        Ok((
            self.io_desc(self.subgraph.outputs(), index)?,
            self.location(raw),
        ))
    }
}

#[cfg(feature = "alloc")]
impl<'m> Graph<'m, nanoglia_core::allocator_api2::alloc::Global> {
    /// Loads a model with the default [`Registry`], taking its memory from the global allocator.
    pub fn load(builders: &[&'m [u8]], encoding: Encoding, target: Target) -> Result<Self> {
        Self::load_in(
            builders,
            encoding,
            target,
            nanoglia_core::allocator_api2::alloc::Global,
        )
    }

    /// Creates an execution context whose arena comes from the global allocator.
    pub fn init_execution_context(
        &self,
    ) -> Result<
        ExecutionContext<
            '_,
            'm,
            nanoglia_core::allocator_api2::alloc::Global,
            nanoglia_core::allocator_api2::alloc::Global,
        >,
    > {
        self.init_execution_context_in(nanoglia_core::allocator_api2::alloc::Global)
    }
}

/// Memory a model needs, computed without loading it.
///
/// Use it to size a static [`Arena`](nanoglia_core::Arena): compute it once on the host (or in
/// a test) for your model, then reserve [`Self::arena_bytes`] bytes in the firmware.
///
/// ```
/// # #[repr(C, align(16))]
/// # struct Aligned<T: ?Sized>(T);
/// # static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
/// #     env!("CARGO_MANIFEST_DIR"),
/// #     "/doc/classifier.tflite"
/// # )));
/// # fn main() -> Result<(), nanoglia::Error> {
/// use nanoglia::{Arena, Encoding, Graph, MemoryRequirements, Target};
///
/// // Only the memory planner's working memory is needed, and given back.
/// let mut scratch = [0u8; 512];
/// let needs = MemoryRequirements::compute_in(
///     &[&MODEL.0],
///     Encoding::Autodetect,
///     Target::Auto,
///     &Arena::new(&mut scratch),
/// )?;
///
/// // A buffer of `arena_bytes(1)` holds the graph and one execution context, at any alignment.
/// let mut memory = vec![0u8; needs.arena_bytes(1)];
/// let arena = Arena::new(&mut memory);
/// let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
/// let _ctx = graph.init_execution_context_in(&arena)?;
/// assert_eq!(graph.memory_requirements(), needs);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRequirements {
    /// Bytes of the graph block (kernel table, prepared parameters, memory plan).
    pub graph_bytes: usize,
    /// Bytes of each execution context (activations and scratch memory).
    pub context_bytes: usize,
}

impl MemoryRequirements {
    /// Size of an [`Arena`](nanoglia_core::Arena) buffer that holds the graph and `contexts`
    /// execution contexts, whatever the alignment of the buffer.
    pub const fn arena_bytes(&self, contexts: usize) -> usize {
        let graph = self.graph_bytes.next_multiple_of(ALIGN);
        let context = self.context_bytes.next_multiple_of(ALIGN);
        // Up to `ALIGN - 1` bytes are lost aligning the first block.
        ALIGN - 1 + graph + contexts * context
    }

    /// Computes the requirements of a model with the default [`Registry`]. Only the memory
    /// planner's working memory (about 20 bytes per tensor) is taken from `alloc`, and it is
    /// freed before returning.
    pub fn compute_in<A: Allocator>(
        builders: &[&[u8]],
        encoding: Encoding,
        target: Target,
        alloc: A,
    ) -> Result<Self> {
        Self::compute_with(&Registry::default(), builders, encoding, target, alloc)
    }

    /// Computes the requirements of a model, picking kernels from `registry`.
    pub fn compute_with<A: Allocator>(
        registry: &Registry<'_>,
        builders: &[&[u8]],
        encoding: Encoding,
        target: Target,
        alloc: A,
    ) -> Result<Self> {
        let (_, subgraph) = open(builders, encoding)?;
        let sizing = Sizing::compute(registry, target, &subgraph)?;
        // Reject the models loading would reject for their constants, too.
        for t in 0..subgraph.tensor_count() {
            let desc = subgraph.tensor(t)?;
            if let Some(data) = desc.data
                && data.as_ptr().align_offset(desc.dtype.size().unwrap_or(1)) != 0
            {
                return Err(Error::invalid_argument(Detail::Misaligned {
                    tensor: t as u32,
                }));
            }
        }

        // Only the planner's working memory: the intervals and the order, laid out as in the
        // graph block.
        let tensor_count = subgraph.tensor_count();
        let mut layout = Layout::from_size_align(0, ALIGN).map_err(|_| Error::out_of_memory(0))?;
        let sections = WorkingSections {
            intervals: Section::new(&mut layout, tensor_count)?,
            order: Section::new(&mut layout, tensor_count)?,
        };
        let mut block = Block::new_zeroed(layout.size(), alloc)?;
        let (intervals, order) = sections.working_memory(block.as_mut_slice());
        let activations = plan_activations(&subgraph, intervals, order)?;

        let context_bytes = align_up(activations)
            .and_then(|a| a.checked_add(align_up(sizing.scratch_size)?))
            .ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        Ok(Self {
            graph_bytes: sizing.block_size,
            context_bytes,
        })
    }

    /// Computes the requirements of a model with the default [`Registry`], taking the planner's
    /// working memory from the global allocator.
    #[cfg(feature = "alloc")]
    pub fn compute(builders: &[&[u8]], encoding: Encoding, target: Target) -> Result<Self> {
        Self::compute_in(
            builders,
            encoding,
            target,
            nanoglia_core::allocator_api2::alloc::Global,
        )
    }
}

/// Checks the encoding, verifies the model and returns its bytes and its only subgraph.
fn open<'m>(builders: &[&'m [u8]], encoding: Encoding) -> Result<(&'m [u8], SubGraph<'m>)> {
    let [bytes] = builders else {
        return Err(Error::invalid_argument(Detail::None));
    };
    let bytes: &'m [u8] = bytes;
    match encoding {
        Encoding::TensorFlowLite => {}
        Encoding::Autodetect if Model::is_tflite(bytes) => {}
        Encoding::Autodetect => {
            return Err(Error::new(
                ErrorCode::InvalidEncoding,
                Detail::UnknownEncoding,
            ));
        }
    }
    let model = Model::parse(bytes)?;
    if model.subgraph_count() != 1 {
        return Err(Error::unsupported(Detail::UnsupportedStructure));
    }
    let subgraph = model.subgraph(0)?;
    if subgraph.operator_count() > u32::MAX as usize || subgraph.tensor_count() > u32::MAX as usize
    {
        return Err(Error::unsupported(Detail::UnsupportedStructure));
    }
    Ok((bytes, subgraph))
}

/// The sections of the graph block.
struct BlockSections<'k> {
    ops: Section<OpEntry<'k>>,
    locations: Section<Location>,
    io: Section<i32>,
    working: WorkingSections,
}

impl BlockSections<'_> {
    fn working_memory<'b>(&self, memory: &'b mut [u8]) -> (&'b mut [Interval], &'b mut [u32]) {
        self.working.working_memory(memory)
    }
}

/// The memory planner's working memory.
struct WorkingSections {
    intervals: Section<Interval>,
    order: Section<u32>,
}

impl WorkingSections {
    /// The two working arrays, in a zeroed block laid out with these sections.
    fn working_memory<'b>(&self, memory: &'b mut [u8]) -> (&'b mut [Interval], &'b mut [u32]) {
        assert!(
            self.intervals.offset + self.intervals.len * size_of::<Interval>() <= self.order.offset
        );
        assert!(self.order.offset + self.order.len * size_of::<u32>() <= memory.len());
        // SAFETY: both sections lie in `memory` (checked above), do not overlap (`order` follows
        // `intervals`), and are aligned for their types (the block is `ALIGN`-aligned and the
        // offsets come from `Layout::extend`). `Interval` (four `u32`) and `u32` accept any bit
        // pattern, so the zeroed or previously written bytes are valid values.
        unsafe {
            let base = memory.as_mut_ptr();
            (
                core::slice::from_raw_parts_mut(
                    base.add(self.intervals.offset).cast::<Interval>(),
                    self.intervals.len,
                ),
                core::slice::from_raw_parts_mut(
                    base.add(self.order.offset).cast::<u32>(),
                    self.order.len,
                ),
            )
        }
    }
}

/// Pass 1 of loading: the kernel of every operator, and the size of the graph block.
struct Sizing<'k> {
    sections: BlockSections<'k>,
    persist_start: usize,
    block_size: usize,
    scratch_size: usize,
}

impl Sizing<'_> {
    fn compute(registry: &Registry<'_>, target: Target, subgraph: &SubGraph<'_>) -> Result<Self> {
        let op_count = subgraph.operator_count();
        let tensor_count = subgraph.tensor_count();
        let mut persist_total = 0usize;
        let mut scratch_size = 0usize;
        let mut io_total = 0usize;
        let mut npu_ops = 0usize;
        for index in 0..op_count {
            let mut descs = OpDescs {
                inputs: [None; MAX_OP_INPUTS],
                outputs: [NO_TENSOR; MAX_OP_OUTPUTS],
            };
            let (desc, _, _) = describe(subgraph, index, &mut descs)?;
            let (kernel, kind) = select(registry, target, &desc)?;
            npu_ops += usize::from(kind == BackendKind::Npu);
            let plan = kernel.plan(&desc)?;
            persist_total = persist_total
                .checked_add(
                    align_up(plan.persistent).ok_or_else(|| Error::out_of_memory(usize::MAX))?,
                )
                .ok_or_else(|| Error::out_of_memory(usize::MAX))?;
            scratch_size = scratch_size.max(plan.scratch);
            io_total += desc.inputs.len() + desc.outputs.len();
        }
        if target == Target::Npu && npu_ops == 0 {
            return Err(Error::unsupported(Detail::None));
        }

        let mut layout = Layout::from_size_align(0, ALIGN).map_err(|_| Error::out_of_memory(0))?;
        let ops = Section::new(&mut layout, op_count)?;
        let locations = Section::new(&mut layout, tensor_count)?;
        let io = Section::new(&mut layout, io_total)?;
        let intervals = Section::new(&mut layout, tensor_count)?;
        let order = Section::new(&mut layout, tensor_count)?;
        let persist_start =
            align_up(layout.size()).ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        let block_size = persist_start
            .checked_add(persist_total)
            .ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        Ok(Self {
            sections: BlockSections {
                ops,
                locations,
                io,
                working: WorkingSections { intervals, order },
            },
            persist_start,
            block_size,
            scratch_size,
        })
    }
}

/// Computes when each activation tensor is alive and places it in the context arena.
///
/// Fills `intervals` (one per tensor: size, first and last operator, offset; `first ==
/// u32::MAX` for tensors no operator uses) and returns the size of the activation area.
/// Checks the dataflow: every activation is produced once, before it is consumed.
fn plan_activations(
    subgraph: &SubGraph<'_>,
    intervals: &mut [Interval],
    order: &mut [u32],
) -> Result<usize> {
    let tensor_count = subgraph.tensor_count();
    let op_count = subgraph.operator_count();
    intervals.fill(Interval {
        size: 0,
        first: u32::MAX,
        last: 0,
        offset: 0,
    });
    for raw in subgraph.inputs().iter() {
        let t = tensor_index(raw, tensor_count)?.ok_or_else(malformed)?;
        intervals[t].first = 0;
    }
    for index in 0..op_count {
        let op_index = index as u32;
        let op = subgraph.operator(index)?;
        for raw in op.inputs.iter() {
            if let Some(t) = tensor_index(raw, tensor_count)? {
                if desc_is_constant(subgraph, t)? {
                    continue;
                }
                let iv = &mut intervals[t];
                if iv.first == u32::MAX {
                    // Consumed before being produced (or never produced).
                    return Err(malformed());
                }
                iv.last = iv.last.max(op_index);
            }
        }
        for raw in op.outputs.iter() {
            let t = tensor_index(raw, tensor_count)?.ok_or_else(malformed)?;
            if desc_is_constant(subgraph, t)? {
                return Err(malformed());
            }
            let iv = &mut intervals[t];
            if iv.first != u32::MAX {
                // Produced twice, or an output overwriting a graph input.
                return Err(malformed());
            }
            iv.first = op_index;
            iv.last = iv.last.max(op_index);
        }
    }
    for raw in subgraph.outputs().iter() {
        let t = tensor_index(raw, tensor_count)?.ok_or_else(malformed)?;
        if desc_is_constant(subgraph, t)? {
            continue;
        }
        let iv = &mut intervals[t];
        if iv.first == u32::MAX {
            return Err(malformed());
        }
        iv.last = op_count as u32;
    }
    for (t, iv) in intervals.iter_mut().enumerate() {
        if iv.first == u32::MAX {
            continue;
        }
        let desc = subgraph.tensor(t)?;
        if desc.data.is_some() {
            continue;
        }
        let len = desc
            .byte_len()
            .ok_or_else(|| Error::unsupported(Detail::ShapeMismatch { tensor: t as u32 }))?;
        iv.size = u32::try_from(len).map_err(|_| Error::out_of_memory(len))?;
    }
    planner::plan(intervals, order).ok_or_else(|| Error::out_of_memory(usize::MAX))
}

fn desc_is_constant(subgraph: &SubGraph<'_>, t: usize) -> Result<bool> {
    Ok(subgraph.tensor(t)?.data.is_some())
}

/// Builds the description of operator `index` in `descs`.
fn describe<'m, 'd>(
    subgraph: &SubGraph<'m>,
    index: usize,
    descs: &'d mut OpDescs<'m>,
) -> Result<(OpDesc<'m, 'd>, LeSlice<'m, i32>, LeSlice<'m, i32>)> {
    let op = subgraph.operator(index)?;
    let tensor_count = subgraph.tensor_count();
    if op.inputs.len() > MAX_OP_INPUTS || op.outputs.len() > MAX_OP_OUTPUTS {
        return Err(Error::unsupported(Detail::UnsupportedStructure));
    }
    for (slot, raw) in descs.inputs.iter_mut().zip(op.inputs.iter()) {
        *slot = match tensor_index(raw, tensor_count)? {
            Some(t) => {
                if subgraph.tensor_is_variable(t)? {
                    return Err(Error::unsupported(Detail::UnsupportedStructure));
                }
                Some(subgraph.tensor(t)?)
            }
            None => None,
        };
    }
    for (slot, raw) in descs.outputs.iter_mut().zip(op.outputs.iter()) {
        let t = tensor_index(raw, tensor_count)?.ok_or_else(malformed)?;
        *slot = subgraph.tensor(t)?;
    }
    let desc = OpDesc {
        index: index as u32,
        code: op.code,
        options: op.options,
        inputs: &descs.inputs[..op.inputs.len()],
        outputs: &descs.outputs[..op.outputs.len()],
    };
    Ok((desc, op.inputs, op.outputs))
}

fn select<'r>(
    registry: &Registry<'r>,
    target: Target,
    desc: &OpDesc<'_, '_>,
) -> Result<(&'r dyn Kernel, BackendKind)> {
    for &backend in registry.backends() {
        let kind = backend.kind();
        if target == Target::Cpu && kind == BackendKind::Npu {
            continue;
        }
        if let Some(kernel) = backend.select(desc) {
            return Ok((kernel, kind));
        }
    }
    Err(Error::unsupported(Detail::UnsupportedOp {
        op_index: desc.index,
        opcode: desc.opcode_number(),
    }))
}
