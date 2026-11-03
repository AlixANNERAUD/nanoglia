use crate::planner::{self, Interval};
use crate::{Encoding, ExecutionContext, Registry, Target};
use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use nanonn_core::backend::{BackendKind, Kernel};
use nanonn_core::le::LeSlice;
use nanonn_core::{
    ALIGN, Allocator, Block, DataType, Detail, Dims, Error, ErrorCode, MAX_OP_INPUTS,
    MAX_OP_OUTPUTS, OpDesc, Quantization, Result, TensorDesc, align_up,
};
use nanonn_tflite::{Model, SubGraph};

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
pub(crate) struct OpEntry {
    pub kernel: &'static Kernel,
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
/// The graph borrows the model bytes (`'m`): weights are never copied. It owns one block of
/// memory from `A` holding the selected kernels, their prepared parameters and the memory plan.
/// A graph is immutable: create one [`ExecutionContext`] per concurrent inference.
pub struct Graph<'m, A: Allocator> {
    model: &'m [u8],
    subgraph: SubGraph<'m>,
    block: Block<A>,
    ops: Section<OpEntry>,
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
    pub fn load_in(
        builders: &[&'m [u8]],
        encoding: Encoding,
        target: Target,
        alloc: A,
    ) -> Result<Self> {
        Self::load_with(&Registry::default(), builders, encoding, target, alloc)
    }

    /// Loads a model, picking kernels from `registry`.
    pub fn load_with(
        registry: &Registry<'_>,
        builders: &[&'m [u8]],
        encoding: Encoding,
        target: Target,
        alloc: A,
    ) -> Result<Self> {
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
        let op_count = subgraph.operator_count();
        let tensor_count = subgraph.tensor_count();
        if op_count > u32::MAX as usize || tensor_count > u32::MAX as usize {
            return Err(Error::unsupported(Detail::UnsupportedStructure));
        }

        // Pass 1: select the kernels and size everything.
        let mut persist_total = 0usize;
        let mut scratch_size = 0usize;
        let mut io_total = 0usize;
        let mut npu_ops = 0usize;
        for index in 0..op_count {
            let mut descs = OpDescs {
                inputs: [None; MAX_OP_INPUTS],
                outputs: [NO_TENSOR; MAX_OP_OUTPUTS],
            };
            let (desc, _, _) = describe(&subgraph, index, &mut descs)?;
            let (kernel, kind) = select(registry, target, &desc)?;
            npu_ops += usize::from(kind == BackendKind::Npu);
            let plan = (kernel.plan)(&desc)?;
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

        // Lay out the graph block.
        let mut layout = Layout::from_size_align(0, ALIGN).map_err(|_| Error::out_of_memory(0))?;
        let ops = Section::<OpEntry>::new(&mut layout, op_count)?;
        let locations = Section::<Location>::new(&mut layout, tensor_count)?;
        let io = Section::<i32>::new(&mut layout, io_total)?;
        let intervals = Section::<Interval>::new(&mut layout, tensor_count)?;
        let order = Section::<u32>::new(&mut layout, tensor_count)?;
        let persist = Section::<u8>::new(&mut layout, 0)?;
        let persist_start =
            align_up(persist.offset).ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        let block_size = persist_start
            .checked_add(persist_total)
            .ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        let mut block = Block::new_zeroed(block_size, alloc)?;
        let memory = block.as_mut_slice();

        // Lifetimes of the tensors. `first == u32::MAX` means "not available yet".
        for slot in intervals.uninit(memory) {
            slot.write(Interval {
                size: 0,
                first: u32::MAX,
                last: 0,
                offset: 0,
            });
        }
        for raw in subgraph.inputs().iter() {
            let t = tensor_index(raw, tensor_count)?.ok_or_else(malformed)?;
            let iv = interval_mut(&intervals, memory, t);
            iv.first = 0;
        }

        // Pass 2: prepare the kernels, fill the tables and compute lifetimes.
        let mut persist_offset = persist_start;
        let mut io_next = 0usize;
        for index in 0..op_count {
            let mut descs = OpDescs {
                inputs: [None; MAX_OP_INPUTS],
                outputs: [NO_TENSOR; MAX_OP_OUTPUTS],
            };
            let (desc, input_ids, output_ids) = describe(&subgraph, index, &mut descs)?;
            let (kernel, _) = select(registry, target, &desc)?;
            let plan = (kernel.plan)(&desc)?;
            let persist_len = plan.persistent;
            let region = memory
                .get_mut(persist_offset..persist_offset + persist_len)
                .ok_or_else(|| Error::new(ErrorCode::Unknown, Detail::None))?;
            (kernel.prepare)(&desc, region)?;

            let op = index as u32;
            for raw in input_ids.iter() {
                if let Some(t) = tensor_index(raw, tensor_count)? {
                    if desc_is_constant(&subgraph, t)? {
                        continue;
                    }
                    let iv = interval_mut(&intervals, memory, t);
                    if iv.first == u32::MAX {
                        // Consumed before being produced (or never produced).
                        return Err(malformed());
                    }
                    iv.last = iv.last.max(op);
                }
            }
            for raw in output_ids.iter() {
                let t = tensor_index(raw, tensor_count)?.ok_or_else(malformed)?;
                if desc_is_constant(&subgraph, t)? {
                    return Err(malformed());
                }
                let iv = interval_mut(&intervals, memory, t);
                if iv.first != u32::MAX {
                    // Produced twice, or an output overwriting a graph input.
                    return Err(malformed());
                }
                iv.first = op;
                iv.last = iv.last.max(op);
            }

            let io_slots = io.uninit(memory);
            let io_start = io_next;
            for raw in input_ids.iter().chain(output_ids.iter()) {
                io_slots[io_next].write(raw);
                io_next += 1;
            }
            ops.uninit(memory)[index].write(OpEntry {
                kernel,
                persist_offset: persist_offset as u32,
                persist_len: persist_len as u32,
                io_start: io_start as u32,
                inputs: input_ids.len() as u16,
                outputs: output_ids.len() as u16,
            });
            persist_offset +=
                align_up(persist_len).ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        }
        for raw in subgraph.outputs().iter() {
            let t = tensor_index(raw, tensor_count)?.ok_or_else(malformed)?;
            if desc_is_constant(&subgraph, t)? {
                continue;
            }
            let iv = interval_mut(&intervals, memory, t);
            if iv.first == u32::MAX {
                return Err(malformed());
            }
            iv.last = op_count as u32;
        }

        // Sizes and constant locations.
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
                let iv = interval_mut(&intervals, memory, t);
                if iv.first == u32::MAX {
                    Location::Unused
                } else {
                    let len = desc.byte_len().ok_or_else(|| {
                        Error::unsupported(Detail::ShapeMismatch { tensor: t as u32 })
                    })?;
                    iv.size = u32::try_from(len).map_err(|_| Error::out_of_memory(len))?;
                    Location::Arena {
                        offset: 0,
                        len: len as u32,
                    }
                }
            };
            locations.uninit(memory)[t].write(location);
        }

        // Memory plan.
        let (interval_slice, order_slice) = {
            // SAFETY: `intervals` and `order` are disjoint sections of the block, and every
            // interval was initialised above; `order` is plain `u32` working memory.
            let ivs = unsafe {
                core::slice::from_raw_parts_mut(
                    memory.as_mut_ptr().add(intervals.offset).cast::<Interval>(),
                    intervals.len,
                )
            };
            // SAFETY: as above; any bit pattern is a valid `u32`, and the block is zeroed.
            let ord = unsafe {
                core::slice::from_raw_parts_mut(
                    memory.as_mut_ptr().add(order.offset).cast::<u32>(),
                    order.len,
                )
            };
            (ivs, ord)
        };
        let activations = planner::plan(interval_slice, order_slice)
            .ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        for t in 0..tensor_count {
            let offset = interval_mut(&intervals, memory, t).offset;
            // SAFETY: every location was initialised above.
            let location = unsafe { locations.uninit(memory)[t].assume_init_mut() };
            if let Location::Arena { offset: slot, .. } = location {
                *slot = offset;
            }
        }
        let scratch_offset =
            align_up(activations).ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        let scratch_size =
            align_up(scratch_size).ok_or_else(|| Error::out_of_memory(usize::MAX))?;
        let arena_size = scratch_offset
            .checked_add(scratch_size)
            .ok_or_else(|| Error::out_of_memory(usize::MAX))?;

        Ok(Self {
            model: bytes,
            subgraph,
            block,
            ops,
            locations,
            io,
            arena_size,
            scratch_offset,
            scratch_size,
        })
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
        self.ops().get(index).map(|op| op.kernel.name)
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

    pub(crate) fn ops(&self) -> &[OpEntry] {
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

    pub(crate) fn persist(&self, op: &OpEntry) -> &[u8] {
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
impl<'m> Graph<'m, nanonn_core::allocator_api2::alloc::Global> {
    /// Loads a model with the default [`Registry`], taking its memory from the global allocator.
    pub fn load(builders: &[&'m [u8]], encoding: Encoding, target: Target) -> Result<Self> {
        Self::load_in(
            builders,
            encoding,
            target,
            nanonn_core::allocator_api2::alloc::Global,
        )
    }

    /// Creates an execution context whose arena comes from the global allocator.
    pub fn init_execution_context(
        &self,
    ) -> Result<
        ExecutionContext<
            '_,
            'm,
            nanonn_core::allocator_api2::alloc::Global,
            nanonn_core::allocator_api2::alloc::Global,
        >,
    > {
        self.init_execution_context_in(nanonn_core::allocator_api2::alloc::Global)
    }
}

fn interval_mut<'b>(
    section: &Section<Interval>,
    memory: &'b mut [u8],
    t: usize,
) -> &'b mut Interval {
    // SAFETY: every interval was initialised at the start of `load`.
    unsafe { section.uninit(memory)[t].assume_init_mut() }
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

fn select(
    registry: &Registry<'_>,
    target: Target,
    desc: &OpDesc<'_, '_>,
) -> Result<(&'static Kernel, BackendKind)> {
    for backend in registry.backends() {
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
