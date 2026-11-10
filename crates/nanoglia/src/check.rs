//! Checking whether a model runs, operator by operator, without loading it.

use crate::graph::{self, WorkingSections};
use crate::{Encoding, MemoryBreakdown, Registry, Target};
use core::alloc::Layout;
use nanoglia_core::backend::{BackendKind, KernelPlan};
use nanoglia_core::{ALIGN, Allocator, Block, Detail, Error, OpCode, Result, TensorDesc, align_up};
use nanoglia_tflite::{OperatorHeader, SubGraph};

/// The kernel selected for an operator, in an [`OperatorReport`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct SelectedKernel {
    /// Name of the kernel, as [`Graph::kernel_name`](crate::Graph::kernel_name) reports it (for
    /// example `"cmsis-nn/conv2d_s8"`).
    pub name: &'static str,
    /// Name of the backend that lends the kernel.
    pub backend: &'static str,
    /// Where the backend computes.
    pub kind: BackendKind,
    /// The memory the kernel asks for this operator: persistent parameters (in the graph block)
    /// and scratch (in each execution context, shared by all operators).
    pub plan: KernelPlan,
}

/// What [`ModelReport::check_with`] found for one operator of a model.
///
/// It describes the operator from the model alone (code, version, tensors), so it exists even
/// for an operator nanoglia cannot decode or run; [`Self::kernel`] then holds the reason.
///
/// It borrows the model bytes (`'a`), not the check: the operator code and the tensor
/// descriptions it returns can be kept after the closure returns.
#[derive(Clone, Copy)]
pub struct OperatorReport<'a> {
    subgraph: SubGraph<'a>,
    index: usize,
    header: OperatorHeader<'a>,
    kernel: Result<SelectedKernel>,
}

impl core::fmt::Debug for OperatorReport<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OperatorReport")
            .field("index", &self.index)
            .field("opcode", &self.header.code)
            .field("version", &self.header.version)
            .field("kernel", &self.kernel)
            .finish()
    }
}

impl<'a> OperatorReport<'a> {
    /// Index of the operator in the graph (execution order).
    pub fn index(&self) -> usize {
        self.index
    }

    /// The operator code. [`BuiltinOp::name`](nanoglia_core::BuiltinOp::name) gives the name of
    /// a builtin operator.
    pub fn opcode(&self) -> OpCode<'a> {
        self.header.code
    }

    /// The version of the operator in the model.
    pub fn version(&self) -> i32 {
        self.header.version
    }

    /// Number of inputs, omitted optional ones included.
    pub fn input_count(&self) -> usize {
        self.header.inputs.len()
    }

    /// Number of outputs.
    pub fn output_count(&self) -> usize {
        self.header.outputs.len()
    }

    /// Description of input `i` (type, shape, quantization, constant data); `None` for an omitted
    /// optional input.
    ///
    /// # Errors
    ///
    /// [`Detail::IndexOutOfRange`] if `i >= self.input_count()`, and
    /// [`Detail::UnsupportedType`] for a tensor nanoglia cannot represent (see
    /// [`SubGraph::tensor`]).
    pub fn input(&self, i: usize) -> Result<Option<TensorDesc<'a>>> {
        self.tensor(self.header.inputs.get(i), i)
    }

    /// Description of output `i`.
    ///
    /// # Errors
    ///
    /// As [`Self::input`].
    pub fn output(&self, i: usize) -> Result<TensorDesc<'a>> {
        self.tensor(self.header.outputs.get(i), i)?
            .ok_or_else(|| Error::malformed(0))
    }

    /// The kernel that runs the operator, or why none does: the error
    /// [`Graph::load_with`](crate::Graph::load_with) returns for this operator.
    ///
    /// The usual reasons are [`Detail::UnsupportedOp`] (no backend of the registry claims the
    /// operator with these types, shapes and options, or its options cannot be decoded),
    /// [`Detail::UnsupportedType`] (one of its tensors has a type nanoglia cannot represent) and
    /// [`Detail::UnsupportedStructure`] (a variable tensor, too many inputs or outputs); a kernel
    /// may also refuse the operator while planning or preparing it.
    pub fn kernel(&self) -> Result<SelectedKernel> {
        self.kernel
    }

    fn tensor(&self, raw: Option<i32>, i: usize) -> Result<Option<TensorDesc<'a>>> {
        let raw = raw.ok_or_else(|| {
            Error::invalid_argument(Detail::IndexOutOfRange {
                index: u32::try_from(i).unwrap_or(u32::MAX),
            })
        })?;
        match graph::tensor_index(raw, self.subgraph.tensor_count())? {
            Some(t) => self.subgraph.tensor(t).map(Some),
            None => Ok(None),
        }
    }
}

/// Whether a model runs, and what it needs: the answer to "will `Graph::load` accept this model?"
/// with every operator that stands in the way, where loading stops at the first.
///
/// Checking walks the operators of the model and hands an [`OperatorReport`] for each to a
/// closure: its code, version and tensors, and the kernel selected for it or the reason none is.
/// The kernels are selected, planned and prepared exactly as loading does, with the same
/// [`Registry`] and [`Target`], so the verdict depends on the backends and Cargo features of the
/// program that checks: check with those of your firmware.
///
/// ```
/// # #[repr(C, align(16))]
/// # struct Aligned<T: ?Sized>(T);
/// # static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
/// #     env!("CARGO_MANIFEST_DIR"),
/// #     "/doc/classifier.tflite"
/// # )));
/// # fn main() -> Result<(), nanoglia::Error> {
/// use nanoglia::{Arena, BuiltinOp, Encoding, ModelReport, OpCode, Registry, Target};
///
/// // Working memory only, given back before returning.
/// let mut memory = [0u8; 1024];
/// let arena = Arena::new(&mut memory);
///
/// let report = ModelReport::check_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena, |op| {
///     assert_eq!(op.opcode(), OpCode::Builtin(BuiltinOp::FULLY_CONNECTED));
///     assert_eq!(op.kernel().unwrap().name, "cmsis-nn/fully_connected_s8");
/// })?;
/// assert!(report.runs());
/// let memory = report.memory?;
/// assert_eq!(memory.context_bytes, memory.activation_bytes + memory.scratch_bytes);
///
/// // Without any backend, the same model does not run: its only operator has no kernel.
/// let empty = Registry::new(&[]);
/// let mut blocking = 0;
/// let report = ModelReport::check_with(
///     &empty,
///     &[&MODEL.0],
///     Encoding::Autodetect,
///     Target::Auto,
///     &arena,
///     |op| blocking += usize::from(op.kernel().is_err()),
/// )?;
/// assert!(!report.runs());
/// assert_eq!((report.operators, report.unsupported, blocking), (1, 1, 1));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ModelReport {
    /// Number of operators of the model.
    pub operators: usize,
    /// Number of operators without a kernel: those whose [`OperatorReport::kernel`] is an error.
    pub unsupported: usize,
    /// What loading the model allocates, or the error [`Graph::load_with`](crate::Graph::load_with)
    /// returns: the one of the first operator no backend runs, or a reason that belongs to the
    /// model as a whole (a misaligned constant, a tensor consumed before it is produced,
    /// [`Target::Npu`] without any operator on an accelerator...).
    pub memory: Result<MemoryBreakdown>,
}

impl ModelReport {
    /// Whether the model loads: every operator has a kernel and the model can be planned. An
    /// allocator that gives [`MemoryBreakdown::graph_bytes`] then
    /// [`MemoryBreakdown::context_bytes`] bytes is all it still needs.
    pub fn runs(&self) -> bool {
        self.memory.is_ok()
    }

    /// Checks a model with the default [`Registry`] (the backends enabled by the crate features),
    /// calling `visit` once per operator, in execution order.
    ///
    /// `alloc` only lends working memory, freed before returning: one block of 20 bytes per
    /// tensor plus the largest prepared parameters of an operator. Nothing else is allocated.
    ///
    /// # Errors
    ///
    /// When the operators cannot be walked at all: the bytes are not a TFLite model
    /// ([`Detail::UnknownEncoding`], [`Detail::MalformedModel`], [`Detail::UnsupportedSchema`]),
    /// the model has several subgraphs ([`Detail::UnsupportedStructure`]), or `alloc` cannot
    /// lend the working memory ([`Detail::OutOfMemory`]). `visit` is not called then. A model
    /// that parses but does not run is not an error: see [`Self::memory`].
    pub fn check_in<'m, A: Allocator, F: FnMut(&OperatorReport<'m>)>(
        builders: &[&'m [u8]],
        encoding: Encoding,
        target: Target,
        alloc: A,
        visit: F,
    ) -> Result<Self> {
        Self::check_with(
            &Registry::default(),
            builders,
            encoding,
            target,
            alloc,
            visit,
        )
    }

    /// Checks a model, picking kernels from `registry`: see [`Self::check_in`].
    pub fn check_with<'m, A: Allocator, F: FnMut(&OperatorReport<'m>)>(
        registry: &Registry<'_>,
        builders: &[&'m [u8]],
        encoding: Encoding,
        target: Target,
        alloc: A,
        mut visit: F,
    ) -> Result<Self> {
        let (_, subgraph) = graph::open(builders, encoding)?;
        let operators = subgraph.operator_count();

        // The largest prepared parameters, to prepare each kernel in turn in one buffer.
        let mut persistent = 0usize;
        for index in 0..operators {
            if let Ok(choice) = graph::choose(registry, target, &subgraph, index, None) {
                persistent = persistent.max(choice.plan.persistent);
            }
        }

        // One block: the memory planner's working memory, then that buffer.
        let overflow = || Error::out_of_memory(usize::MAX);
        let mut layout = Layout::from_size_align(0, ALIGN).map_err(|_| overflow())?;
        let working = WorkingSections::new(&mut layout, subgraph.tensor_count())?;
        let persist_start = align_up(layout.size()).ok_or_else(overflow)?;
        let size = persist_start.checked_add(persistent).ok_or_else(overflow)?;
        let mut block = Block::new_zeroed(size, alloc)?;
        let (planning, persist) = block
            .as_mut_slice()
            .split_at_mut_checked(persist_start)
            .ok_or_else(overflow)?;

        let mut unsupported = 0usize;
        let mut first_error = None;
        for index in 0..operators {
            let header = subgraph.operator_header(index)?;
            let kernel = graph::choose(registry, target, &subgraph, index, Some(&mut *persist))
                .map(|choice| SelectedKernel {
                    name: choice.kernel.name(),
                    backend: choice.backend,
                    kind: choice.kind,
                    plan: choice.plan,
                });
            if let Err(error) = kernel {
                unsupported += 1;
                first_error.get_or_insert(error);
            }
            visit(&OperatorReport {
                subgraph,
                index,
                header,
                kernel,
            });
        }

        // The sizes, computed as loading does; then what only preparing a kernel finds.
        let memory = graph::measure(registry, target, &subgraph, &working, planning)
            .and_then(|memory| first_error.map_or(Ok(memory), Err));
        Ok(Self {
            operators,
            unsupported,
            memory,
        })
    }

    /// Checks a model with the default [`Registry`], taking the working memory from the global
    /// allocator: see [`Self::check_in`].
    #[cfg(feature = "alloc")]
    pub fn check<'m, F: FnMut(&OperatorReport<'m>)>(
        builders: &[&'m [u8]],
        encoding: Encoding,
        target: Target,
        visit: F,
    ) -> Result<Self> {
        Self::check_in(
            builders,
            encoding,
            target,
            nanoglia_core::allocator_api2::alloc::Global,
            visit,
        )
    }
}
