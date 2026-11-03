use crate::graph::{Graph, Location};
use crate::{TensorMut, TensorRef};
use nanonn_core::backend::EvalCtx;
use nanonn_core::{Allocator, Block, Detail, Error, MAX_OP_INPUTS, MAX_OP_OUTPUTS, Result};

/// The mutable state of one inference: the arena holding the activations and the scratch memory.
///
/// Write the inputs ([`Self::set_input`] or [`Self::input_mut`]) before every
/// [`Self::compute`]: the memory plan reuses input memory once it has been consumed. Outputs
/// stay valid until the next call that borrows the context mutably.
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
    /// tensor size exactly.
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
    pub fn compute(&mut self) -> Result<()> {
        let graph = self.graph;
        let model = graph.model();
        let io = graph.io_table();
        let (scratch_offset, scratch_len) = graph.scratch();
        let arena_len = self.arena.len();
        let base = self.arena.as_mut_ptr();

        for op in graph.ops() {
            let start = op.io_start as usize;
            let inputs = &io[start..start + op.inputs as usize];
            let outputs =
                &io[start + op.inputs as usize..start + op.inputs as usize + op.outputs as usize];

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
            (op.kernel.eval)(&mut ctx)?;
        }
        Ok(())
    }
}
