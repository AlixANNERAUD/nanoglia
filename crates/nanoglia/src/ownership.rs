//! Owning graph storage and graph handles for persistent execution contexts.
use crate::{Encoding, Graph, Target};
use nanoglia_core::{Allocator, Block, Result};

/// A stable graph reference or owner retained by an execution context.
///
/// Implemented for graphs, borrowed graphs, owned models, and (`alloc`, pointer atomics)
/// shared owners. Custom OS shared pointers can implement this without a dependency on an OS.
///
/// # Safety
/// Every call must return the same logical immutable graph, model and memory plan for the
/// entire lifetime of the handle. The graph and all borrowed storage must remain valid for
/// each returned borrow. Switching graphs would invalidate the execution arena's bounds
/// and aliasing guarantees, even when both graphs have the same arena size.
pub unsafe trait GraphHandle {
    /// The allocator of the prepared graph.
    type Allocator: Allocator;

    /// Borrows the same prepared graph for the lifetime of this handle.
    fn graph(&self) -> &Graph<'_, Self::Allocator>;
}

// SAFETY: direct access always borrows this same immutable graph.
unsafe impl<A: Allocator> GraphHandle for Graph<'_, A> {
    type Allocator = A;

    fn graph(&self) -> &Graph<'_, A> {
        self
    }
}
// SAFETY: direct access always borrows this same immutable graph.
unsafe impl<A: Allocator> GraphHandle for &Graph<'_, A> {
    type Allocator = A;

    fn graph(&self) -> &Graph<'_, A> {
        self
    }
}

/// A prepared graph owning an aligned model allocation, without copying its weights.
///
/// Both allocations use caller-provided allocators. No additional metadata allocation is
/// required. Custom borrowed registries continue to use `Graph::load_with`.
///
/// `OwnedGraph` is `Send` and `Sync` when both allocators are, so an `Arc<OwnedGraph>` can be
/// shared between tasks or threads.
///
/// Model views borrow this owner and cannot outlive it:
/// ```compile_fail
/// # use nanoglia::{OwnedGraph, Encoding, Target};
/// # use nanoglia::core::{Block, allocator_api2::alloc::Global};
/// let graph = OwnedGraph::load_in(Block::new_zeroed(32, Global).unwrap(),
///     Encoding::Autodetect, Target::Cpu, Global).unwrap();
/// let input = graph.graph().input(0).unwrap();
/// drop(graph);
/// println!("{:?}", input);
/// ```
pub struct OwnedGraph<A: Allocator, M: Allocator = A> {
    // Fields drop in declaration order: prepared kernel data first, model bytes last.
    graph: Graph<'static, A>,
    model: Block<M>,
}

impl<A: Allocator, M: Allocator> OwnedGraph<A, M> {
    /// Takes an already aligned model allocation, verifies it and prepares its graph.
    /// On error the model is released. Uses only the default, static kernel registry.
    pub fn load_in(model: Block<M>, encoding: Encoding, target: Target, alloc: A) -> Result<Self> {
        // SAFETY: Block owns a stable allocation. It is immutable for the graph's lifetime,
        // and is dropped after the graph. The extended lifetime is private: graph() shortens
        // BOTH the model lifetime and the graph reference lifetime to a borrow of this owner.
        // Default registry kernels are static; no borrowed registry lifetime is extended.
        let bytes = unsafe { core::slice::from_raw_parts(model.as_slice().as_ptr(), model.len()) };
        let graph = Graph::load_in(&[bytes], encoding, target, alloc)?;
        Ok(Self { graph, model })
    }

    /// Borrows the prepared graph without exposing the internal model lifetime.
    pub fn graph(&self) -> &Graph<'_, A> {
        &self.graph
    }

    /// Size of the retained model allocation in bytes.
    pub fn model_bytes(&self) -> usize {
        self.model.len()
    }
}
// SAFETY: the owned model and prepared graph are immutable until this owner is dropped.
unsafe impl<A: Allocator, M: Allocator> GraphHandle for OwnedGraph<A, M> {
    type Allocator = A;

    fn graph(&self) -> &Graph<'_, A> {
        self.graph()
    }
}

#[cfg(all(feature = "alloc", target_has_atomic = "ptr"))]
mod shared {
    extern crate alloc;
    use super::*;
    use alloc::sync::Arc;
    // SAFETY: Arc retains the same immutable GraphHandle and cannot replace its value.
    unsafe impl<G: GraphHandle> GraphHandle for Arc<G> {
        type Allocator = G::Allocator;

        fn graph(&self) -> &Graph<'_, G::Allocator> {
            self.as_ref().graph()
        }
    }
}
