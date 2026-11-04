//! Checks the memory contract: every byte comes from the caller's allocator, a graph makes one
//! allocation, a context one, `compute` none, and a static arena works without any heap.

use core::alloc::Layout;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};
use nanoglia::core::allocator_api2::alloc::{AllocError, Allocator, Global};
use nanoglia::{Arena, Encoding, Graph, MemoryRequirements, Target};

mod common;

#[derive(Default)]
struct Counting {
    allocations: AtomicUsize,
    live_bytes: AtomicUsize,
}

// SAFETY: forwards to the global allocator.
unsafe impl Allocator for Counting {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocations.fetch_add(1, Ordering::SeqCst);
        self.live_bytes.fetch_add(layout.size(), Ordering::SeqCst);
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.live_bytes.fetch_sub(layout.size(), Ordering::SeqCst);
        // SAFETY: allocated by `Global` with this layout in `allocate`.
        unsafe { Global.deallocate(ptr, layout) }
    }
}

fn kws() -> common::Aligned {
    let path = common::repo_root().join("testdata/mlperf-tiny/kws_ref_model.tflite");
    common::Aligned::read(&path)
}

#[test]
fn one_allocation_per_graph_and_context_none_in_compute() {
    let model = kws();
    let counting = Counting::default();
    let graph = Graph::load_in(
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        &counting,
    )
    .unwrap();
    assert_eq!(counting.allocations.load(Ordering::SeqCst), 1);
    assert_eq!(
        counting.live_bytes.load(Ordering::SeqCst),
        graph.graph_bytes()
    );

    let mut ctx = graph.init_execution_context_in(&counting).unwrap();
    assert_eq!(counting.allocations.load(Ordering::SeqCst), 2);

    ctx.input_mut(0)
        .unwrap()
        .as_mut_slice::<i8>()
        .unwrap()
        .fill(3);
    ctx.compute().unwrap();
    ctx.compute().unwrap();
    assert_eq!(counting.allocations.load(Ordering::SeqCst), 2);
    assert_eq!(
        counting.live_bytes.load(Ordering::SeqCst),
        graph.graph_bytes() + graph.context_bytes()
    );

    drop(ctx);
    drop(graph);
    assert_eq!(counting.live_bytes.load(Ordering::SeqCst), 0);
}

#[test]
fn static_arena_without_heap() {
    let model = kws();
    let mut memory = vec![0u8; 256 * 1024];
    let arena = Arena::new(&mut memory);
    let graph =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, &arena).unwrap();
    let mut ctx = graph.init_execution_context_in(&arena).unwrap();
    ctx.input_mut(0)
        .unwrap()
        .as_mut_slice::<i8>()
        .unwrap()
        .fill(-7);
    ctx.compute().unwrap();
    let out = ctx.output(0).unwrap();
    assert_eq!(out.as_slice::<i8>().unwrap().len(), 12);
    println!(
        "kws: graph {} B, context {} B, arena used {} B",
        graph.graph_bytes(),
        graph.context_bytes(),
        arena.used()
    );
}

#[test]
fn too_small_arena_is_an_error() {
    let model = kws();
    let mut memory = vec![0u8; 512];
    let arena = Arena::new(&mut memory);
    let err =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, &arena).unwrap_err();
    assert_eq!(err.code, nanoglia::ErrorCode::TooLarge);
}

#[test]
fn several_contexts_share_one_graph() {
    let model = kws();
    let graph =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
    let mut a = graph.init_execution_context_in(Global).unwrap();
    let mut b = graph.init_execution_context_in(Global).unwrap();
    a.input_mut(0)
        .unwrap()
        .as_mut_slice::<i8>()
        .unwrap()
        .fill(10);
    b.input_mut(0)
        .unwrap()
        .as_mut_slice::<i8>()
        .unwrap()
        .fill(10);
    a.compute().unwrap();
    b.compute().unwrap();
    assert_eq!(a.output(0).unwrap().bytes(), b.output(0).unwrap().bytes());
}

fn mlperf(name: &str) -> common::Aligned {
    let path = common::repo_root().join(format!("testdata/mlperf-tiny/{name}.tflite"));
    common::Aligned::read(&path)
}

const MLPERF: &[&str] = &[
    "kws_ref_model",
    "vww_96_int8",
    "pretrainedResnet_quant",
    "ad01_int8",
];

#[test]
fn requirements_match_the_loaded_graph() {
    for name in MLPERF {
        let model = mlperf(name);
        let needs = MemoryRequirements::compute_in(
            &[model.bytes()],
            Encoding::Autodetect,
            Target::Auto,
            Global,
        )
        .unwrap();
        let graph =
            Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
        assert_eq!(graph.memory_requirements(), needs, "{name}");
        println!(
            "{name}: graph {} B, context {} B, arena for one context {} B",
            needs.graph_bytes,
            needs.context_bytes,
            needs.arena_bytes(1)
        );
    }
}

#[test]
fn requirements_only_borrow_working_memory() {
    let model = kws();
    let counting = Counting::default();
    MemoryRequirements::compute_in(
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        &counting,
    )
    .unwrap();
    assert_eq!(counting.allocations.load(Ordering::SeqCst), 1);
    assert_eq!(counting.live_bytes.load(Ordering::SeqCst), 0);
}

#[test]
fn arena_of_the_required_size_is_enough_at_any_alignment() {
    for name in MLPERF {
        let model = mlperf(name);
        let needs = MemoryRequirements::compute_in(
            &[model.bytes()],
            Encoding::Autodetect,
            Target::Auto,
            Global,
        )
        .unwrap();
        for contexts in [1, 2] {
            let size = needs.arena_bytes(contexts);
            let mut memory = vec![0u8; size + 16];
            for skew in 0..16 {
                let arena = Arena::new(&mut memory[skew..skew + size]);
                let graph =
                    Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, &arena)
                        .unwrap_or_else(|e| panic!("{name}, skew {skew}: {e}"));
                let mut first = graph.init_execution_context_in(&arena).unwrap();
                if contexts == 2 {
                    let _second = graph.init_execution_context_in(&arena).unwrap();
                }
                first.compute().unwrap();
            }
        }
    }
}
