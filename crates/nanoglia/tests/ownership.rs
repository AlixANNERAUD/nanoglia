//! Owning contexts retain their model and graph without allocating during inference.
use nanoglia::core::{Block, allocator_api2::alloc::Global};
use nanoglia::{Encoding, Graph, GraphExecutionContext, OwnedGraph, Target};
use std::sync::Arc;
mod common;

#[test]
fn heap_model_survives_device_owner_and_contexts_are_independent() {
    let source = common::Aligned::read(
        &common::repo_root().join("testdata/mlperf-tiny/kws_ref_model.tflite"),
    );
    let mut model = Block::new_zeroed(source.bytes().len(), Global).unwrap();
    model.as_mut_slice().copy_from_slice(source.bytes());
    let graph =
        Arc::new(OwnedGraph::load_in(model, Encoding::Autodetect, Target::Cpu, Global).unwrap());
    drop(source);
    let weak = Arc::downgrade(&graph);
    let mut first = GraphExecutionContext::new_in(graph.clone(), Global).unwrap();
    let mut second = GraphExecutionContext::new_in(graph.clone(), Global).unwrap();
    drop(graph);
    for value in [3, -2, 5] {
        first
            .input_mut(0)
            .unwrap()
            .as_mut_slice::<i8>()
            .unwrap()
            .fill(value);
        second
            .input_mut(0)
            .unwrap()
            .as_mut_slice::<i8>()
            .unwrap()
            .fill(value);
        first.compute().unwrap();
        second.compute().unwrap();
        assert_eq!(
            first.output(0).unwrap().bytes(),
            second.output(0).unwrap().bytes()
        );
    }
    drop(first);
    assert!(weak.upgrade().is_some());
    drop(second);
    assert!(weak.upgrade().is_none());
}

#[test]
fn owning_a_borrowed_rom_graph_preserves_the_borrowed_api() {
    #[repr(align(16))]
    struct Aligned<const N: usize>([u8; N]);
    static MODEL: Aligned<{ include_bytes!("../doc/classifier.tflite").len() }> =
        Aligned(*include_bytes!("../doc/classifier.tflite"));
    let graph = Arc::new(Graph::load(&[&MODEL.0], Encoding::Autodetect, Target::Cpu).unwrap());
    let mut context = GraphExecutionContext::new_in(graph, Global).unwrap();
    context.input_mut(0).unwrap().bytes_mut().fill(0);
    context.compute().unwrap();
    assert!(!context.output(0).unwrap().bytes().is_empty());
}

#[test]
fn owned_graphs_and_their_contexts_can_move_between_threads() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<OwnedGraph<Global>>();
    send_sync::<GraphExecutionContext<Arc<OwnedGraph<Global>>, Global>>();
}
