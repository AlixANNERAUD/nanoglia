//! The standard traits of the public types.

use nanoglia::core::allocator_api2::alloc::Global;
use nanoglia::core::backend::{BackendKind, KernelPlan};
use nanoglia::{Encoding, Graph, MemoryRequirements, Target};
use std::collections::HashSet;

mod common;

fn kws() -> common::Aligned {
    common::Aligned::read(&common::repo_root().join("testdata/mlperf-tiny/kws_ref_model.tflite"))
}

fn fill(mut tensor: impl AsMut<[u8]>, value: u8) {
    tensor.as_mut().fill(value);
}

fn sum(tensor: impl AsRef<[u8]>) -> u64 {
    tensor.as_ref().iter().map(|&b| u64::from(b)).sum()
}

#[test]
fn tensors_are_byte_buffers() {
    let model = kws();
    let graph =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
    let mut ctx = graph.init_execution_context_in(Global).unwrap();
    fill(ctx.input_mut(0).unwrap(), 2);
    assert_eq!(sum(ctx.input_mut(0).unwrap()), 2 * 490);
    ctx.compute().unwrap();
    let output = ctx.output(0).unwrap();
    assert_eq!(output.as_ref(), output.bytes());
}

#[test]
fn inputs_and_outputs_iterate_over_the_descriptions() {
    let model = kws();
    let graph =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
    let inputs: Vec<_> = graph.inputs().collect();
    assert_eq!(inputs.len(), graph.input_count());
    assert_eq!(inputs[0].index, graph.input(0).unwrap().index);
    let outputs: Vec<_> = graph.outputs().collect();
    assert_eq!(outputs.len(), graph.output_count());
    assert_eq!(outputs[0].dims.as_slice(), &[1, 12]);
}

#[test]
fn plain_values_hash() {
    let set: HashSet<_> = [Encoding::TensorFlowLite, Encoding::Autodetect].into();
    assert_eq!(set.len(), 2);
    let set: HashSet<_> = [Target::Cpu, Target::Npu, Target::Auto, Target::default()].into();
    assert_eq!(set.len(), 3);
    let set: HashSet<_> = [BackendKind::Cpu, BackendKind::Npu].into();
    assert_eq!(set.len(), 2);
    let set: HashSet<_> = [KernelPlan::default(), KernelPlan::default()].into();
    assert_eq!(set.len(), 1);

    let model = kws();
    let needs = MemoryRequirements::compute_in(
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        Global,
    )
    .unwrap();
    let set: HashSet<_> = [needs, needs].into();
    assert_eq!(set.len(), 1);
}
