//! `ExecutionContext::compute_with`: what an observer sees of each operator.

use core::ops::ControlFlow;
use nanoglia::core::allocator_api2::alloc::Global;
use nanoglia::core::backend::{Backend, FnKernel, Kernel, KernelPlan};
use nanoglia::core::{BuiltinOp, OpDesc};
use nanoglia::{
    DataType, Detail, Encoding, Error, ErrorCode, Graph, Observer, OpCode, OpInfo, Registry,
    Result, Target,
};

mod common;

fn kws() -> common::Aligned {
    common::Aligned::read(&common::repo_root().join("testdata/mlperf-tiny/kws_ref_model.tflite"))
}

fn input() -> Vec<u8> {
    (0..490).map(|i| (i * 13 % 251) as u8).collect()
}

/// Records every call, in order.
#[derive(Default)]
struct Trace {
    calls: Vec<(&'static str, usize, &'static str)>,
    results: Vec<Result<()>>,
    opcodes: Vec<i32>,
}

impl Observer for Trace {
    fn before(&mut self, op: &OpInfo<'_>) -> ControlFlow<()> {
        self.calls.push(("before", op.index(), op.kernel_name()));
        self.opcodes.push(match op.opcode().unwrap() {
            OpCode::Builtin(code) => code.0,
            OpCode::Custom(_) => -1,
        });
        ControlFlow::Continue(())
    }

    fn after(&mut self, op: &OpInfo<'_>, result: &Result<()>) {
        self.calls.push(("after", op.index(), op.kernel_name()));
        self.results.push(*result);
    }
}

#[test]
fn the_observer_sees_every_operator_in_order() {
    let model = kws();
    let graph =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
    let mut ctx = graph.init_execution_context_in(Global).unwrap();
    ctx.set_input(0, &input()).unwrap();
    let mut trace = Trace::default();
    ctx.compute_with(&mut trace).unwrap();
    let observed = ctx.output(0).unwrap().bytes().to_vec();

    let n = graph.operator_count();
    assert_eq!(trace.calls.len(), 2 * n);
    for i in 0..n {
        let name = graph.kernel_name(i).unwrap();
        assert_eq!(trace.calls[2 * i], ("before", i, name));
        assert_eq!(trace.calls[2 * i + 1], ("after", i, name));
    }
    assert!(trace.results.iter().all(Result::is_ok));
    // DS-CNN: a convolution first, a softmax last.
    assert_eq!(trace.opcodes[0], BuiltinOp::CONV_2D.0);
    assert_eq!(trace.opcodes[n - 1], BuiltinOp::SOFTMAX.0);

    // Observing changes nothing.
    ctx.set_input(0, &input()).unwrap();
    ctx.compute().unwrap();
    assert_eq!(ctx.output(0).unwrap().bytes(), observed);
}

/// Copies the tensors around each operator.
#[derive(Default)]
struct Capture {
    /// Input 0 of each operator, before it runs.
    inputs: Vec<Vec<u8>>,
    /// Output 0 of each operator, after it ran.
    outputs: Vec<Vec<u8>>,
    output_types: Vec<DataType>,
}

impl Observer for Capture {
    fn before(&mut self, op: &OpInfo<'_>) -> ControlFlow<()> {
        assert_eq!(
            op.input(op.input_count()).unwrap_err().detail,
            Detail::IndexOutOfRange {
                index: op.input_count() as u32
            }
        );
        self.inputs.push(op.input(0).unwrap().bytes().to_vec());
        ControlFlow::Continue(())
    }

    fn after(&mut self, op: &OpInfo<'_>, _result: &Result<()>) {
        assert_eq!(op.output_count(), 1);
        let output = op.output(0).unwrap();
        self.output_types.push(output.dtype());
        self.outputs.push(output.bytes().to_vec());
    }
}

#[test]
fn the_observer_reads_intermediate_tensors() {
    let model = kws();
    let graph =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
    let mut ctx = graph.init_execution_context_in(Global).unwrap();
    ctx.set_input(0, &input()).unwrap();
    let mut capture = Capture::default();
    ctx.compute_with(&mut capture).unwrap();

    let n = graph.operator_count();
    assert_eq!(capture.inputs[0], input());
    // DS-CNN is a chain: what an operator computed is what the next one reads.
    for i in 0..n - 1 {
        assert_eq!(capture.outputs[i], capture.inputs[i + 1], "operator {i}");
        assert!(!capture.outputs[i].is_empty());
    }
    assert_eq!(capture.outputs[n - 1], ctx.output(0).unwrap().bytes());
    assert!(capture.output_types.iter().all(|&t| t == DataType::I8));
}

static FAILING: FnKernel = FnKernel {
    name: "test/failing",
    plan: |_| Ok(KernelPlan::default()),
    prepare: |_, _| Ok(()),
    eval: |_| Err(Error::new(ErrorCode::RuntimeError, Detail::None)),
};

/// Claims the softmax, with a kernel that always fails.
struct FailingSoftmax;

impl Backend for FailingSoftmax {
    fn name(&self) -> &'static str {
        "failing"
    }

    fn select<'s>(&'s self, op: &OpDesc<'_, '_>) -> Option<&'s dyn Kernel> {
        (op.builtin()? == BuiltinOp::SOFTMAX).then_some(&FAILING)
    }
}

#[test]
fn the_observer_sees_a_kernel_error_then_compute_returns_it() {
    let model = kws();
    let mut backends: Vec<&dyn Backend> = vec![&FailingSoftmax];
    backends.extend(Registry::default().backends());
    let graph = Graph::load_with(
        &Registry::new(&backends),
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        Global,
    )
    .unwrap();
    let mut ctx = graph.init_execution_context_in(Global).unwrap();
    ctx.set_input(0, &input()).unwrap();
    let mut trace = Trace::default();
    let error = ctx.compute_with(&mut trace).unwrap_err();
    assert_eq!(error.code, ErrorCode::RuntimeError);

    let n = graph.operator_count();
    assert_eq!(trace.calls.len(), 2 * n);
    assert_eq!(trace.calls[2 * n - 1], ("after", n - 1, "test/failing"));
    assert_eq!(trace.results[n - 1], Err(error));
    assert!(trace.results[..n - 1].iter().all(Result::is_ok));
}

#[test]
fn observers_can_be_trait_objects() {
    let model = kws();
    let graph =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
    let mut ctx = graph.init_execution_context_in(Global).unwrap();
    ctx.set_input(0, &input()).unwrap();
    let mut trace = Trace::default();
    let observer: &mut dyn Observer = &mut trace;
    ctx.compute_with(observer).unwrap();
    assert_eq!(trace.calls.len(), 2 * graph.operator_count());
}
