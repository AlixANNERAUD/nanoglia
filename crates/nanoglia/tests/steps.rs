//! Running an inference in slices: `step`, `cancel`, `compute_async`, and an observer that
//! stops the computation.

use core::ops::ControlFlow;
use nanoglia::core::allocator_api2::alloc::Global;
use nanoglia::{Detail, Encoding, ErrorCode, Graph, Observer, OpInfo, Step, Target};
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

mod common;

fn kws() -> common::Aligned {
    common::Aligned::read(&common::repo_root().join("testdata/mlperf-tiny/kws_ref_model.tflite"))
}

fn input(seed: u32) -> Vec<u8> {
    (0..490u32)
        .map(|i| (i.wrapping_mul(seed) % 251) as u8)
        .collect()
}

/// Polls a future to completion with a waker that does nothing, counting the polls.
fn block_on<F: Future>(future: F) -> (F::Output, usize) {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    let mut polls = 0;
    loop {
        polls += 1;
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return (output, polls);
        }
    }
}

#[test]
fn stepping_gives_the_same_outputs_as_compute() {
    let model = kws();
    let graph =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
    let mut ctx = graph.init_execution_context_in(Global).unwrap();

    ctx.set_input(0, &input(13)).unwrap();
    ctx.compute().unwrap();
    let expected = ctx.output(0).unwrap().bytes().to_vec();

    ctx.set_input(0, &input(13)).unwrap();
    let mut steps = 0;
    assert_eq!(ctx.next_operator(), None);
    while ctx.step().unwrap() == Step::Pending {
        steps += 1;
        assert_eq!(ctx.next_operator(), Some(steps));
    }
    assert_eq!(steps + 1, graph.operator_count());
    assert_eq!(ctx.next_operator(), None);
    assert_eq!(ctx.output(0).unwrap().bytes(), &expected[..]);
}

#[test]
fn cancel_starts_over() {
    let model = kws();
    let graph =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
    let mut ctx = graph.init_execution_context_in(Global).unwrap();

    ctx.set_input(0, &input(7)).unwrap();
    ctx.compute().unwrap();
    let expected = ctx.output(0).unwrap().bytes().to_vec();

    // Half an inference on other inputs, abandoned.
    ctx.set_input(0, &input(29)).unwrap();
    for _ in 0..5 {
        assert_eq!(ctx.step().unwrap(), Step::Pending);
    }
    ctx.cancel();
    assert_eq!(ctx.next_operator(), None);

    ctx.set_input(0, &input(7)).unwrap();
    while ctx.step().unwrap() == Step::Pending {}
    assert_eq!(ctx.output(0).unwrap().bytes(), &expected[..]);
}

#[test]
fn compute_async_yields_between_operators() {
    let model = kws();
    let graph =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
    let mut ctx = graph.init_execution_context_in(Global).unwrap();

    ctx.set_input(0, &input(3)).unwrap();
    ctx.compute().unwrap();
    let expected = ctx.output(0).unwrap().bytes().to_vec();

    ctx.set_input(0, &input(3)).unwrap();
    let (result, polls) = block_on(ctx.compute_async());
    result.unwrap();
    // Pending once after each operator but the last.
    assert_eq!(polls, graph.operator_count());
    assert_eq!(ctx.output(0).unwrap().bytes(), &expected[..]);
}

/// Lets `budget` operators run, then stops the computation.
struct Budget(usize);

impl Observer for Budget {
    fn before(&mut self, _op: &OpInfo<'_>) -> ControlFlow<()> {
        if self.0 == 0 {
            return ControlFlow::Break(());
        }
        self.0 -= 1;
        ControlFlow::Continue(())
    }
}

#[test]
fn an_observer_can_stop_the_computation() {
    let model = kws();
    let graph =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
    let mut ctx = graph.init_execution_context_in(Global).unwrap();
    ctx.set_input(0, &input(5)).unwrap();

    let error = ctx.compute_with(&mut Budget(4)).unwrap_err();
    assert_eq!(error.code, ErrorCode::Timeout);
    assert_eq!(error.detail, Detail::Interrupted { op_index: 4 });

    // Stepping: the interrupted step abandons the inference.
    ctx.set_input(0, &input(5)).unwrap();
    let mut budget = Budget(2);
    assert_eq!(ctx.step_with(&mut budget).unwrap(), Step::Pending);
    assert_eq!(ctx.step_with(&mut budget).unwrap(), Step::Pending);
    let error = ctx.step_with(&mut budget).unwrap_err();
    assert_eq!(error.detail, Detail::Interrupted { op_index: 2 });
    assert_eq!(ctx.next_operator(), None);

    // The context still works afterwards.
    ctx.set_input(0, &input(5)).unwrap();
    ctx.compute().unwrap();
}
