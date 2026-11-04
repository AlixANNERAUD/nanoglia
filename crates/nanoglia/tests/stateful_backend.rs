//! A backend with state: it borrows a counter owned by the test (not `'static`), and its kernels
//! delegate to the default ones while counting their evaluations.

use nanoglia::core::OpDesc;
use nanoglia::core::allocator_api2::alloc::Global;
use nanoglia::core::backend::{Backend, EvalCtx, Kernel, KernelPlan};
use nanoglia::{Encoding, Graph, Registry, Result, Target};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

mod common;

/// A kernel of the default registry, counting its evaluations in a counter it borrows.
struct CountingKernel<'a> {
    inner: &'static dyn Kernel,
    evaluations: &'a AtomicUsize,
}

impl Kernel for CountingKernel<'_> {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn plan(&self, op: &OpDesc<'_, '_>) -> Result<KernelPlan> {
        self.inner.plan(op)
    }

    fn prepare(&self, op: &OpDesc<'_, '_>, persist: &mut [u8]) -> Result<()> {
        self.inner.prepare(op, persist)
    }

    fn eval(&self, ctx: &mut EvalCtx<'_, '_>) -> Result<()> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(ctx)
    }
}

/// Wraps the default backends. It owns one wrapper per distinct kernel they select, and hands
/// out references to them: the kernels live in the backend, not in a `static`.
struct Counting<'a> {
    evaluations: &'a AtomicUsize,
    kernels: [OnceLock<CountingKernel<'a>>; 32],
}

impl<'a> Counting<'a> {
    fn new(evaluations: &'a AtomicUsize) -> Self {
        Self {
            evaluations,
            kernels: [const { OnceLock::new() }; 32],
        }
    }
}

impl Backend for Counting<'_> {
    fn name(&self) -> &'static str {
        "counting"
    }

    fn select<'s>(&'s self, op: &OpDesc<'_, '_>) -> Option<&'s dyn Kernel> {
        let inner = Registry::default()
            .backends()
            .iter()
            .find_map(|backend| backend.select(op))?;
        for slot in &self.kernels {
            let kernel = slot.get_or_init(|| CountingKernel {
                inner,
                evaluations: self.evaluations,
            });
            if core::ptr::addr_eq(kernel.inner, inner) {
                return Some(kernel);
            }
        }
        None
    }
}

fn kws() -> common::Aligned {
    common::Aligned::read(&common::repo_root().join("testdata/mlperf-tiny/kws_ref_model.tflite"))
}

#[test]
fn a_backend_borrowing_local_state_counts_its_evaluations() {
    let model = kws();
    let input: Vec<u8> = (0..490).map(|i| (i * 7 % 256) as u8).collect();

    // Reference: the default backends.
    let reference =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap();
    let mut ctx = reference.init_execution_context_in(Global).unwrap();
    ctx.set_input(0, &input).unwrap();
    ctx.compute().unwrap();
    let expected = ctx.output(0).unwrap().bytes().to_vec();

    // Local state: the counter, then the backend borrowing it, then the registry borrowing the
    // backend. None of them is `'static`; the graph borrows them all.
    let evaluations = AtomicUsize::new(0);
    let counting = Counting::new(&evaluations);
    let backends: [&dyn Backend; 1] = [&counting];
    let registry = Registry::new(&backends);
    let graph = Graph::load_with(
        &registry,
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        Global,
    )
    .unwrap();
    assert_eq!(graph.memory_requirements(), reference.memory_requirements());
    for i in 0..graph.operator_count() {
        assert_eq!(
            graph.kernel_name(i),
            reference.kernel_name(i),
            "operator {i}"
        );
    }

    let mut ctx = graph.init_execution_context_in(Global).unwrap();
    for run in 1..=2 {
        ctx.set_input(0, &input).unwrap();
        ctx.compute().unwrap();
        assert_eq!(ctx.output(0).unwrap().bytes(), expected);
        assert_eq!(
            evaluations.load(Ordering::Relaxed),
            run * graph.operator_count()
        );
    }
    drop(ctx);
    drop(graph);
    // The backend is still ours once the graph is gone.
    assert!(counting.kernels.iter().any(|slot| slot.get().is_some()));
}

#[test]
fn graphs_are_send_and_sync_with_borrowed_backends() {
    fn send_sync<T: Send + Sync>(_: &T) {}
    let model = kws();
    let evaluations = AtomicUsize::new(0);
    let counting = Counting::new(&evaluations);
    let backends: [&dyn Backend; 1] = [&counting];
    let graph = Graph::load_with(
        &Registry::new(&backends),
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        Global,
    )
    .unwrap();
    send_sync(&graph);

    let run = || {
        let mut ctx = graph.init_execution_context_in(Global).unwrap();
        ctx.input_mut(0)
            .unwrap()
            .as_mut_slice::<i8>()
            .unwrap()
            .fill(5);
        ctx.compute().unwrap();
    };
    // Two threads share the graph and its borrowed backend, each with its own context (one after
    // the other where threads are not available, under wasm32-wasip1).
    if cfg!(target_os = "wasi") {
        run();
        run();
    } else {
        std::thread::scope(|scope| {
            scope.spawn(run);
            scope.spawn(run);
        });
    }
    assert_eq!(
        evaluations.load(Ordering::Relaxed),
        2 * graph.operator_count()
    );
}
