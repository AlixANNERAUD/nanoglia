//! Checks `ModelReport`: it reports every operator, agrees with what loading selects, allocates
//! and returns, and never panics on a damaged model.

use core::alloc::Layout;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};
use nanoglia::core::OpDesc;
use nanoglia::core::allocator_api2::alloc::{AllocError, Allocator, Global};
use nanoglia::core::backend::{Backend, FnKernel, Kernel, KernelPlan};
use nanoglia::{
    Arena, BuiltinOp, Detail, Encoding, Error, ErrorCode, Graph, ModelReport, OpCode, Registry,
    SelectedKernel, Target,
};
use std::path::{Path, PathBuf};

mod common;

use common::Aligned;

/// What a check reports for one operator.
#[derive(Clone, Debug, PartialEq)]
struct Seen {
    opcode: Option<BuiltinOp>,
    kernel: Result<SelectedKernel, Error>,
}

/// Checks `model` with `registry` and returns the report with what each operator got.
fn check(registry: &Registry<'_>, model: &[u8], target: Target) -> (ModelReport, Vec<Seen>) {
    let mut seen = Vec::new();
    let report = ModelReport::check_with(
        registry,
        &[model],
        Encoding::Autodetect,
        target,
        Global,
        |op| {
            assert_eq!(op.index(), seen.len());
            seen.push(Seen {
                opcode: match op.opcode() {
                    OpCode::Builtin(builtin) => Some(builtin),
                    OpCode::Custom(_) => None,
                },
                kernel: op.kernel(),
            });
        },
    )
    .unwrap();
    assert_eq!(report.operators, seen.len());
    assert_eq!(
        report.unsupported,
        seen.iter().filter(|op| op.kernel.is_err()).count()
    );
    (report, seen)
}

fn kws() -> Aligned {
    Aligned::read(&common::repo_root().join("testdata/mlperf-tiny/kws_ref_model.tflite"))
}

/// Every `.tflite` file of `testdata/`.
fn models() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "tflite") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(&common::repo_root().join("testdata"), &mut out);
    out.sort();
    out
}

/// The model with the builtin code `from` of its operator-code table replaced by `to` (both
/// below 128: the table stores such codes in one byte, and takes the larger of that byte and of
/// the 32-bit code). The byte is the one after which the reader sees `to` wherever it saw `from`.
fn with_opcode(model: &[u8], from: BuiltinOp, to: BuiltinOp) -> Aligned {
    let registry = Registry::default();
    let (_, before) = check(&registry, model, Target::Auto);
    let expected: Vec<Option<BuiltinOp>> = before
        .iter()
        .map(|op| {
            if op.opcode == Some(from) {
                Some(to)
            } else {
                op.opcode
            }
        })
        .collect();
    assert!(expected.contains(&Some(to)), "no operator {from:?}");
    let (from_byte, to_byte) = (u8::try_from(from.0).unwrap(), u8::try_from(to.0).unwrap());
    assert!(from_byte < 128 && to_byte < 128 && to_byte > from_byte);
    for at in 0..model.len() {
        if model[at] != from_byte {
            continue;
        }
        let mut patched = model.to_vec();
        patched[at] = to_byte;
        let patched = Aligned::copy(&patched);
        let mut seen = Vec::new();
        let walked = ModelReport::check_in(
            &[patched.bytes()],
            Encoding::Autodetect,
            Target::Auto,
            Global,
            |op| {
                seen.push(match op.opcode() {
                    OpCode::Builtin(builtin) => Some(builtin),
                    OpCode::Custom(_) => None,
                })
            },
        );
        if walked.is_ok() && seen == expected {
            return patched;
        }
    }
    panic!("no builtin code {from:?} to replace");
}

const WHILE: BuiltinOp = BuiltinOp(119);
const IF: BuiltinOp = BuiltinOp(118);

#[test]
fn every_model_reports_the_kernels_loading_selects() {
    let registry = Registry::default();
    let models = models();
    assert!(models.len() > 100, "testdata/ not found");
    for path in models {
        let model = Aligned::read(&path);
        let (report, seen) = check(&registry, model.bytes(), Target::Auto);
        let graph = Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(report.runs(), "{}", path.display());
        assert_eq!(report.unsupported, 0);
        assert_eq!(seen.len(), graph.operator_count());
        for (i, op) in seen.iter().enumerate() {
            let kernel = op.kernel.unwrap();
            assert_eq!(
                Some(kernel.name),
                graph.kernel_name(i),
                "{}: operator {i}",
                path.display()
            );
            // A kernel name starts with the name of its backend.
            assert!(kernel.name.starts_with(kernel.backend), "{}", kernel.name);
        }
        let memory = report.memory.unwrap();
        assert_eq!(
            memory.requirements(),
            graph.memory_requirements(),
            "{}",
            path.display()
        );
        assert_eq!(
            memory.context_bytes,
            memory.activation_bytes + memory.scratch_bytes
        );
        assert!(memory.persistent_bytes <= memory.graph_bytes);
        let planned = seen.iter().map(|op| op.kernel.unwrap().plan);
        assert_eq!(
            memory.persistent_bytes,
            planned
                .clone()
                .map(|plan| plan.persistent.next_multiple_of(16))
                .sum::<usize>()
        );
        assert_eq!(
            memory.scratch_bytes,
            planned
                .map(|plan| plan.scratch)
                .max()
                .unwrap_or(0)
                .next_multiple_of(16)
        );
    }
}

#[test]
fn operators_describe_their_tensors() {
    let model = kws();
    let mut rows = Vec::new();
    ModelReport::check_in(
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        Global,
        |op| {
            let inputs: Vec<_> = (0..op.input_count())
                .map(|i| op.input(i).unwrap().unwrap())
                .collect();
            let output = op.output(0).unwrap();
            assert_eq!(op.output_count(), 1);
            assert_eq!(
                op.input(op.input_count()).unwrap_err().detail,
                Detail::IndexOutOfRange {
                    index: op.input_count() as u32
                }
            );
            assert!(op.output(1).is_err());
            rows.push((
                op.opcode(),
                op.version(),
                inputs[0].dims.as_slice().to_vec(),
                inputs.iter().filter(|t| t.is_constant()).count(),
                output.dims.as_slice().to_vec(),
            ));
        },
    )
    .unwrap();
    assert_eq!(rows.len(), 13);
    assert_eq!(
        rows[0],
        (
            OpCode::Builtin(BuiltinOp::CONV_2D),
            3,
            vec![1, 49, 10, 1],
            2,
            vec![1, 25, 5, 64]
        )
    );
    assert_eq!(
        rows[12],
        (
            OpCode::Builtin(BuiltinOp::SOFTMAX),
            2,
            vec![1, 12],
            0,
            vec![1, 12]
        )
    );
}

#[test]
fn two_unsupported_operators_are_both_reported() {
    // Keyword spotting with its RESHAPE (operator 10) and SOFTMAX (operator 12) turned into
    // operators nanoglia has no kernel for.
    let model = kws();
    let model = with_opcode(model.bytes(), BuiltinOp::RESHAPE, WHILE);
    let model = with_opcode(model.bytes(), BuiltinOp::SOFTMAX, IF);
    let registry = Registry::default();
    let (report, seen) = check(&registry, model.bytes(), Target::Auto);

    assert!(!report.runs());
    assert_eq!((report.operators, report.unsupported), (13, 2));
    let blocking: Vec<_> = seen
        .iter()
        .enumerate()
        .filter_map(|(i, op)| Some((i, op.opcode?.name()?, op.kernel.err()?)))
        .collect();
    assert_eq!(
        blocking,
        [
            (
                10,
                "WHILE",
                Error::unsupported(Detail::UnsupportedOp {
                    op_index: 10,
                    opcode: 119
                })
            ),
            (
                12,
                "IF",
                Error::unsupported(Detail::UnsupportedOp {
                    op_index: 12,
                    opcode: 118
                })
            ),
        ]
    );
    // The others keep their kernels.
    assert_eq!(seen[11].kernel.unwrap().name, "cmsis-nn/fully_connected_s8");

    // Loading stops at the first one, with the error the report gives for the model.
    let error =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global).unwrap_err();
    assert_eq!(report.memory, Err(error));
    assert_eq!(Err(error), seen[10].kernel);
}

#[test]
fn the_verdict_depends_on_the_registry() {
    let model = kws();
    // Only the Rust kernels: they leave int8 convolutions, pooling, fully-connected and softmax
    // to CMSIS-NN.
    let backends: [&dyn Backend; 1] = [Registry::default().backends().last().copied().unwrap()];
    assert_eq!(backends[0].name(), "rust");
    let registry = Registry::new(&backends);
    let (report, seen) = check(&registry, model.bytes(), Target::Auto);
    assert_eq!(report.unsupported, 12);
    assert_eq!(seen[10].kernel.unwrap().name, "rust/reshape");
    let error = Graph::load_with(
        &registry,
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        Global,
    )
    .unwrap_err();
    assert_eq!(report.memory, Err(error));

    // No accelerator in the default registry: every operator has a kernel, the model is refused.
    let registry = Registry::default();
    let (report, _) = check(&registry, model.bytes(), Target::Npu);
    assert_eq!(report.unsupported, 0);
    let error =
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Npu, Global).unwrap_err();
    assert_eq!(report.memory, Err(error));
    assert!(!report.runs());
}

#[test]
fn an_operator_whose_options_cannot_be_decoded_is_still_described() {
    // A convolution with one byte of its options damaged (an unknown padding, stored because
    // VALID is not the default):
    // the reader refuses to decode the operator, the report still describes it.
    let path = common::repo_root().join("testdata/ops/conv2d_3x3_stride2_valid_int8/model.tflite");
    let original = Aligned::read(&path);
    let mut found = 0;
    for at in 0..original.bytes().len() {
        let mut bytes = original.bytes().to_vec();
        bytes[at] = 0x7F;
        let model = Aligned::copy(&bytes);
        let Ok(graph) =
            nanoglia_tflite::Model::parse(model.bytes()).and_then(|model| model.subgraph(0))
        else {
            continue;
        };
        let Err(error) = graph.operator(0) else {
            continue;
        };
        found += 1;
        assert_eq!(
            error,
            Error::unsupported(Detail::UnsupportedOp {
                op_index: 0,
                opcode: BuiltinOp::CONV_2D.0
            })
        );
        let mut visited = 0;
        let report = ModelReport::check_in(
            &[model.bytes()],
            Encoding::Autodetect,
            Target::Auto,
            Global,
            |op| {
                visited += 1;
                assert_eq!(op.opcode(), OpCode::Builtin(BuiltinOp::CONV_2D));
                assert_eq!(op.input_count(), 3);
                let input = op.input(0).unwrap().unwrap();
                assert_eq!(input.dims.rank(), 4);
                assert_eq!(op.output(0).unwrap().dtype, nanoglia::DataType::I8);
                assert_eq!(op.kernel(), Err(error));
            },
        )
        .unwrap();
        assert_eq!((visited, report.unsupported), (1, 1));
        assert_eq!(report.memory, Err(error));
        let loaded = Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global);
        assert_eq!(loaded.unwrap_err(), error);
    }
    assert!(found > 0, "no byte of the options was found");
}

/// A backend whose kernel for operator 3 plans and then fails to prepare; the default backends
/// run the others.
struct FailsToPrepare;

static REFUSING: FnKernel = FnKernel {
    name: "test/refusing",
    plan: |_| {
        Ok(KernelPlan {
            persistent: 24,
            scratch: 0,
        })
    },
    prepare: |op, persist| {
        // The contract of `prepare`: the planned size, zeroed and aligned.
        assert_eq!(persist.len(), 24);
        assert!(persist.iter().all(|&b| b == 0));
        assert_eq!(persist.as_ptr().align_offset(16), 0);
        persist.fill(0xAA);
        Err(Error::unsupported(Detail::ShapeMismatch {
            tensor: op.outputs[0].index,
        }))
    },
    eval: |_| Ok(()),
};

impl Backend for FailsToPrepare {
    fn name(&self) -> &'static str {
        "test"
    }

    fn select<'s>(&'s self, op: &OpDesc<'_, '_>) -> Option<&'s dyn Kernel> {
        if op.index == 3 || op.index == 5 {
            return Some(&REFUSING);
        }
        Registry::default()
            .backends()
            .iter()
            .find_map(|backend| backend.select(op))
    }
}

#[test]
fn a_kernel_that_fails_to_prepare_blocks_its_operator() {
    let model = kws();
    let backend = FailsToPrepare;
    let backends: [&dyn Backend; 1] = [&backend];
    let registry = Registry::new(&backends);
    let (report, seen) = check(&registry, model.bytes(), Target::Auto);
    assert_eq!(report.unsupported, 2);
    assert!(matches!(
        seen[3].kernel.unwrap_err().detail,
        Detail::ShapeMismatch { .. }
    ));
    assert!(seen[5].kernel.is_err());
    assert_eq!(
        seen[4].kernel.unwrap().name,
        "cmsis-nn/conv2d_s8",
        "prepared after a failure, in the same buffer"
    );
    let error = Graph::load_with(
        &registry,
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        Global,
    )
    .unwrap_err();
    assert_eq!(report.memory, Err(error));
}

#[derive(Default)]
struct Counting {
    allocations: AtomicUsize,
    live_bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
}

// SAFETY: forwards to the global allocator.
unsafe impl Allocator for Counting {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocations.fetch_add(1, Ordering::SeqCst);
        let live = self.live_bytes.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
        self.peak_bytes.fetch_max(live, Ordering::SeqCst);
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.live_bytes.fetch_sub(layout.size(), Ordering::SeqCst);
        // SAFETY: allocated by `Global` with this layout in `allocate`.
        unsafe { Global.deallocate(ptr, layout) }
    }
}

#[test]
fn checking_borrows_one_block_and_reports_what_loading_allocates() {
    let model = kws();
    let counting = Counting::default();
    let mut largest = 0;
    let report = ModelReport::check_in(
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        &counting,
        |op| largest = largest.max(op.kernel().unwrap().plan.persistent),
    )
    .unwrap();
    // One block, given back: 20 bytes per tensor and the largest prepared parameters.
    assert_eq!(counting.allocations.load(Ordering::SeqCst), 1);
    assert_eq!(counting.live_bytes.load(Ordering::SeqCst), 0);
    let peak = counting.peak_bytes.load(Ordering::SeqCst);
    assert!(peak >= largest && peak < report.memory.unwrap().graph_bytes);

    // The sizes reported are the sizes of the two allocations of an inference.
    let memory = report.memory.unwrap();
    let counting = Counting::default();
    let graph = Graph::load_in(
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        &counting,
    )
    .unwrap();
    assert_eq!(
        counting.live_bytes.load(Ordering::SeqCst),
        memory.graph_bytes
    );
    let ctx = graph.init_execution_context_in(&counting).unwrap();
    assert_eq!(
        counting.live_bytes.load(Ordering::SeqCst),
        memory.graph_bytes + memory.context_bytes
    );
    assert_eq!(counting.allocations.load(Ordering::SeqCst), 2);
    drop(ctx);

    // Without a heap: an arena of the peak size (and room to align it) is enough, a smaller one
    // is an error that says how much was asked, before any operator is visited.
    let mut buffer = vec![0u8; peak + 16];
    let arena = Arena::new(&mut buffer);
    let again = ModelReport::check_in(
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        &arena,
        |_| {},
    )
    .unwrap();
    assert_eq!(again, report);
    let mut buffer = vec![0u8; peak - 16];
    let mut visited = 0;
    let error = ModelReport::check_in(
        &[model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        Arena::new(&mut buffer),
        |_| visited += 1,
    )
    .unwrap_err();
    assert_eq!(error, Error::out_of_memory(peak));
    assert_eq!(visited, 0);
}

#[test]
fn what_is_not_a_single_graph_model_is_an_error() {
    let visit = |_: &nanoglia::OperatorReport<'_>| panic!("no operator to visit");
    let error = ModelReport::check_in(
        &[b"not a model"],
        Encoding::Autodetect,
        Target::Auto,
        Global,
        visit,
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::new(ErrorCode::InvalidEncoding, Detail::UnknownEncoding)
    );
    let model = kws();
    let error = ModelReport::check_in(
        &[model.bytes(), model.bytes()],
        Encoding::Autodetect,
        Target::Auto,
        Global,
        visit,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    let error = ModelReport::check_in(
        &[&model.bytes()[..4096]],
        Encoding::TensorFlowLite,
        Target::Auto,
        Global,
        visit,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidEncoding);
}

/// The verdict of the check, to compare with loading.
fn verdict(model: &[u8]) -> Result<(), Error> {
    ModelReport::check_in(&[model], Encoding::Autodetect, Target::Auto, Global, |op| {
        // Read everything a report exposes.
        for i in 0..op.input_count() {
            let _ = op.input(i);
        }
        for i in 0..op.output_count() {
            let _ = op.output(i);
        }
        let _ = format!("{op:?}");
    })?
    .memory
    .map(|_| ())
}

#[test]
fn damaged_models_never_panic_and_get_the_verdict_of_loading() {
    // Every byte of a small model, damaged in turn: the check must not panic, and must agree
    // with loading on whether the model runs and on the error.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("doc/classifier.tflite");
    let original = Aligned::read(&path);
    assert_eq!(verdict(original.bytes()), Ok(()));
    let mut refused = 0;
    for at in 0..original.bytes().len() {
        for damage in [0x01, 0x80, 0xFF] {
            let mut bytes = original.bytes().to_vec();
            bytes[at] ^= damage;
            let model = Aligned::copy(&bytes);
            let loaded =
                Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global)
                    .map(|_| ());
            assert_eq!(verdict(model.bytes()), loaded, "byte {at} ^ {damage:#x}");
            refused += usize::from(loaded.is_err());
        }
    }
    assert!(refused > 100, "only {refused} damaged models were refused");

    // Truncated at every length.
    for len in 0..original.bytes().len() {
        let model = Aligned::copy(&original.bytes()[..len]);
        let loaded = Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global)
            .map(|_| ());
        assert_eq!(verdict(model.bytes()), loaded, "{len} bytes");
    }
}
