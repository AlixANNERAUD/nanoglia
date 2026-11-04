//! What every host benchmark of `benchmarks/compare` shares, so that the engines are measured the
//! same way:
//!
//! - the inputs are the golden samples of `testdata/mlperf-tiny/golden/<model>/sample_*`, and the
//!   outputs are compared with their `output_0.bin` (LiteRT reference kernels);
//! - the timed call is the engine's whole inference from an `&[i8]` input to an `&mut [i8]`
//!   output, input and output copies included;
//! - after a warm-up, the call is repeated for a time budget, cycling through the samples, and
//!   the median is reported.
//!
//! Each benchmark prints one tab-separated `RESULT` line per model, which
//! `benchmarks/compare/tools/compare.py` collects.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The MLPerf Tiny models of `testdata/mlperf-tiny`.
pub const MODELS: &[&str] = &[
    "kws_ref_model",
    "vww_96_int8",
    "pretrainedResnet_quant",
    "ad01_int8",
];

/// One golden sample: an int8 input and the int8 output LiteRT's reference kernels give.
pub struct Sample {
    pub input: Vec<i8>,
    pub expected: Vec<i8>,
}

/// `testdata/mlperf-tiny` of the repository.
pub fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../testdata/mlperf-tiny")
}

/// The bytes of `testdata/mlperf-tiny/<model>.tflite`.
pub fn model_bytes(model: &str) -> Vec<u8> {
    let path = testdata().join(format!("{model}.tflite"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn read_i8(path: &Path) -> Vec<i8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.into_iter().map(|b| b as i8).collect()
}

/// The golden samples of a model, in order.
pub fn samples(model: &str) -> Vec<Sample> {
    let dir = testdata().join("golden").join(model);
    let mut samples = Vec::new();
    for k in 0.. {
        let case = dir.join(format!("sample_{k}"));
        if !case.is_dir() {
            break;
        }
        samples.push(Sample {
            input: read_i8(&case.join("input_0.bin")),
            expected: read_i8(&case.join("output_0.bin")),
        });
    }
    assert!(
        !samples.is_empty(),
        "no golden samples in {}",
        dir.display()
    );
    samples
}

/// The `(scale, zero_point)` of the model's first output, read from the `case.json` of its first
/// golden sample: for engines that only return dequantized `f32` outputs.
pub fn output_quantization(model: &str) -> (f32, i32) {
    let path = testdata()
        .join("golden")
        .join(model)
        .join("sample_0/case.json");
    let json = std::fs::read_to_string(&path).unwrap();
    let outputs = &json[json.find("\"outputs\"").expect("outputs")..];
    let number = |key: &str| {
        let rest = &outputs[outputs.find(key).expect(key) + key.len()..];
        let rest = rest.trim_start_matches([':', ' ']);
        let end = rest.find([',', '\n', '}']).unwrap();
        rest[..end].trim().to_string()
    };
    (
        number("\"scale\"").parse().unwrap(),
        number("\"zero_point\"").parse().unwrap(),
    )
}

/// Time budget per model: `BENCH_MS` milliseconds (default 2000).
fn budget() -> Duration {
    Duration::from_millis(
        std::env::var("BENCH_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2000),
    )
}

/// Checks `infer` on every golden sample of `model`, then times it, and prints a `RESULT` line:
/// `RESULT engine model samples max_abs_diff differing_elements total_elements top1_agree
/// median_us runs`.
///
/// `infer` runs one whole inference: it reads the int8 input and writes the int8 output.
pub fn bench(engine: &str, model: &str, mut infer: impl FnMut(&[i8], &mut [i8])) {
    let samples = samples(model);
    let mut output = vec![0i8; samples[0].expected.len()];

    // Correctness on every sample.
    let (mut max_diff, mut differing, mut total, mut top1) = (0i32, 0usize, 0usize, 0usize);
    for sample in &samples {
        output.fill(0);
        infer(&sample.input, &mut output);
        for (&a, &b) in output.iter().zip(&sample.expected) {
            let d = (i32::from(a) - i32::from(b)).abs();
            max_diff = max_diff.max(d);
            differing += usize::from(d != 0);
            total += 1;
        }
        top1 += usize::from(argmax(&output) == argmax(&sample.expected));
        if std::env::var_os("BENCH_DEBUG").is_some() {
            let n = output.len().min(16);
            eprintln!("{engine} {model}: got      {:?}", &output[..n]);
            eprintln!("{engine} {model}: expected {:?}", &sample.expected[..n]);
        }
    }

    // Warm-up: at least 10 runs and 200 ms.
    let start = Instant::now();
    let mut i = 0;
    while i < 10 || start.elapsed() < Duration::from_millis(200) {
        infer(&samples[i % samples.len()].input, &mut output);
        i += 1;
    }

    let budget = budget();
    let mut times = Vec::new();
    let start = Instant::now();
    while start.elapsed() < budget || times.len() < 11 {
        let sample = &samples[times.len() % samples.len()];
        let t = Instant::now();
        infer(std::hint::black_box(&sample.input), &mut output);
        times.push(t.elapsed());
        std::hint::black_box(&output);
    }
    times.sort();
    let median = times[times.len() / 2];
    println!(
        "RESULT\t{engine}\t{model}\t{}\t{max_diff}\t{differing}\t{total}\t{top1}\t{:.2}\t{}",
        samples.len(),
        median.as_secs_f64() * 1e6,
        times.len()
    );
}

/// Runs `f` on a thread with a 256 MiB stack: the ahead-of-time engines keep their activations
/// on the stack, more than the 8 MiB of the main thread for the larger models. Every engine runs
/// this way.
pub fn run(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

/// Prints a `SKIP` line for a model the engine cannot run, with the reason.
pub fn skip(engine: &str, model: &str, reason: &str) {
    println!("SKIP\t{engine}\t{model}\t{reason}");
}

fn argmax(v: &[i8]) -> usize {
    // First maximum, as numpy.argmax.
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
