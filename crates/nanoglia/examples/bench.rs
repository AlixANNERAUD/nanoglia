//! Times one inference of each MLPerf Tiny model (median of many runs) and prints a Markdown
//! table. Runs on the host and on wasm32-wasip1 under wasmtime (`mise run bench-wasm`).
//!
//! ```sh
//! cargo run --release -p nanoglia --example bench
//! ```

use nanoglia::core::allocator_api2::alloc::Global;
use nanoglia::{Encoding, Graph, Target};
use std::path::Path;
use std::time::{Duration, Instant};

const MODELS: &[&str] = &[
    "kws_ref_model",
    "vww_96_int8",
    "pretrainedResnet_quant",
    "ad01_int8",
];

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/mlperf-tiny");
    let budget = Duration::from_millis(
        std::env::var("NANOGLIA_BENCH_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2000),
    );
    println!("| Model | Operators | Median per inference | Runs |");
    println!("|---|---:|---:|---:|");
    for name in MODELS {
        let bytes = std::fs::read(root.join(format!("{name}.tflite"))).expect("model");
        // 16-byte aligned copy, as firmware would place it.
        let mut storage = vec![0u128; bytes.len().div_ceil(16)];
        // SAFETY: `storage` holds at least `bytes.len()` bytes.
        let model = unsafe {
            std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast::<u8>(), bytes.len())
        };
        model.copy_from_slice(&bytes);

        let graph = Graph::load_in(&[model], Encoding::Autodetect, Target::Auto, Global).unwrap();
        let mut ctx = graph.init_execution_context_in(Global).unwrap();
        let mut seed = 0x1234_5678u32;
        let mut input: Vec<u8> = vec![0; ctx.input_mut(0).unwrap().bytes_mut().len()];
        for x in &mut input {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *x = seed as u8;
        }

        let mut times = Vec::new();
        let start = Instant::now();
        while start.elapsed() < budget || times.len() < 5 {
            ctx.set_input(0, &input).unwrap();
            let t = Instant::now();
            ctx.compute().unwrap();
            times.push(t.elapsed());
        }
        times.sort();
        let median = times[times.len() / 2];
        println!(
            "| {name} | {} | {:.1} µs | {} |",
            graph.operator_count(),
            median.as_secs_f64() * 1e6,
            times.len()
        );
    }
}
