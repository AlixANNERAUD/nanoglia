//! Keyword spotting on an Arduino Nano 33 BLE: loads the MLPerf Tiny DS-CNN model with nanoglia,
//! runs it on a test input and prints the scores and the inference time over RTT.
//!
//! Nothing here uses a heap: the graph and its execution context live in a static [`Arena`].

#![no_std]
#![no_main]

use cortex_m::peripheral::DWT;
use cortex_m_rt::entry;
use defmt_rtt as _;
use nanoglia::{Arena, Encoding, Graph, Target};
use panic_probe as _;

/// The nRF52840 runs from its 64 MHz internal oscillator after reset.
const CPU_HZ: u32 = 64_000_000;

/// The model bytes, 16-byte aligned like the TFLite converter expects.
#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);

static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!("../../../testdata/mlperf-tiny/kws_ref_model.tflite"));

/// Memory for the graph and its execution context (about 24 KiB for this model).
static mut MEMORY: [u8; 32 * 1024] = [0; 32 * 1024];

#[entry]
fn main() -> ! {
    let mut core = cortex_m::Peripherals::take().unwrap();
    core.DCB.enable_trace();
    DWT::unlock();
    core.DWT.enable_cycle_counter();

    // SAFETY: `main` runs once and is the only user of `MEMORY`.
    let arena = Arena::new(unsafe { &mut *core::ptr::addr_of_mut!(MEMORY) });

    let start = DWT::cycle_count();
    let graph = match Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena) {
        Ok(graph) => graph,
        Err(error) => defmt::panic!("load failed: {}", error),
    };
    let load_cycles = DWT::cycle_count().wrapping_sub(start);
    let mut ctx = graph.init_execution_context_in(&arena).unwrap();
    defmt::info!(
        "loaded {} operators in {} us: graph {} B, context {} B, arena used {} B",
        graph.operator_count(),
        load_cycles / (CPU_HZ / 1_000_000),
        graph.graph_bytes(),
        graph.context_bytes(),
        arena.used()
    );
    for i in 0..graph.operator_count() {
        defmt::debug!("op {}: {}", i, graph.kernel_name(i).unwrap_or("?"));
    }

    loop {
        // A deterministic test pattern in place of real MFCC features. The inputs are written
        // before every compute: the memory plan reuses their memory once consumed.
        let mut seed = 0x1234_5678u32;
        for x in ctx.input_mut(0).unwrap().as_mut_slice::<i8>().unwrap() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *x = seed as i8;
        }

        let start = DWT::cycle_count();
        ctx.compute().unwrap();
        let cycles = DWT::cycle_count().wrapping_sub(start);
        let scores = ctx.output(0).unwrap();
        let scores = scores.as_slice::<i8>().unwrap();
        let best = (0..scores.len()).max_by_key(|&i| scores[i]).unwrap_or(0);
        defmt::info!(
            "inference: {} cycles ({} us), class {}, scores {=[?]}",
            cycles,
            cycles / (CPU_HZ / 1_000_000),
            best,
            scores
        );
        cortex_m::asm::delay(CPU_HZ);
    }
}
