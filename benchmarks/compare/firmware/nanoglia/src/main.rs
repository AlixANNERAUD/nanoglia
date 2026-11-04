//! nanoglia: verifies and loads the `.tflite` bytes from flash at run time, then runs one
//! inference. The graph and its execution context come from an `Arena` over a static buffer.

#![no_std]
#![no_main]

use nanoglia::{Arena, Encoding, Graph, Target};
use panic_halt as _;

#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);

static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(
    "../../../../../testdata/mlperf-tiny/kws_ref_model.tflite"
));

/// `graph_bytes + context_bytes` on thumbv7em-none-eabihf (CMSIS-NN with DSP), measured by
/// `tools/emulate.py` (`BENCH_INFO`). The buffer is 16-byte aligned, so `Arena` loses nothing to
/// alignment; `MemoryRequirements::arena_bytes(1)` adds 15 bytes for a buffer of unknown
/// alignment.
const ARENA_BYTES: usize = 23872;

static mut MEMORY: Aligned<[u8; ARENA_BYTES]> = Aligned([0; ARENA_BYTES]);

#[cortex_m_rt::entry]
fn main() -> ! {
    // SAFETY: `main` runs once and is the only user of `MEMORY`.
    let arena = Arena::new(unsafe { &mut (*core::ptr::addr_of_mut!(MEMORY)).0 });
    let Ok(graph) = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena) else {
        fw_common::fail()
    };
    let Ok(mut ctx) = graph.init_execution_context_in(&arena) else {
        fw_common::fail()
    };
    fw_common::info(arena.used() as u32);

    // SAFETY: i8 and u8 have the same size and alignment.
    let input = unsafe { &*(fw_common::input() as *const [i8] as *const [u8]) };
    if ctx.set_input(0, input).is_err() || ctx.compute().is_err() {
        fw_common::fail()
    }
    match ctx.output(0).as_ref().map(|o| o.as_slice::<i8>()) {
        Ok(Ok(scores)) => fw_common::finish(scores),
        _ => fw_common::fail(),
    }
}
