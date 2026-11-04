//! What every comparison firmware shares: the static keyword-spotting input (golden sample 0) and
//! the symbols through which `tools/emulate.py` reads the result (`BENCH_STATUS`, `BENCH_OUTPUT`).
//! The firmware run one inference and stop; nothing is printed.

#![no_std]

/// Input elements of the keyword-spotting model: 49 frames x 10 MFCC features.
pub const INPUT_LEN: usize = 49 * 10;
/// Output elements: 12 class scores.
pub const OUTPUT_LEN: usize = 12;

#[repr(C, align(4))]
struct Aligned<T>(T);

static INPUT: Aligned<[u8; INPUT_LEN]> = Aligned(*include_bytes!(
    "../../../../../testdata/mlperf-tiny/golden/kws_ref_model/sample_0/input_0.bin"
));

/// The int8 input: `testdata/mlperf-tiny/golden/kws_ref_model/sample_0/input_0.bin`.
pub fn input() -> &'static [i8; INPUT_LEN] {
    // SAFETY: i8 and u8 have the same size and alignment; the static lives forever.
    unsafe { &*INPUT.0.as_ptr().cast::<[i8; INPUT_LEN]>() }
}

/// The output of the inference, read by the emulator.
#[unsafe(no_mangle)]
pub static mut BENCH_OUTPUT: [i8; OUTPUT_LEN] = [0; OUTPUT_LEN];

/// An engine-specific number read by the emulator (nanoglia: the arena bytes used).
#[unsafe(no_mangle)]
pub static mut BENCH_INFO: u32 = 0;

/// 0 while running, 1 when the output is written, 2 on an engine error.
#[unsafe(no_mangle)]
pub static mut BENCH_STATUS: u32 = 0;

/// Publishes the output and stops.
pub fn finish(output: &[i8]) -> ! {
    let output = core::hint::black_box(output);
    // SAFETY: single-threaded firmware; nothing else accesses these statics.
    unsafe {
        let dst = core::ptr::addr_of_mut!(BENCH_OUTPUT).cast::<i8>();
        for (i, &x) in output.iter().take(OUTPUT_LEN).enumerate() {
            dst.add(i).write_volatile(x);
        }
        core::ptr::addr_of_mut!(BENCH_STATUS).write_volatile(1);
    }
    halt()
}

/// Records an engine-specific number for the emulator.
pub fn info(value: u32) {
    // SAFETY: single-threaded firmware.
    unsafe { core::ptr::addr_of_mut!(BENCH_INFO).write_volatile(value) }
}

/// Reports an engine error and stops.
pub fn fail() -> ! {
    // SAFETY: single-threaded firmware.
    unsafe { core::ptr::addr_of_mut!(BENCH_STATUS).write_volatile(2) }
    halt()
}

fn halt() -> ! {
    loop {
        cortex_m::asm::wfi();
    }
}
