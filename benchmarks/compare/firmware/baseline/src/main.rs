//! No engine: copies the first input bytes to the output. Its size is the part of every other
//! firmware that is not the engine (vector table, start-up code, input, result symbols).

#![no_std]
#![no_main]

use panic_halt as _;

#[cortex_m_rt::entry]
fn main() -> ! {
    let input = core::hint::black_box(fw_common::input());
    fw_common::finish(&input[..fw_common::OUTPUT_LEN])
}
