//! embedded-nn: the model is imported and compiled at build time (path relative to this crate's
//! manifest); the activations live in a static arena of the size its planner computed.

#![no_std]
#![no_main]

use panic_halt as _;

#[embedded_nn_macros::embedded_nn_model("../../../../testdata/mlperf-tiny/kws_ref_model.tflite")]
struct Model;

static mut ARENA: [u8; Model::ARENA_SIZE] = [0; Model::ARENA_SIZE];

#[cortex_m_rt::entry]
fn main() -> ! {
    // SAFETY: `main` runs once and is the only user of `ARENA`.
    let arena = unsafe { &mut *core::ptr::addr_of_mut!(ARENA) };
    fw_common::info(Model::ARENA_SIZE as u32);
    match Model::predict(fw_common::input(), arena) {
        Ok(scores) => fw_common::finish(scores),
        Err(_) => fw_common::fail(),
    }
}
