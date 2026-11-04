//! ember-rs: the model is compiled into code at build time (path relative to the workspace root)
//! and run with the pure Rust reference backend. Activations are local arrays of the generated
//! function, on the stack. The reference backend's softmax allocates, hence the small heap.

#![no_std]
#![no_main]

extern crate alloc;

use embedded_alloc::LlffHeap as Heap;
use panic_halt as _;

#[ember_infer_macros::model("../../testdata/mlperf-tiny/kws_ref_model.tflite")]
struct Model;

#[global_allocator]
static HEAP: Heap = Heap::empty();

/// Heap for the softmax's temporary `Vec<f32>` (12 classes).
const HEAP_BYTES: usize = 256;

#[cortex_m_rt::entry]
fn main() -> ! {
    {
        use core::mem::MaybeUninit;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_BYTES] = [MaybeUninit::uninit(); HEAP_BYTES];
        // SAFETY: called once, before any allocation; `HEAP_MEM` is used by nothing else.
        unsafe { HEAP.init(core::ptr::addr_of_mut!(HEAP_MEM) as usize, HEAP_BYTES) }
    }
    let mut backend = ember_infer_ref::RefBackend;
    let mut output = [0i8; fw_common::OUTPUT_LEN];
    match Model::predict_quantized(&mut backend, fw_common::input(), &mut output) {
        Ok(()) => fw_common::finish(&output),
        Err(_) => fw_common::fail(),
    }
}
