//! nanoglia: each model is loaded at run time from a 16-byte aligned copy of its `.tflite` bytes.

use nanoglia::{Encoding, Graph, Target};

fn main() {
    harness::run(run);
}

fn run() {
    for model in harness::MODELS {
        let bytes = harness::model_bytes(model);
        // 16-byte aligned copy, as firmware would place the model.
        let mut storage = vec![0u128; bytes.len().div_ceil(16)];
        // SAFETY: `storage` holds at least `bytes.len()` initialized bytes.
        let aligned = unsafe {
            std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast::<u8>(), bytes.len())
        };
        aligned.copy_from_slice(&bytes);

        let graph = match Graph::load(&[aligned], Encoding::Autodetect, Target::Auto) {
            Ok(graph) => graph,
            Err(e) => {
                harness::skip("nanoglia", model, &format!("load failed: {e}"));
                continue;
            }
        };
        let mut ctx = graph.init_execution_context().unwrap();
        harness::bench("nanoglia", model, |input, output| {
            // SAFETY: i8 and u8 have the same size and alignment.
            let bytes = unsafe { &*(input as *const [i8] as *const [u8]) };
            ctx.set_input(0, bytes).unwrap();
            ctx.compute().unwrap();
            output.copy_from_slice(ctx.output(0).unwrap().as_slice::<i8>().unwrap());
        });
    }
}
