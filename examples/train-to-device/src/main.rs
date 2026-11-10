//! Runs the gesture classifier on the test vectors exported with it and checks every output
//! against the one LiteRT computed. It is the program the tutorial
//! `documentation/tutorials/train-and-deploy.md` writes, but for the model it embeds.
//!
//! ```sh
//! cargo run --release -- reference
//! ```
//!
//! The argument is a directory written by `training/train_pytorch.py` or
//! `training/train_keras.py`: `sample_<k>/window.bin` holds an accelerometer window (float32,
//! in g), `input_0.bin` the int8 input and `output_0.bin` the int8 output of LiteRT. The model
//! is embedded at build time: `reference/model.tflite`, or the file named by the
//! `GESTURE_MODEL` environment variable (see `build.rs`). It must be the model of the directory.
//!
//! The program uses nanoglia as a firmware would: the model in a static, every byte of working
//! memory in a static buffer. Only reading the test vectors and printing need `std`.

use nanoglia::{Arena, Encoding, Graph, Target};

/// The model bytes, placed on a 16-byte boundary so that its constant buffers are aligned.
#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);

static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(env!("GESTURE_MODEL_PATH")));

/// All the memory nanoglia may use: the graph and one execution context.
static mut MEMORY: [u8; 8 * 1024] = [0; 8 * 1024];

/// The classes of the model, in the order of its outputs.
const LABELS: [&str; 3] = ["idle", "shake", "circle"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: `main` runs once and is the only user of `MEMORY`.
    let arena = Arena::new(unsafe { &mut *core::ptr::addr_of_mut!(MEMORY) });

    // Load the model: verify it, choose a kernel for each operator, plan the memory.
    let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena)?;
    println!(
        "model {} B, {} operators, graph {} B, context {} B",
        MODEL.0.len(),
        graph.operator_count(),
        graph.graph_bytes(),
        graph.context_bytes()
    );
    for i in 0..graph.operator_count() {
        println!("  operator {i}: {}", graph.kernel_name(i).unwrap_or("?"));
    }
    let input = graph.input(0)?;
    let (scale, zero_point) = (input.quant.scale(), input.quant.zero_point());
    println!(
        "input: {:?} {:?}, scale {scale}, zero point {zero_point}",
        input.dtype, input.dims
    );

    // The directory of the test vectors, from the command line.
    let vectors = std::env::args()
        .nth(1)
        .ok_or("usage: gesture VECTORS_DIRECTORY")?;

    // The execution context holds the activations of one inference.
    let mut ctx = graph.init_execution_context_in(&arena)?;
    println!("arena: {} of {} bytes used", arena.used(), arena.capacity());

    let mut samples = 0;
    let mut matches = 0;
    while let Ok(window) = std::fs::read(format!("{vectors}/sample_{samples}/window.bin")) {
        // The accelerometer window: 50 samples of 3 axes, float32, in g.
        let window = window
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b));

        // Quantize it into the input tensor: q = round(x / scale) + zero_point.
        let mut tensor = ctx.input_mut(0)?;
        let quantized = tensor.as_mut_slice::<i8>()?;
        for (q, x) in quantized.iter_mut().zip(window) {
            *q = ((x / scale).round() as i32 + zero_point).clamp(-128, 127) as i8;
        }
        let expected = std::fs::read(format!("{vectors}/sample_{samples}/input_0.bin"))?;
        let same_input = quantized.iter().map(|&q| q as u8).eq(expected);

        ctx.compute()?;

        // Dequantize the best score: probability = scale * (q - zero_point).
        let output = ctx.output(0)?;
        let quantization = output.quantization();
        let scores = output.as_slice::<i8>()?;
        // The first of the highest scores, as NumPy's `argmax`.
        let best = (1..scores.len()).fold(0, |b, i| if scores[i] > scores[b] { i } else { b });
        let probability =
            quantization.scale() * (i32::from(scores[best]) - quantization.zero_point()) as f32;

        let expected = std::fs::read(format!("{vectors}/sample_{samples}/output_0.bin"))?;
        let same = same_input && scores.iter().map(|&q| q as u8).eq(expected);
        println!(
            "sample_{samples}: {scores:4?} -> {:6} {probability:.3}  {}",
            LABELS.get(best).copied().unwrap_or("?"),
            if same {
                "same as LiteRT"
            } else {
                "DIFFERS from LiteRT"
            }
        );
        samples += 1;
        matches += usize::from(same);
    }

    println!("{matches} of {samples} samples match LiteRT");
    if samples == 0 || matches != samples {
        return Err("the outputs are not LiteRT's".into());
    }
    Ok(())
}
