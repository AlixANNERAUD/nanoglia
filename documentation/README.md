# nanoglia documentation

User documentation of nanoglia, which runs TensorFlow Lite (LiteRT)
models loaded at run time on microcontrollers, through one `no_std` Rust
API and the best kernels of each chip. The pages follow the
[Diátaxis](https://diataxis.fr) structure:

- **Tutorials** teach by doing: follow one from start to finish.
- **How-to guides** reach one goal, in steps, for readers who know what
  they want.
- **Explanation** says why nanoglia works the way it does.
- **Reference** describes each item, input and number, to look up.

The API documentation of each item is on
[docs.rs](https://docs.rs/nanoglia) once the crates are published; until
then, run `cargo doc --open` in your project. The rules for contributors
are in [`AGENTS.md`](../AGENTS.md).

## Tutorials

| Page | What you build |
| --- | --- |
| [Your first inference](tutorials/first-inference.md) | A host program that runs the MLPerf Tiny keyword spotting model, reads its scores, then runs it again from a static buffer without a heap |
| [From training to the device](tutorials/train-and-deploy.md) | A gesture classifier trained with Keras or PyTorch, exported to an int8 `.tflite` model with test vectors, checked, then run from Rust and from C with the same outputs as LiteRT |

## How-to guides

| Page | Goal |
| --- | --- |
| [Check whether a model runs](how-to-guides/check-whether-a-model-runs.md) | List, on your computer, the kernel of every operator of a `.tflite` file, every operator that blocks and why, and the memory needed, with the `nanoglia check` command or `ModelReport` |
| [Size the memory](how-to-guides/size-memory.md) | Compute the exact size of a static `Arena` with `MemoryRequirements`, guard it with a test, confirm it on the device |
| [Convert a PyTorch model](how-to-guides/convert-a-pytorch-model.md) | Produce a full-integer `.tflite` file with litert-torch (or onnx2tf from ONNX, or TensorFlow's converter from Keras) and check it against LiteRT |
| [Run on a microcontroller](how-to-guides/run-on-a-microcontroller.md) | Run the Nano 33 BLE example, then use nanoglia without a heap in your own Cortex-M or ESP32-S3 firmware |
| [Run the golden cases on a board](how-to-guides/run-the-golden-cases-on-a-board.md) | Flash an Arduino Nano 33 BLE over USB, without a debug probe, run every golden case on it, compare with LiteRT and record the time of each operator of the models kept in flash; the first session on a new board, step by step |
| [Reduce the code size](how-to-guides/reduce-code-size.md) | Link only the kernels of the operators your models use, with the `op-*` Cargo features, and check that the models still load |
| [Run in the browser](how-to-guides/run-in-the-browser.md) | Build for `wasm32-unknown-unknown` with CMSIS-NN, enable SIMD128, run under wasmtime |
| [Run long inferences](how-to-guides/run-long-inferences.md) | Run an inference one operator at a time with `step`, as a future with `compute_async`, or stop it at a deadline from an observer |
| [Write a backend](how-to-guides/write-a-backend.md) | Add a backend whose kernels use a driver it borrows, register it with `Graph::load_with`, time each operator with `compute_with` |
| [Use nanoglia from C](how-to-guides/use-from-c.md) | Build the C library with CMake, link it on your computer or into a firmware, size its one buffer, put the model in the program, run inferences in one call or step by step, handle the errors |

## Explanation

| Page | Question it answers |
| --- | --- |
| [Architecture](explanation/architecture.md) | How the crates fit, what loading a model does, how kernels are chosen, why nothing is copied |
| [Bit-exactness](explanation/bit-exactness.md) | Why outputs match TFLite bit for bit, how this is tested, where LiteRT and TensorFlow Lite Micro disagree, why TFLite and not ONNX |

## Reference

| Page | Contents |
| --- | --- |
| [Supported operators](reference/supported-operators.md) | Every operator and type, the backend and kernel that runs it, and their restrictions |
| [Cargo features](reference/features.md) | Every feature of every crate and what it adds, with the feature of each operator |
| [Errors](reference/errors.md) | `Error`, `ErrorCode`, every `Detail` variant and the functions that return them |
| [Memory](reference/memory.md) | The allocation contract, alignment, what the blocks hold, the footprints of the MLPerf Tiny models, the code size of a Cortex-M4F firmware by feature set, the flash and RAM of the Nano 33 BLE test bench |
| [Performance](reference/performance.md) | Host and wasmtime timings of the MLPerf Tiny models, with the machine they come from; comparison with the other Rust engines (outputs, host time, firmware size and RAM) |
| [Development tasks](reference/development-tasks.md) | The mise tasks and the CI workflows that run them |
