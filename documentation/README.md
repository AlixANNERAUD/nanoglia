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

## How-to guides

| Page | Goal |
| --- | --- |
| [Size the memory](how-to-guides/size-memory.md) | Compute the exact size of a static `Arena` with `MemoryRequirements`, guard it with a test, confirm it on the device |
| [Convert a PyTorch model](how-to-guides/convert-a-pytorch-model.md) | Produce a full-integer `.tflite` file with litert-torch (or onnx2tf from ONNX) and check it against LiteRT |
| [Run on a microcontroller](how-to-guides/run-on-a-microcontroller.md) | Run the Nano 33 BLE example, then use nanoglia without a heap in your own Cortex-M or ESP32-S3 firmware |
| [Run in the browser](how-to-guides/run-in-the-browser.md) | Build for `wasm32-unknown-unknown` with CMSIS-NN, enable SIMD128, run under wasmtime |
| [Run long inferences](how-to-guides/run-long-inferences.md) | Run an inference one operator at a time with `step`, as a future with `compute_async`, or stop it at a deadline from an observer |
| [Write a backend](how-to-guides/write-a-backend.md) | Add a backend whose kernels use a driver it borrows, register it with `Graph::load_with`, time each operator with `compute_with` |

## Explanation

| Page | Question it answers |
| --- | --- |
| [Architecture](explanation/architecture.md) | How the crates fit, what loading a model does, how kernels are chosen, why nothing is copied |
| [Bit-exactness](explanation/bit-exactness.md) | Why outputs match TFLite bit for bit, how this is tested, where LiteRT and TensorFlow Lite Micro disagree, why TFLite and not ONNX |

## Reference

| Page | Contents |
| --- | --- |
| [Supported operators](reference/supported-operators.md) | Every operator and type, the backend and kernel that runs it, and their restrictions |
| [Cargo features](reference/features.md) | Every feature of every crate and what it adds or removes |
| [Errors](reference/errors.md) | `Error`, `ErrorCode`, every `Detail` variant and the functions that return them |
| [Memory](reference/memory.md) | The allocation contract, alignment, what the blocks hold, the footprints of the MLPerf Tiny models |
| [Performance](reference/performance.md) | Host and wasmtime timings of the MLPerf Tiny models, with the machine they come from |
| [Development tasks](reference/development-tasks.md) | The mise tasks and the CI workflows that run them |
