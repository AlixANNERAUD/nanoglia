# nanoglia-tflite

A reader for TensorFlow Lite (`.tflite`) models, used by [nanoglia](https://github.com/AlixANNERAUD/nanoglia):

- **zero-copy**: tensor shapes, quantization parameters, operator inputs and constant data are views into the model bytes;
- **allocation-free**: `#![no_std]`, no `alloc`, a small flatbuffer reader of its own instead of the `flatbuffers` crate;
- **panic-free on untrusted input**: [`Model::parse`] verifies the whole flatbuffer once (offsets, vtables, vectors, strings, indices), and every accessor stays bounds-checked afterwards;
- **no alignment assumption**: metadata is read byte by byte, so the model can live anywhere in flash or RAM.

Tensors and operators are converted into the [`nanoglia_core`] vocabulary: [`TensorDesc`](nanoglia_core::TensorDesc),
[`OpCode`](nanoglia_core::OpCode) and [`OpOptions`](nanoglia_core::OpOptions).

```rust,no_run
# fn main() -> nanoglia_core::Result<()> {
# let bytes: &[u8] = &[];
use nanoglia_tflite::Model;

let model = Model::parse(bytes)?;
let graph = model.subgraph(0)?;
for i in 0..graph.operator_count() {
    let op = graph.operator(i)?;
    for input in op.inputs.iter().filter(|&t| t >= 0) {
        let tensor = graph.tensor(input as usize)?;
        let _ = (tensor.dtype, tensor.dims, tensor.data);
    }
}
# Ok(())
# }
```
