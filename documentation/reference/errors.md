# Errors

Every fallible function of nanoglia returns `nanoglia::Result<T>`, whose
error is `nanoglia::Error` (defined in `nanoglia-core`, re-exported by
`nanoglia`):

```rust
pub struct Error {
    pub code: ErrorCode,
    pub detail: Detail,
}
```

`Error` is `Copy`, `PartialEq`, `Eq` and `Debug`, never allocates, and
implements `Display` and `core::error::Error`. `Display` prints the code,
then the detail when there is one: `too large: OutOfMemory { requested:
16160 }`. With feature `defmt`, `Error`, `ErrorCode` and `Detail`
implement `defmt::Format`.

## ErrorCode

The category of the error. The variants are those of the `error-code`
enum of [wasi-nn](https://github.com/WebAssembly/wasi-nn), so a wasi-nn
host built on nanoglia can forward them unchanged. `ErrorCode` is
`Copy`, `Eq` and `Hash`; its `Display` text is in the last column.

| Variant | Meaning | `Display` |
| --- | --- | --- |
| `InvalidArgument` | The caller passed an invalid argument (wrong tensor index, type, shape or size), or a constant buffer is misaligned. | invalid argument |
| `InvalidEncoding` | The bytes are not a valid model of the requested encoding. | invalid encoding |
| `Timeout` | An operation, typically on an accelerator, timed out. Not returned yet. | timeout |
| `RuntimeError` | A kernel library failed while computing. | runtime error |
| `UnsupportedOperation` | The model uses an operator, type or structure no enabled backend supports. | unsupported operation |
| `TooLarge` | The allocator could not provide the memory the graph or context needs. | too large |
| `NotFound` | No input or output has the requested name. | not found |
| `Security` | The operation is not allowed, or a hardware feature cannot be accessed. Not returned yet. | security |
| `Unknown` | Unspecified failure (an internal inconsistency). | unknown |

## Detail

The precise reason, with the indices that locate it in the model.
`Detail` is `#[non_exhaustive]`: match it with a `_` arm.

| Variant | Usual code | Meaning |
| --- | --- | --- |
| `None` | any | No further detail. |
| `MalformedModel { offset }` | `InvalidEncoding` | The model failed verification at this byte offset (0 for errors in the dataflow of the graph). |
| `UnsupportedSchema { version }` | `UnsupportedOperation` | The TFLite schema version is not 3. |
| `UnknownEncoding` | `InvalidEncoding` | The bytes do not carry the `TFL3` identifier. |
| `UnsupportedStructure` | `UnsupportedOperation` | Several subgraphs, variable tensors, more than 16 inputs or 8 outputs on an operator, or more than `u32::MAX` operators or tensors. |
| `Misaligned { tensor }` | `InvalidArgument` | A constant buffer is not aligned for its element type. `tensor` is `u32::MAX` when the buffer is not a tensor of the model (a typed view, the scratch buffer). |
| `UnsupportedOp { op_index, opcode }` | `UnsupportedOperation` | No backend supports this operator with these parameters; `opcode` is the TFLite builtin code (`32` for a custom operator). |
| `UnsupportedType { tensor }` | `UnsupportedOperation` or `InvalidArgument` | The model reader cannot represent the tensor, or a typed view was asked for the wrong element type. |
| `ShapeMismatch { tensor }` | `UnsupportedOperation` or `InvalidArgument` | A tensor has a dynamic shape, or not the shape an operator expects. |
| `SizeMismatch { expected, actual }` | `InvalidArgument` | A buffer does not have the expected size, in bytes. |
| `IndexOutOfRange { index }` | `InvalidArgument` | An input, output, tensor, operator or subgraph index is out of range. |
| `OutOfMemory { requested }` | `TooLarge` | The allocator refused an allocation of `requested` bytes (`usize::MAX` when a size computation overflowed). |
| `TensorNotFound` | `NotFound` | No input or output has the requested name. |
| `Backend { backend, status }` | `RuntimeError` | A C kernel library returned a failure status (`backend` is `"cmsis-nn"`). |

## Which function returns what

### Loading: `Graph::load`, `load_in`, `load_with`, and `MemoryRequirements::compute`, `compute_in`, `compute_with`

| Error | When | What to do |
| --- | --- | --- |
| `InvalidArgument` / `None` | `builders` does not hold exactly one buffer. | Pass `&[model_bytes]`. |
| `InvalidEncoding` / `UnknownEncoding` | Not a TFLite flatbuffer (with `Encoding::Autodetect` or `Encoding::TensorFlowLite`). | Check that you pass the `.tflite` file. |
| `InvalidEncoding` / `MalformedModel` | The flatbuffer is truncated or corrupted, or the graph consumes a tensor before producing it, produces it twice, or writes a constant. | Re-export the model; compare the bytes with the file. |
| `UnsupportedOperation` / `UnsupportedSchema` | Schema version other than 3. | Convert with a current LiteRT converter. |
| `UnsupportedOperation` / `UnsupportedStructure` | See the table above. | Export a single-subgraph model without state (no control flow, no variables). |
| `UnsupportedOperation` / `UnsupportedType` | A tensor type nanoglia cannot represent (string, complex, resource...), rank above 6, sparse tensor, data in an external file, non-affine quantization. | Change the export. |
| `UnsupportedOperation` / `UnsupportedOp` | No backend claims operator `op_index`: operator, types, shapes or options outside [Supported operators](supported-operators.md), or a feature that would provide it is off ([Cargo features](features.md)). Option values the reader does not know give the same error. | Check the matrix; enable the feature; change the model. |
| `UnsupportedOperation` / `ShapeMismatch` | An activation tensor has a dynamic (negative) dimension. | Export with static shapes. |
| `UnsupportedOperation` / `None` | `Target::Npu` and no operator runs on an NPU backend. | Use `Target::Auto` or `Target::Cpu`. |
| `InvalidArgument` / `Misaligned` | A constant buffer of the model is not aligned for its type. | Place the model bytes on a 16-byte boundary ([Memory](memory.md#model-bytes)). |
| `TooLarge` / `OutOfMemory` | The allocator refused the graph block (`load*`) or the planner's working memory (`compute*`). | Give a larger arena ([Size the memory](../how-to-guides/size-memory.md)). |

A kernel's `prepare` can also fail while loading (for example
`InvalidArgument` / `SizeMismatch` when a bias buffer is shorter than its
shape says). `MemoryRequirements::compute*` does not run `prepare`, so only
`load*` reports these.

### Creating a context: `Graph::init_execution_context`, `init_execution_context_in`

| Error | When |
| --- | --- |
| `TooLarge` / `OutOfMemory { requested }` | The allocator could not provide `context_bytes()` bytes. |

### Graph queries

| Function | Error |
| --- | --- |
| `Graph::input`, `Graph::output` | `InvalidArgument` / `IndexOutOfRange` |
| `Graph::input_index`, `Graph::output_index` | `NotFound` / `TensorNotFound` |

### Execution context

| Function | Error | When |
| --- | --- | --- |
| `input_mut` | `InvalidArgument` / `IndexOutOfRange` | No such input, or the input is not an activation (a constant or unused tensor). |
| `set_input` | as `input_mut`, and `InvalidArgument` / `SizeMismatch` | The data length differs from the tensor size in bytes. |
| `output` | `InvalidArgument` / `IndexOutOfRange` | No such output. |
| `output_by_name` | `NotFound` / `TensorNotFound` | No output has this name. |
| `compute` | `RuntimeError` / `Backend` | CMSIS-NN returned a failure status. |
| `compute` | `InvalidArgument` / `SizeMismatch`, `Misaligned` | A kernel found buffers that do not match its prepared parameters. This does not happen with buffers laid out by `Graph`. |

### Tensor views: `TensorRef::as_slice`, `TensorMut::as_mut_slice`

| Error | When |
| --- | --- |
| `InvalidArgument` / `UnsupportedType { tensor }` | The Rust element type does not match the tensor's `DataType` (for example `as_slice::<f32>()` on an int8 output). |
| `InvalidArgument` / `Misaligned { tensor: u32::MAX }` | The bytes are not aligned for the element type. Arena tensors are always 16-byte aligned. |

## Handling errors

```rust
use nanoglia::{Detail, ErrorCode};

match nanoglia::Graph::load(&[model], nanoglia::Encoding::Autodetect, nanoglia::Target::Auto) {
    Ok(graph) => { /* ... */ }
    Err(e) if e.code == ErrorCode::UnsupportedOperation => {
        if let Detail::UnsupportedOp { op_index, opcode } = e.detail {
            // Report which operator of the model is not supported.
        }
    }
    Err(e) => { /* `e` implements Display */ }
}
```
