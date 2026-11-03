# nanonn-core

Types shared by the [nanonn](https://github.com/AlixANNERAUD/nanonn) crates:

- tensors ([`TensorDesc`], [`DataType`], [`Dims`], [`Quantization`]) and operators ([`OpDesc`], [`BuiltinOp`], [`OpOptions`]), using the TFLite vocabulary;
- the [`Error`] type, whose [`ErrorCode`] matches wasi-nn;
- the [`backend::Backend`] / [`backend::Kernel`] interface implemented by kernel libraries;
- TFLite-exact fixed-point [`quant`]ization math;
- [`Arena`], a bump allocator over a caller buffer, and [`Block`], the only kind of allocation nanonn makes.

`#![no_std]`, no allocation of its own: every byte comes from the allocator the caller passes.
