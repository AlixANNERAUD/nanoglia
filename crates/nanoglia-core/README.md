# nanoglia-core

**The `no_std` building blocks of [nanoglia](https://github.com/AlixANNERAUD/nanoglia), the TensorFlow Lite inference engine for microcontrollers:** tensor and operator types, TFLite-exact fixed-point quantization math, a kernel-backend interface and an arena allocator. Use it to write your own kernel backend; Most applications depend on [`nanoglia`](https://crates.io/crates/nanoglia) instead, which enables this crate through its Cargo features.

Types shared by the [nanoglia](https://github.com/AlixANNERAUD/nanoglia) crates:

- tensors ([`TensorDesc`], [`DataType`], [`Dims`], [`Quantization`]) and operators ([`OpDesc`], [`BuiltinOp`], [`OpOptions`]), using the TFLite vocabulary;
- the [`Error`] type, whose [`ErrorCode`] matches wasi-nn;
- the [`backend::Backend`] / [`backend::Kernel`] interface implemented by kernel libraries: a backend lends out kernels borrowed from itself, so they can use its state; [`backend::FnKernel`] is a stateless kernel made of three functions;
- TFLite-exact fixed-point [`quant`]ization math;
- [`Arena`], a bump allocator over a caller buffer, and [`Block`], the only kind of allocation nanoglia makes.

`#![no_std]`, no allocation of its own: every byte comes from the allocator the caller passes.
