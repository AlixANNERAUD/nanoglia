//! The bound on the 16x8 accumulators of the CMSIS-NN backend (issue #38: CMSIS-NN v8.0.0 keeps
//! `acc * multiplier >> (14 - shift)` in an `int32_t`): a fully-connected layer beyond it runs on
//! the Rust backend, a convolution beyond it does not load (there is no Rust 16x8 convolution).

#![cfg(all(feature = "cmsis-nn", feature = "kernels", feature = "int16"))]

use nanoglia::core::allocator_api2::alloc::Global;
use nanoglia::core::backend::{EvalCtx, Kernel};
use nanoglia::core::le::LeSlice;
use nanoglia::core::{BuiltinOp, FullyConnectedOptions, OpDesc, OpOptions};
use nanoglia::{
    Activation, DataType, Detail, Dims, Encoding, ErrorCode, Graph, OpCode, Quantization, Registry,
    Target, TensorDesc,
};

mod common;

/// A 16-byte aligned, zero-initialised byte buffer.
struct Buffer {
    storage: Vec<u128>,
    len: usize,
}

impl Buffer {
    fn new(bytes: &[u8]) -> Self {
        let mut buffer = Self {
            storage: vec![0; bytes.len().div_ceil(16)],
            len: bytes.len(),
        };
        buffer.bytes_mut().copy_from_slice(bytes);
        buffer
    }

    fn zeroed(len: usize) -> Self {
        Self::new(&vec![0; len])
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: `storage` holds at least `len` initialised bytes.
        unsafe { std::slice::from_raw_parts(self.storage.as_ptr().cast(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `storage` holds at least `len` initialised bytes.
        unsafe { std::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast(), self.len) }
    }
}

/// A tensor with the per-tensor quantization `quant` (scale then zero point, little-endian), or
/// none if `quant` is empty.
fn tensor<'a>(
    index: u32,
    dtype: DataType,
    dims: &[i32],
    quant: &'a [u8],
    data: Option<&'a [u8]>,
) -> TensorDesc<'a> {
    TensorDesc {
        index,
        dtype,
        dims: Dims::new(dims).unwrap(),
        quant: Quantization {
            scale: LeSlice::new(quant.get(..4).unwrap_or(&[])),
            zero_point: LeSlice::new(quant.get(4..).unwrap_or(&[])),
            axis: 0,
        },
        data,
        name: "",
    }
}

/// The kernel the default registry picks for a 16x8 `FULLY_CONNECTED` of one input `-32768`,
/// weight `-128` (scale 0.5: multiplier 2^30, shift 0) and bias `bias` (accumulator
/// `2^22 + bias`), and its output.
fn fully_connected(bias: i64) -> (&'static str, i16) {
    let quant = |scale: f32| [&scale.to_le_bytes()[..], &0i64.to_le_bytes()].concat();
    let (unit, half) = (quant(1.0), quant(0.5));
    let weights = Buffer::new(&[(-128i8) as u8]);
    let bias = Buffer::new(&bias.to_le_bytes());
    let inputs = [
        Some(tensor(0, DataType::I16, &[1, 1], &unit, None)),
        Some(tensor(
            1,
            DataType::I8,
            &[1, 1],
            &half,
            Some(weights.bytes()),
        )),
        Some(tensor(2, DataType::I64, &[1], &[], Some(bias.bytes()))),
    ];
    let outputs = [tensor(3, DataType::I16, &[1, 1], &unit, None)];
    let op = OpDesc {
        index: 0,
        code: OpCode::Builtin(BuiltinOp::FULLY_CONNECTED),
        options: OpOptions::FullyConnected(FullyConnectedOptions {
            activation: Activation::None,
            default_weights_format: true,
            keep_num_dims: false,
            asymmetric_quantize_inputs: false,
        }),
        inputs: &inputs,
        outputs: &outputs,
    };
    let kernel: &dyn Kernel = Registry::default()
        .backends()
        .iter()
        .find_map(|b| b.select(&op))
        .expect("no kernel");
    let plan = kernel.plan(&op).unwrap();
    let mut persist = Buffer::zeroed(plan.persistent);
    kernel.prepare(&op, persist.bytes_mut()).unwrap();
    let mut scratch = Buffer::zeroed(plan.scratch);
    let input = Buffer::new(&i16::MIN.to_le_bytes());
    let mut output = Buffer::zeroed(2);
    let mut outputs = [output.bytes_mut()];
    kernel
        .eval(&mut EvalCtx {
            inputs: &[input.bytes(), weights.bytes(), bias.bytes()],
            outputs: &mut outputs,
            persist: persist.bytes(),
            scratch: scratch.bytes_mut(),
        })
        .unwrap();
    let value = i16::from_le_bytes(output.bytes().try_into().unwrap());
    (kernel.name(), value)
}

#[test]
fn fully_connected_beyond_the_bound_runs_on_the_rust_backend() {
    // Accumulator 2^31 - 2: CMSIS-NN's `int32_t` intermediate holds `2^31 - 2`.
    assert_eq!(
        fully_connected((1 << 31) - 2 - (1 << 22)),
        ("cmsis-nn/fully_connected_s16", 32767)
    );
    // Accumulator 2^31: CMSIS-NN v8.0.0 would give -32768, TFLite saturates to 32767.
    assert_eq!(
        fully_connected((1 << 31) - (1 << 22)),
        ("rust/fully_connected_16x8", 32767)
    );
}

#[test]
fn convolution_beyond_the_bound_does_not_load() {
    let path = common::repo_root().join("testdata/ops/conv2d_3x3_same_int16/model.tflite");
    let model = common::Aligned::read(&path);
    let original = model.bytes();
    let load = |bytes: &[u8]| {
        let model = Buffer::new(bytes);
        Graph::load_in(&[model.bytes()], Encoding::Autodetect, Target::Auto, Global)
            .map(|graph| graph.kernel_name(0))
    };
    assert_eq!(load(original).unwrap(), Some("cmsis-nn/conv2d_s16"));

    // The offset of the convolution's int64 bias in the model.
    let bias_offset = {
        let model = nanoglia_tflite::Model::parse(original).unwrap();
        let subgraph = model.subgraph(0).unwrap();
        assert_eq!(subgraph.operator_count(), 1);
        let op = subgraph.operator(0).unwrap();
        let bias = subgraph.tensor(op.inputs.get(2).unwrap() as usize).unwrap();
        assert_eq!(bias.dtype, DataType::I64);
        bias.data.unwrap().as_ptr() as usize - original.as_ptr() as usize
    };
    let mut patched = original.to_vec();
    patched[bias_offset..bias_offset + 8].copy_from_slice(&(1i64 << 50).to_le_bytes());
    let error = load(&patched).unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedOperation);
    assert!(matches!(
        error.detail,
        Detail::UnsupportedOp { op_index: 0, .. }
    ));
}
