//! Builds operator descriptions by hand and runs a selected kernel through plan, prepare and
//! eval with buffers owned by the test, once on ESP-NN and once on its CMSIS-NN fallback.

use std::sync::Mutex;

use nanoglia_core::backend::{Backend, EvalCtx};
use nanoglia_core::le::LeSlice;
use nanoglia_core::{
    Activation, BuiltinOp, DataType, Dims, OpCode, OpDesc, OpOptions, Quantization, TensorDesc,
};
use nanoglia_esp_nn::BACKEND;

/// Deterministic pseudo-random numbers (xorshift64*).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A value in `lo..=hi`.
    pub fn range(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi as i64 - lo as i64 + 1) as u64;
        (lo as i64 + (self.next_u64() % span) as i64) as i32
    }

    pub fn i8s(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.range(-128, 127) as i8).collect()
    }

    pub fn i32s(&mut self, n: usize, lo: i32, hi: i32) -> Vec<i32> {
        (0..n).map(|_| self.range(lo, hi)).collect()
    }

    /// A float in `[lo, hi)`.
    pub fn float(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

/// A byte buffer aligned to 16 bytes.
#[derive(Clone)]
pub struct Aligned {
    words: Vec<u128>,
    len: usize,
}

impl Aligned {
    pub fn zeroed(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(16).max(1)],
            len,
        }
    }

    pub fn filled(len: usize, byte: u8) -> Self {
        let mut out = Self::zeroed(len);
        out.bytes_mut().fill(byte);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut out = Self::zeroed(bytes.len());
        out.bytes_mut().copy_from_slice(bytes);
        out
    }

    pub fn from_i8(values: &[i8]) -> Self {
        Self::from_bytes(&values.iter().map(|&v| v as u8).collect::<Vec<_>>())
    }

    pub fn from_i32(values: &[i32]) -> Self {
        Self::from_bytes(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: `words` holds at least `len` initialized bytes.
        unsafe { core::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, and the borrow is unique.
        unsafe { core::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast::<u8>(), self.len) }
    }

    pub fn as_i8(&self) -> Vec<i8> {
        self.bytes().iter().map(|&b| b as i8).collect()
    }
}

/// An owned tensor from which a `TensorDesc` borrows.
pub struct Tensor {
    pub dtype: DataType,
    pub dims: Vec<i32>,
    scales: Vec<u8>,
    zero_points: Vec<u8>,
    axis: i32,
    /// Contents; constant tensors expose them in their description.
    pub data: Aligned,
    pub constant: bool,
}

impl Tensor {
    /// An activation tensor with per-tensor quantization.
    pub fn activation(dtype: DataType, dims: &[i32], scale: f32, zero_point: i32) -> Self {
        let len: i32 = dims.iter().product();
        let size = dtype.bytes_for(len as usize).unwrap();
        Self {
            dtype,
            dims: dims.to_vec(),
            scales: scale.to_le_bytes().to_vec(),
            zero_points: (zero_point as i64).to_le_bytes().to_vec(),
            axis: 0,
            data: Aligned::zeroed(size),
            constant: false,
        }
    }

    /// A constant int8 weight tensor with one scale per entry of `scales` along `axis`.
    pub fn weights(dims: &[i32], values: &[i8], scales: &[f32], axis: i32) -> Self {
        assert_eq!(values.len() as i32, dims.iter().product::<i32>());
        Self {
            dtype: DataType::I8,
            dims: dims.to_vec(),
            scales: scales.iter().flat_map(|s| s.to_le_bytes()).collect(),
            zero_points: scales.iter().flat_map(|_| 0i64.to_le_bytes()).collect(),
            axis,
            data: Aligned::from_i8(values),
            constant: true,
        }
    }

    /// A constant int32 bias tensor.
    pub fn bias(values: &[i32]) -> Self {
        Self {
            dtype: DataType::I32,
            dims: vec![values.len() as i32],
            scales: Vec::new(),
            zero_points: Vec::new(),
            axis: 0,
            data: Aligned::from_i32(values),
            constant: true,
        }
    }

    pub fn with_data_i8(mut self, values: &[i8]) -> Self {
        self.data = Aligned::from_i8(values);
        self
    }

    pub fn desc(&self, index: u32) -> TensorDesc<'_> {
        TensorDesc {
            index,
            dtype: self.dtype,
            dims: Dims::new(&self.dims).unwrap(),
            quant: Quantization {
                scale: LeSlice::new(&self.scales),
                zero_point: LeSlice::new(&self.zero_points),
                axis: self.axis,
            },
            data: self.constant.then(|| self.data.bytes()),
            name: "",
        }
    }
}

/// An operator over owned tensors.
pub struct Op<'t> {
    pub code: BuiltinOp,
    pub options: OpOptions<'static>,
    pub inputs: Vec<Option<&'t Tensor>>,
    pub outputs: Vec<&'t Tensor>,
}

impl<'t> Op<'t> {
    pub fn new(code: BuiltinOp, options: OpOptions<'static>) -> Self {
        Self {
            code,
            options,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn input(mut self, t: &'t Tensor) -> Self {
        self.inputs.push(Some(t));
        self
    }

    pub fn optional(mut self, t: Option<&'t Tensor>) -> Self {
        self.inputs.push(t);
        self
    }

    pub fn output(mut self, t: &'t Tensor) -> Self {
        self.outputs.push(t);
        self
    }

    /// Calls `f` with the operator description.
    pub fn with_desc<R>(&self, f: impl FnOnce(&OpDesc<'_, '_>) -> R) -> R {
        let inputs: Vec<Option<TensorDesc<'_>>> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(i, t)| t.map(|t| t.desc(i as u32)))
            .collect();
        let outputs: Vec<TensorDesc<'_>> = self
            .outputs
            .iter()
            .enumerate()
            .map(|(i, t)| t.desc(100 + i as u32))
            .collect();
        let desc = OpDesc {
            index: 0,
            code: OpCode::Builtin(self.code),
            options: self.options,
            inputs: &inputs,
            outputs: &outputs,
        };
        f(&desc)
    }

    /// Whether the ESP-NN backend claims the operator.
    pub fn selected(&self) -> bool {
        self.with_desc(|desc| BACKEND.select(desc).is_some())
    }

    /// Runs the operator with the ESP-NN backend and returns the first output, of type int8.
    pub fn run(&self) -> Vec<i8> {
        self.run_bytes().as_i8()
    }

    /// The name of the kernel the ESP-NN backend selects.
    pub fn kernel_name(&self) -> &'static str {
        self.with_desc(|desc| BACKEND.select(desc).expect("refused").name())
    }

    /// Runs the operator with the ESP-NN backend, once with the ESP-NN lock free (ESP-NN runs)
    /// and once with the lock held (the CMSIS-NN fallback runs), checks that both give the same
    /// bytes and returns them.
    pub fn run_bytes(&self) -> Aligned {
        // The tests of this binary run in parallel threads: serialize the evaluations so that
        // the lock is free when a test wants ESP-NN to run.
        static SERIAL: Mutex<()> = Mutex::new(());
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        self.with_desc(|desc| {
            let kernel = BACKEND
                .select(desc)
                .expect("the backend refused the operator");
            let plan = kernel.plan(desc).expect("plan");
            let mut persist = Aligned::zeroed(plan.persistent);
            kernel.prepare(desc, persist.bytes_mut()).expect("prepare");
            let eval = || {
                // Garbage in the scratch buffer: kernels must not rely on its contents.
                let mut scratch = Aligned::filled(plan.scratch, 0xA5);
                let inputs: Vec<&[u8]> = self
                    .inputs
                    .iter()
                    .map(|t| t.map_or(&[][..], |t| t.data.bytes()))
                    .collect();
                let mut output = Aligned::filled(self.outputs[0].data.bytes().len(), 0x5A);
                {
                    let mut outputs = [output.bytes_mut()];
                    let mut ctx = EvalCtx {
                        inputs: &inputs,
                        outputs: &mut outputs,
                        persist: persist.bytes(),
                        scratch: scratch.bytes_mut(),
                    };
                    kernel.eval(&mut ctx).expect("eval");
                }
                output
            };
            drop(nanoglia_esp_nn::try_lock().expect("the ESP-NN lock is free"));
            let esp_nn = eval();
            let lock = nanoglia_esp_nn::try_lock().expect("the ESP-NN lock is free");
            let fallback = eval();
            drop(lock);
            assert_eq!(
                esp_nn.bytes(),
                fallback.bytes(),
                "ESP-NN and its CMSIS-NN fallback differ"
            );
            esp_nn
        })
    }
}

/// All four fused activations ESP-NN supports.
pub const ACTIVATIONS: [Activation; 4] = [
    Activation::None,
    Activation::Relu,
    Activation::Relu6,
    Activation::ReluN1To1,
];

/// Checks that an expected output takes enough distinct values for the comparison to exercise
/// the requantization rather than only the clamps.
pub fn assert_spread(values: &[i8]) {
    let mut distinct = values.to_vec();
    distinct.sort();
    distinct.dedup();
    assert!(
        distinct.len() > values.len().min(8) / 2,
        "degenerate output: {distinct:?}"
    );
}
