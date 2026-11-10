//! Runs the check of the library and keeps what it reports.

use crate::args::Backends;
use nanoglia::core::backend::Backend;
use nanoglia::{
    Arena, BuiltinOp, DataType, Detail, Encoding, Error, ErrorCode, MemoryBreakdown, ModelReport,
    OpCode, OperatorReport, Registry, SelectedKernel, Target, TensorDesc,
};

/// A tensor of an operator.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Index of the tensor in the model.
    pub index: u32,
    /// TFLite name of the element type (`int8`, `float32`...).
    pub dtype: &'static str,
    pub shape: Vec<i32>,
    /// Whether the model holds its data (weights, biases, axes...).
    pub constant: bool,
}

/// An input or output of an operator.
#[derive(Clone, Debug)]
pub enum Slot {
    Tensor(Tensor),
    /// An optional input the model leaves out.
    Omitted,
    /// A tensor nanoglia cannot represent.
    Unknown(Error),
}

/// One operator of the model.
#[derive(Clone, Debug)]
pub struct Operator {
    pub index: usize,
    /// TFLite builtin operator code (32 for a custom operator).
    pub opcode: i32,
    /// Name of the builtin operator in the TFLite schema, or the custom code.
    pub name: String,
    pub custom: bool,
    pub version: i32,
    pub inputs: Vec<Slot>,
    pub outputs: Vec<Slot>,
    /// The kernel that runs it, or why none does.
    pub kernel: Result<SelectedKernel, Error>,
    /// The `op-*` feature this build of the tool lacks, when that is why no kernel runs the
    /// operator.
    pub missing_feature: Option<&'static str>,
}

/// What the library reports for a model.
#[derive(Clone, Debug)]
pub struct Report {
    pub operators: Vec<Operator>,
    /// Number of operators without a kernel.
    pub unsupported: usize,
    /// What loading allocates, or why the model does not load.
    pub memory: Result<MemoryBreakdown, Error>,
}

/// The TFLite name of a tensor type.
fn dtype_name(dtype: DataType) -> &'static str {
    match dtype {
        DataType::F32 => "float32",
        DataType::F16 => "float16",
        DataType::I32 => "int32",
        DataType::U8 => "uint8",
        DataType::I64 => "int64",
        DataType::Bool => "bool",
        DataType::I16 => "int16",
        DataType::I8 => "int8",
        DataType::F64 => "float64",
        DataType::U64 => "uint64",
        DataType::U32 => "uint32",
        DataType::U16 => "uint16",
        DataType::I4 => "int4",
        DataType::BF16 => "bfloat16",
        _ => "unknown",
    }
}

fn tensor(desc: &TensorDesc<'_>) -> Tensor {
    Tensor {
        index: desc.index,
        dtype: dtype_name(desc.dtype),
        shape: desc.dims.as_slice().to_vec(),
        constant: desc.is_constant(),
    }
}

fn operator(op: &OperatorReport<'_>) -> Operator {
    let (opcode, name, custom) = match op.opcode() {
        OpCode::Builtin(builtin) => (
            builtin.0,
            builtin
                .name()
                .map_or_else(|| format!("BUILTIN_{}", builtin.0), str::to_owned),
            false,
        ),
        OpCode::Custom(name) => (BuiltinOp::CUSTOM.0, name.to_owned(), true),
    };
    let kernel = op.kernel();
    let missing_feature = match (op.opcode(), kernel) {
        (
            OpCode::Builtin(builtin),
            Err(Error {
                detail: Detail::UnsupportedOp { .. },
                ..
            }),
        ) => missing_feature(builtin),
        _ => None,
    };
    Operator {
        index: op.index(),
        opcode,
        name,
        custom,
        version: op.version(),
        inputs: (0..op.input_count())
            .map(|i| match op.input(i) {
                Ok(Some(desc)) => Slot::Tensor(tensor(&desc)),
                Ok(None) => Slot::Omitted,
                Err(error) => Slot::Unknown(error),
            })
            .collect(),
        outputs: (0..op.output_count())
            .map(|i| match op.output(i) {
                Ok(desc) => Slot::Tensor(tensor(&desc)),
                Err(error) => Slot::Unknown(error),
            })
            .collect(),
        kernel,
        missing_feature,
    }
}

/// The feature of `op`, if it is off in this build: the tool forwards each `op-*` feature to
/// every backend that has kernels for the operator, so none of them has the operator then,
/// whichever `--backends` are checked.
fn missing_feature(op: BuiltinOp) -> Option<&'static str> {
    let built = nanoglia_esp_nn::EspNn::has_operator(op)
        || nanoglia_cmsis_nn::CmsisNn::has_operator(op)
        || nanoglia_kernels::RustKernels::has_operator(op);
    op.feature().filter(|_| !built)
}

/// Why `op` has no kernel, in words a model author can act on.
pub fn kernel_error(op: &Operator, error: Error) -> String {
    match op.missing_feature {
        Some(feature) => format!(
            "this build of the tool has no kernel for the operator: it was built without `{feature}`"
        ),
        None => explain(error),
    }
}

/// The largest working memory the check may ask for: a model that needs more is refused rather
/// than left to exhaust the computer's memory.
const MAX_WORKING_BYTES: usize = 1 << 30;

/// Checks `model` (aligned to 16 bytes) against `backends`, tried in the order of nanoglia's
/// default registry: ESP-NN, CMSIS-NN, then the Rust kernels.
///
/// An error means the operators could not be walked at all: not a TFLite model, or several
/// subgraphs.
pub fn check(model: &[u8], backends: Backends) -> Result<Report, Error> {
    let mut list: Vec<&dyn Backend> = Vec::new();
    if backends.esp_nn {
        list.push(&nanoglia_esp_nn::BACKEND);
    }
    if backends.cmsis_nn {
        list.push(&nanoglia_cmsis_nn::BACKEND);
    }
    if backends.kernels {
        list.push(&nanoglia_kernels::BACKEND);
    }
    let registry = Registry::new(&list);

    // The check only borrows working memory, from a buffer as a firmware would give it. Its size
    // depends on the model: when the buffer is too small, the check says how much it asked for
    // before reporting any operator, and runs again with that.
    let mut size = 64 * 1024;
    loop {
        let mut buffer = vec![0u8; size];
        let arena = Arena::new(&mut buffer);
        let mut operators = Vec::new();
        let result = ModelReport::check_with(
            &registry,
            &[model],
            Encoding::Autodetect,
            Target::Auto,
            &arena,
            |op| operators.push(operator(op)),
        );
        match result {
            Ok(report) => {
                return Ok(Report {
                    operators,
                    unsupported: report.unsupported,
                    memory: report.memory,
                });
            }
            Err(Error {
                detail: Detail::OutOfMemory { requested },
                ..
            }) if requested > size / 2 && requested <= MAX_WORKING_BYTES / 2 => {
                // Room for the block and for aligning it.
                size = requested * 2;
            }
            Err(error) => return Err(error),
        }
    }
}

/// The wasi-nn name of an error code.
pub fn code_name(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    }
}

/// An error in words a model author can act on.
pub fn explain(error: Error) -> String {
    match error.detail {
        Detail::UnsupportedOp { .. } => {
            "no selected backend runs this operator with these types, shapes and options".to_owned()
        }
        Detail::UnsupportedType { tensor } => format!(
            "tensor {tensor} has a type, rank or quantization scheme nanoglia cannot represent"
        ),
        Detail::UnsupportedStructure => "unsupported structure: several subgraphs, a variable \
             (stateful) tensor, or an operator with more than 16 inputs or 8 outputs"
            .to_owned(),
        Detail::ShapeMismatch { tensor } => {
            format!("tensor {tensor} does not have a shape the operator accepts")
        }
        Detail::Misaligned { tensor } => {
            format!("the data of constant tensor {tensor} is not aligned for its type in the file")
        }
        Detail::MalformedModel { offset } => format!("malformed model (byte {offset})"),
        Detail::UnsupportedSchema { version } => {
            format!("TFLite schema version {version} (nanoglia reads version 3)")
        }
        Detail::UnknownEncoding => "not a TensorFlow Lite model (no TFL3 identifier)".to_owned(),
        _ => error.to_string(),
    }
}
