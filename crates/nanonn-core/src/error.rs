use core::fmt;

/// The category of an [`Error`], identical to the `error-code` enum of
/// [wasi-nn](https://github.com/WebAssembly/wasi-nn) so a wasi-nn host can forward it unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum ErrorCode {
    /// The caller passed an invalid argument (wrong tensor index, type, shape or size...).
    InvalidArgument,
    /// The model bytes are not a valid model of the requested encoding.
    InvalidEncoding,
    /// An operation, typically on an accelerator, timed out.
    Timeout,
    /// A kernel or accelerator failed while computing.
    RuntimeError,
    /// The model uses an operator, type or feature that no enabled backend supports.
    UnsupportedOperation,
    /// The model does not fit: the allocator could not provide the memory it needs.
    TooLarge,
    /// A tensor or graph looked up by name does not exist.
    NotFound,
    /// The operation is not allowed, or a hardware feature cannot be accessed.
    Security,
    /// The operation failed for an unspecified reason.
    Unknown,
}

/// The precise reason of an [`Error`], with the indices needed to locate it in the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[non_exhaustive]
pub enum Detail {
    /// No further detail.
    None,
    /// The model is malformed at this byte offset.
    MalformedModel {
        /// Byte offset in the model where verification failed.
        offset: u32,
    },
    /// The model was written with an unsupported schema version.
    UnsupportedSchema {
        /// The schema version found in the model.
        version: u32,
    },
    /// The encoding could not be recognised.
    UnknownEncoding,
    /// The model has a structure nanonn does not support yet (several subgraphs, variable tensors...).
    UnsupportedStructure,
    /// A constant buffer is not aligned enough for the type a kernel reads it as.
    Misaligned {
        /// Index of the offending tensor.
        tensor: u32,
    },
    /// No backend supports this operator with these parameters.
    UnsupportedOp {
        /// Index of the operator in the graph.
        op_index: u32,
        /// TFLite builtin operator code (`32` for custom operators).
        opcode: i32,
    },
    /// A tensor type is not supported by the kernel that uses it.
    UnsupportedType {
        /// Index of the offending tensor.
        tensor: u32,
    },
    /// A tensor does not have the shape the operator or the caller expects.
    ShapeMismatch {
        /// Index of the offending tensor.
        tensor: u32,
    },
    /// A tensor does not have the size the operator or the caller expects.
    SizeMismatch {
        /// Expected size in bytes.
        expected: u32,
        /// Actual size in bytes.
        actual: u32,
    },
    /// An input or output index is out of range.
    IndexOutOfRange {
        /// The index that was requested.
        index: u32,
    },
    /// The allocator could not provide this many bytes.
    OutOfMemory {
        /// The size of the failed allocation, in bytes.
        requested: usize,
    },
    /// No input or output tensor has the requested name.
    TensorNotFound,
    /// A backend library reported a failure.
    Backend {
        /// Name of the backend.
        backend: &'static str,
        /// The status code returned by the backend library.
        status: i32,
    },
}

/// The error type of every fallible nanonn operation. It is `Copy` and never allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct Error {
    /// The wasi-nn compatible category.
    pub code: ErrorCode,
    /// The precise reason.
    pub detail: Detail,
}

/// The result type of nanonn.
pub type Result<T, E = Error> = core::result::Result<T, E>;

impl Error {
    /// Creates an error from its category and detail.
    pub const fn new(code: ErrorCode, detail: Detail) -> Self {
        Self { code, detail }
    }

    /// A malformed model at `offset`.
    pub const fn malformed(offset: usize) -> Self {
        Self::new(
            ErrorCode::InvalidEncoding,
            Detail::MalformedModel {
                offset: if offset > u32::MAX as usize {
                    u32::MAX
                } else {
                    offset as u32
                },
            },
        )
    }

    /// An invalid argument without further detail.
    pub const fn invalid_argument(detail: Detail) -> Self {
        Self::new(ErrorCode::InvalidArgument, detail)
    }

    /// An unsupported operation or feature.
    pub const fn unsupported(detail: Detail) -> Self {
        Self::new(ErrorCode::UnsupportedOperation, detail)
    }

    /// An allocation of `requested` bytes failed.
    pub const fn out_of_memory(requested: usize) -> Self {
        Self::new(ErrorCode::TooLarge, Detail::OutOfMemory { requested })
    }

    /// A backend library returned a failure status.
    pub const fn backend(backend: &'static str, status: i32) -> Self {
        Self::new(ErrorCode::RuntimeError, Detail::Backend { backend, status })
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidArgument => "invalid argument",
            Self::InvalidEncoding => "invalid encoding",
            Self::Timeout => "timeout",
            Self::RuntimeError => "runtime error",
            Self::UnsupportedOperation => "unsupported operation",
            Self::TooLarge => "too large",
            Self::NotFound => "not found",
            Self::Security => "security",
            Self::Unknown => "unknown",
        })
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.detail {
            Detail::None => write!(f, "{}", self.code),
            detail => write!(f, "{}: {:?}", self.code, detail),
        }
    }
}

impl core::error::Error for Error {}
