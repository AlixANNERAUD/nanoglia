//! C API of [`nanoglia`]: load a TensorFlow Lite model and run it from C or C++, with every byte
//! in one buffer the caller provides.
//!
//! The functions are declared for C and C++ in `include/nanoglia.h`, generated from this crate by
//! cbindgen. Build the static library (`libnanoglia_c.a`) with the CMake project of this crate,
//! or with `cargo rustc -p nanoglia-c --release --lib --crate-type staticlib` and the `std` feature on
//! a host, the `panic-handler` feature on a target without `std`. See the guide *Use nanoglia
//! from C* in the repository documentation.
//!
//! # Memory
//!
//! Nothing is allocated: [`nanoglia_load`] places the handle (the graph and one execution
//! context) at the start of the caller's buffer, and the graph block and the context arena after
//! it. [`nanoglia_memory_requirements`] gives the exact size of that buffer.
//!
//! # Safety
//!
//! Every function checks its pointers for null, its handle for a live handle, and its sizes,
//! and returns a status. A model is untrusted input: a malformed model returns an error, never
//! panics. Should a panic happen anyway (a bug), it is caught at the boundary and returns
//! [`Status::Unknown`] with the `std` feature; without it, the panic handler of the program
//! runs (with the `panic-handler` feature, an endless loop).

#![no_std]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

use core::ffi::{c_char, c_void};
use core::mem::MaybeUninit;
use core::ptr;
use nanoglia::core::ALIGN;
use nanoglia::{
    Arena, DataType, Detail, Encoding, Error, ErrorCode, ExecutionContext, Graph,
    MemoryRequirements, Quantization, Step, Target,
};

/// The most dimensions a tensor has.
pub const MAX_RANK: usize = 6;
const _: () = assert!(MAX_RANK == nanoglia::core::MAX_RANK);

/// The alignment, in bytes, the model bytes must have.
pub const MODEL_ALIGNMENT: usize = 16;
const _: () = assert!(MODEL_ALIGNMENT == ALIGN);

/// Size of `nanoglia_error::backend`, its NUL terminator included.
pub const BACKEND_NAME_SIZE: usize = 16;

/// Outcome of a call.
///
/// Besides `NANOGLIA_STATUS_OK`, the wasi-nn `error-code` values, in the same order (wasi-nn's value
/// plus one).
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Status {
    /// Success.
    Ok = 0,
    /// An invalid argument: a null pointer, an unknown handle, a misaligned model, an index out
    /// of range...
    InvalidArgument = 1,
    /// The model bytes are not a valid TensorFlow Lite model.
    InvalidEncoding = 2,
    /// The computation was stopped.
    Timeout = 3,
    /// A kernel failed while computing.
    RuntimeError = 4,
    /// The model uses an operator, a type or a feature that no enabled backend supports.
    UnsupportedOperation = 5,
    /// The buffer is too small.
    TooLarge = 6,
    /// No input or output has this name.
    NotFound = 7,
    /// The operation is not allowed.
    Security = 8,
    /// An unspecified failure.
    Unknown = 9,
}

impl From<ErrorCode> for Status {
    fn from(code: ErrorCode) -> Self {
        match code {
            ErrorCode::InvalidArgument => Self::InvalidArgument,
            ErrorCode::InvalidEncoding => Self::InvalidEncoding,
            ErrorCode::Timeout => Self::Timeout,
            ErrorCode::RuntimeError => Self::RuntimeError,
            ErrorCode::UnsupportedOperation => Self::UnsupportedOperation,
            ErrorCode::TooLarge => Self::TooLarge,
            ErrorCode::NotFound => Self::NotFound,
            ErrorCode::Security => Self::Security,
            ErrorCode::Unknown => Self::Unknown,
        }
    }
}

/// The precise reason of an error. Each value says which fields of
/// `nanoglia_error` it sets; the others are zero.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DetailKind {
    /// No further detail.
    None = 0,
    /// The model is malformed at byte offset `size`.
    MalformedModel = 1,
    /// The model was written with schema version `size`, which is not supported.
    UnsupportedSchema = 2,
    /// The encoding of the model is not recognized.
    UnknownEncoding = 3,
    /// The model has a structure nanoglia does not support (several subgraphs...).
    UnsupportedStructure = 4,
    /// The constant buffer of tensor `index` is not aligned for its type.
    Misaligned = 5,
    /// No backend supports operator `index`, of TensorFlow Lite builtin code `code`.
    UnsupportedOperator = 6,
    /// The type of tensor `index` is not supported.
    UnsupportedType = 7,
    /// Tensor `index` does not have the expected shape.
    ShapeMismatch = 8,
    /// A tensor has `actual` bytes instead of `size`.
    SizeMismatch = 9,
    /// Input or output `index` does not exist.
    IndexOutOfRange = 10,
    /// An allocation of `size` bytes did not fit in the buffer.
    OutOfMemory = 11,
    /// No input or output has the requested name.
    TensorNotFound = 12,
    /// The computation stopped before operator `index`.
    Interrupted = 13,
    /// The backend named `backend` failed with status `code`.
    Backend = 14,
    /// A detail this version of the C API does not know.
    Unknown = 15,
}

/// An error and its detail.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ErrorInfo {
    /// What went wrong, `NANOGLIA_STATUS_OK` on success.
    pub status: Status,
    /// The precise reason, which says which of the fields below are set.
    pub detail: DetailKind,
    /// A tensor, operator, input or output index.
    pub index: u32,
    /// A TensorFlow Lite operator code or a backend status.
    pub code: i32,
    /// A byte offset, a size in bytes or a schema version.
    pub size: usize,
    /// The actual size in bytes.
    pub actual: usize,
    /// The name of the backend, NUL-terminated (truncated if needed), empty if none.
    pub backend: [c_char; BACKEND_NAME_SIZE],
}

impl ErrorInfo {
    const OK: Self = Self {
        status: Status::Ok,
        detail: DetailKind::None,
        index: 0,
        code: 0,
        size: 0,
        actual: 0,
        backend: [0; BACKEND_NAME_SIZE],
    };
}

impl From<Error> for ErrorInfo {
    fn from(error: Error) -> Self {
        let mut info = Self {
            status: error.code.into(),
            ..Self::OK
        };
        info.detail = match error.detail {
            Detail::None => DetailKind::None,
            Detail::MalformedModel { offset } => {
                info.size = offset as usize;
                DetailKind::MalformedModel
            }
            Detail::UnsupportedSchema { version } => {
                info.size = version as usize;
                DetailKind::UnsupportedSchema
            }
            Detail::UnknownEncoding => DetailKind::UnknownEncoding,
            Detail::UnsupportedStructure => DetailKind::UnsupportedStructure,
            Detail::Misaligned { tensor } => {
                info.index = tensor;
                DetailKind::Misaligned
            }
            Detail::UnsupportedOp { op_index, opcode } => {
                info.index = op_index;
                info.code = opcode;
                DetailKind::UnsupportedOperator
            }
            Detail::UnsupportedType { tensor } => {
                info.index = tensor;
                DetailKind::UnsupportedType
            }
            Detail::ShapeMismatch { tensor } => {
                info.index = tensor;
                DetailKind::ShapeMismatch
            }
            Detail::SizeMismatch { expected, actual } => {
                info.size = expected as usize;
                info.actual = actual as usize;
                DetailKind::SizeMismatch
            }
            Detail::IndexOutOfRange { index } => {
                info.index = index;
                DetailKind::IndexOutOfRange
            }
            Detail::OutOfMemory { requested } => {
                info.size = requested;
                DetailKind::OutOfMemory
            }
            Detail::TensorNotFound => DetailKind::TensorNotFound,
            Detail::Interrupted { op_index } => {
                info.index = op_index;
                DetailKind::Interrupted
            }
            Detail::Backend { backend, status } => {
                info.code = status;
                let name = backend.as_bytes();
                for (slot, &byte) in info.backend[..BACKEND_NAME_SIZE - 1].iter_mut().zip(name) {
                    *slot = byte as c_char;
                }
                DetailKind::Backend
            }
            _ => DetailKind::Unknown,
        };
        info
    }
}

/// The memory a model needs.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Memory {
    /// Bytes of the graph block (kernel table, prepared parameters, memory plan).
    pub graph_bytes: usize,
    /// Bytes of the execution context arena (activations and scratch memory).
    pub context_bytes: usize,
    /// Size of the buffer nanoglia_load() needs, at any alignment: the handle, the graph
    /// block and the context arena, with their alignment padding.
    pub memory_bytes: usize,
}

/// The element type of a tensor. The values are the TensorFlow Lite
/// `TensorType` codes.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// A type this version of the C API does not know.
    Unknown = -1,
    /// 32-bit float.
    Float32 = 0,
    /// 16-bit IEEE float.
    Float16 = 1,
    /// 32-bit signed integer.
    Int32 = 2,
    /// 8-bit unsigned integer.
    Uint8 = 3,
    /// 64-bit signed integer.
    Int64 = 4,
    /// Boolean, one byte per element.
    Bool = 6,
    /// 16-bit signed integer.
    Int16 = 7,
    /// 8-bit signed integer.
    Int8 = 9,
    /// 64-bit float.
    Float64 = 10,
    /// 64-bit unsigned integer.
    Uint64 = 12,
    /// 32-bit unsigned integer.
    Uint32 = 15,
    /// 16-bit unsigned integer.
    Uint16 = 16,
    /// 4-bit signed integer, two elements per byte.
    Int4 = 17,
    /// 16-bit brain float.
    Bfloat16 = 18,
}

impl From<DataType> for ElementType {
    fn from(dtype: DataType) -> Self {
        match dtype {
            DataType::F32 => Self::Float32,
            DataType::F16 => Self::Float16,
            DataType::I32 => Self::Int32,
            DataType::U8 => Self::Uint8,
            DataType::I64 => Self::Int64,
            DataType::Bool => Self::Bool,
            DataType::I16 => Self::Int16,
            DataType::I8 => Self::Int8,
            DataType::F64 => Self::Float64,
            DataType::U64 => Self::Uint64,
            DataType::U32 => Self::Uint32,
            DataType::U16 => Self::Uint16,
            DataType::I4 => Self::Int4,
            DataType::BF16 => Self::Bfloat16,
            _ => Self::Unknown,
        }
    }
}

/// An input or output tensor of a loaded model.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    /// The contents, in the buffer of the handle (or in the model bytes for a constant output):
    /// writable for an input, read-only for an output.
    pub data: *mut c_void,
    /// Size of the contents in bytes.
    pub size: usize,
    /// The element type.
    pub element_type: ElementType,
    /// Number of dimensions, at most `NANOGLIA_MAX_RANK`.
    pub rank: u32,
    /// The dimensions; those past `rank` are zero.
    pub dims: [i32; MAX_RANK],
    /// The scale of a quantized tensor (the first one if per-axis), `0` otherwise.
    pub scale: f32,
    /// The zero point of a quantized tensor (the first one if per-axis), `0` otherwise.
    pub zero_point: i32,
    /// The name in the model, not NUL-terminated, borrowed from the model bytes.
    pub name: *const c_char,
    /// Length of `name` in bytes.
    pub name_length: usize,
}

impl Tensor {
    fn new(
        data: *mut c_void,
        size: usize,
        dtype: DataType,
        shape: &[i32],
        quantization: Quantization<'_>,
        name: &str,
    ) -> Self {
        let mut dims = [0; MAX_RANK];
        for (slot, &dim) in dims.iter_mut().zip(shape) {
            *slot = dim;
        }
        Self {
            data,
            size,
            element_type: dtype.into(),
            rank: shape.len().min(MAX_RANK) as u32,
            dims,
            scale: quantization.scale(),
            zero_point: quantization.zero_point(),
            name: name.as_ptr().cast(),
            name_length: name.len(),
        }
    }
}

/// cbindgen:ignore
type HandleArena = Arena<'static>;
/// cbindgen:ignore
type HandleGraph = Graph<'static, &'static HandleArena>;
/// cbindgen:ignore
type HandleContext = ExecutionContext<'static, 'static, &'static HandleArena, &'static HandleArena>;

/// Marks a live handle.
const MAGIC: u32 = u32::from_le_bytes(*b"nngl");

/// A loaded model and its execution context, opaque.
///
/// nanoglia_load() places it at the start of the caller's buffer, which it never leaves, and
/// nanoglia_release() ends it. Use it from one thread at a time.
//
// The lifetimes are `'static` only in name: the handle borrows the model bytes and the buffer,
// which the caller keeps valid until `nanoglia_release`. The context borrows the graph and both
// borrow the arena, all three in the same handle: no reference to the whole handle is ever
// created, only to its fields.
#[repr(align(16))]
pub struct Nanoglia {
    context: HandleContext,
    graph: HandleGraph,
    arena: HandleArena,
    magic: u32,
}

// The handle is followed by the arena: aligned for its blocks without padding.
const _: () = assert!(align_of::<Nanoglia>() == ALIGN);

/// Size of the buffer that holds the handle and an arena of `needs.arena_bytes(1)`, at any
/// alignment, or `None` on overflow.
fn memory_bytes(needs: &MemoryRequirements) -> Option<usize> {
    let graph = needs.graph_bytes.checked_next_multiple_of(ALIGN)?;
    let context = needs.context_bytes.checked_next_multiple_of(ALIGN)?;
    // Up to `ALIGN - 1` bytes align the handle; the arena that follows it is aligned.
    (ALIGN - 1)
        .checked_add(size_of::<Nanoglia>())?
        .checked_add(graph)?
        .checked_add(context)
}

/// A null pointer or another invalid argument.
const INVALID: Error = Error::invalid_argument(Detail::None);

/// Runs `body`, catching a panic with the `std` feature.
fn guard<T>(body: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
    #[cfg(feature = "std")]
    {
        std::panic::catch_unwind(core::panic::AssertUnwindSafe(body))
            .unwrap_or(Err(Error::new(ErrorCode::Unknown, Detail::None)))
    }
    #[cfg(not(feature = "std"))]
    {
        body()
    }
}

/// Writes the outcome to `error` unless it is null, and returns its status.
///
/// # Safety
///
/// `error` is null or valid for writes.
unsafe fn report(outcome: Result<(), Error>, error: *mut ErrorInfo) -> Status {
    let info = match outcome {
        Ok(()) => ErrorInfo::OK,
        Err(e) => e.into(),
    };
    if !error.is_null() {
        // SAFETY: valid for writes (caller contract), and `write` does not read the old value.
        unsafe { error.write_unaligned(info) };
    }
    info.status
}

/// The status of `outcome`.
fn status(outcome: Result<(), Error>) -> Status {
    match outcome {
        Ok(()) => Status::Ok,
        Err(e) => e.code.into(),
    }
}

/// The model bytes, checked for null and alignment.
///
/// # Safety
///
/// `model` is null or points to `length` bytes valid for `'a`.
unsafe fn model_bytes<'a>(model: *const u8, length: usize) -> Result<&'a [u8], Error> {
    if model.is_null() || !model.addr().is_multiple_of(MODEL_ALIGNMENT) {
        return Err(INVALID);
    }
    // SAFETY: non-null, `length` readable bytes (caller contract).
    Ok(unsafe { core::slice::from_raw_parts(model, length) })
}

/// The live handle at `nn`.
///
/// # Safety
///
/// `nn` is null or points to memory that was passed to nanoglia_load() (a live or released
/// handle) or that is readable for a `Nanoglia`.
unsafe fn live(nn: *const Nanoglia) -> Result<*const Nanoglia, Error> {
    if nn.is_null() || !nn.is_aligned() {
        return Err(INVALID);
    }
    // SAFETY: readable (caller contract); a field read creates no reference to the handle.
    let magic = unsafe { ptr::read(&raw const (*nn).magic) };
    if magic != MAGIC {
        return Err(INVALID);
    }
    Ok(nn)
}

/// The context of the live handle at `nn`.
///
/// # Safety
///
/// As [`live`], and no other reference to the context exists while the result lives.
unsafe fn context_mut<'a>(nn: *mut Nanoglia) -> Result<&'a mut HandleContext, Error> {
    // SAFETY: caller contract.
    let nn = unsafe { live(nn) }?.cast_mut();
    // SAFETY: a live handle has an initialized context; it is a field distinct from the graph
    // and the arena it borrows.
    Ok(unsafe { &mut (*nn).context })
}

/// The context of the live handle at `nn`, shared.
///
/// # Safety
///
/// As [`live`], and no mutable reference to the context exists while the result lives.
unsafe fn context<'a>(nn: *const Nanoglia) -> Result<&'a HandleContext, Error> {
    // SAFETY: caller contract.
    let nn = unsafe { live(nn) }?;
    // SAFETY: a live handle has an initialized context.
    Ok(unsafe { &(*nn).context })
}

/// Computes the memory a model needs.
///
/// Verifies the model, selects its kernels and plans its memory with the default backends of
/// the library, as nanoglia_load() does, and writes the sizes to `memory`. Only the memory
/// planner's working memory (about 20 bytes per tensor) is taken from `scratch`, and given back:
/// pass any buffer, such as the one you will load the model into. If it is too small, the
/// status is `NANOGLIA_STATUS_TOO_LARGE` with detail `NANOGLIA_DETAIL_OUT_OF_MEMORY` and the size that
/// did not fit (plus up to 15 bytes of alignment).
///
/// Writes the status and its detail to `error` unless it is null.
///
/// # Safety
///
/// `model` must point to `model_length` readable bytes, `scratch` to `scratch_length` writable
/// bytes, `memory` to a writable `nanoglia_memory`, and `error` to a writable `nanoglia_error`
/// or be null. A null `model`, `scratch` or `memory`, or a `model` not aligned on
/// `NANOGLIA_MODEL_ALIGNMENT` bytes, returns `NANOGLIA_STATUS_INVALID_ARGUMENT`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_memory_requirements(
    model: *const u8,
    model_length: usize,
    scratch: *mut c_void,
    scratch_length: usize,
    memory: *mut Memory,
    error: *mut ErrorInfo,
) -> Status {
    let outcome = guard(|| {
        if scratch.is_null() || memory.is_null() {
            return Err(INVALID);
        }
        // SAFETY: caller contract.
        let model = unsafe { model_bytes(model, model_length) }?;
        // SAFETY: `scratch_length` writable bytes (caller contract), possibly uninitialized.
        let scratch = unsafe {
            core::slice::from_raw_parts_mut(scratch.cast::<MaybeUninit<u8>>(), scratch_length)
        };
        let needs = MemoryRequirements::compute_in(
            &[model],
            Encoding::Autodetect,
            Target::Auto,
            Arena::new_uninit(scratch),
        )?;
        let memory_bytes = memory_bytes(&needs).ok_or(Error::out_of_memory(usize::MAX))?;
        // SAFETY: valid for writes (caller contract).
        unsafe {
            memory.write_unaligned(Memory {
                graph_bytes: needs.graph_bytes,
                context_bytes: needs.context_bytes,
                memory_bytes,
            })
        };
        Ok(())
    });
    // SAFETY: caller contract.
    unsafe { report(outcome, error) }
}

/// Loads a model into `memory` and creates its execution context.
///
/// Verifies the model, selects its kernels with the default backends of the library and plans
/// its memory. The handle, the graph block and the context arena are all placed in `memory`;
/// nothing is allocated. `memory` needs `memory_bytes` bytes of nanoglia_memory_requirements(),
/// at any alignment, and need not be initialized. On success, writes the handle to `nn`.
///
/// The model bytes are borrowed, never copied: they and `memory` must stay valid and unchanged
/// (except through the handle) until nanoglia_release(). They must not overlap.
///
/// Writes the status and its detail to `error` unless it is null. A buffer too small gives
/// `NANOGLIA_STATUS_TOO_LARGE`, a malformed model `NANOGLIA_STATUS_INVALID_ENCODING`, an operator no backend
/// supports `NANOGLIA_STATUS_UNSUPPORTED_OPERATION`.
///
/// # Safety
///
/// `model` must point to `model_length` readable bytes, `memory` to `memory_length` writable
/// bytes, `nn` to a writable `nanoglia *`, and `error` to a writable `nanoglia_error` or be null.
/// A null `model`, `memory` or `nn`, or a `model` not aligned on `NANOGLIA_MODEL_ALIGNMENT`
/// bytes, returns `NANOGLIA_STATUS_INVALID_ARGUMENT`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_load(
    model: *const u8,
    model_length: usize,
    memory: *mut c_void,
    memory_length: usize,
    nn: *mut *mut Nanoglia,
    error: *mut ErrorInfo,
) -> Status {
    let outcome = guard(|| {
        if memory.is_null() || nn.is_null() {
            return Err(INVALID);
        }
        // SAFETY: caller contract; the bytes stay valid until the handle is released.
        let model: &'static [u8] = unsafe { model_bytes(model, model_length) }?;

        let padding = memory.addr().wrapping_neg() % align_of::<Nanoglia>();
        let arena_start = padding
            .checked_add(size_of::<Nanoglia>())
            .filter(|&end| end <= memory_length)
            .ok_or(Error::out_of_memory(size_of::<Nanoglia>()))?;
        // SAFETY: `padding` is within the buffer (checked above).
        let handle = unsafe { memory.byte_add(padding) }.cast::<Nanoglia>();
        // SAFETY: the rest of the buffer, after the handle, possibly uninitialized; it stays
        // valid until the handle is released (caller contract).
        let rest: &'static mut [MaybeUninit<u8>] = unsafe {
            core::slice::from_raw_parts_mut(
                memory.byte_add(arena_start).cast(),
                memory_length - arena_start,
            )
        };

        // SAFETY: `handle` is aligned and its `size_of::<Nanoglia>()` bytes are within the
        // buffer. Each field is written before it is borrowed, through raw pointers only; the
        // arena, then the graph and the context never move.
        unsafe {
            (&raw mut (*handle).magic).write(0);
            (&raw mut (*handle).arena).write(Arena::new_uninit(rest));
            let arena: &'static HandleArena = &(*handle).arena;
            let graph = Graph::load_in(&[model], Encoding::Autodetect, Target::Auto, arena)?;
            (&raw mut (*handle).graph).write(graph);
            let graph: &'static HandleGraph = &(*handle).graph;
            match graph.init_execution_context_in(arena) {
                Ok(context) => (&raw mut (*handle).context).write(context),
                Err(e) => {
                    ptr::drop_in_place(&raw mut (*handle).graph);
                    return Err(e);
                }
            }
            (&raw mut (*handle).magic).write(MAGIC);
            nn.write_unaligned(handle);
        }
        Ok(())
    });
    // SAFETY: caller contract.
    unsafe { report(outcome, error) }
}

/// Releases a handle: the model bytes and the buffer are no longer borrowed, and the handle can
/// no longer be used. Releasing a handle twice returns `NANOGLIA_STATUS_INVALID_ARGUMENT`, as long as its
/// buffer was not reused.
///
/// # Safety
///
/// `nn` must be null or a handle written by nanoglia_load() whose buffer is still valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_release(nn: *mut Nanoglia) -> Status {
    status(guard(|| {
        // SAFETY: caller contract.
        let nn = unsafe { live(nn) }?.cast_mut();
        // SAFETY: a live handle: mark it released, then drop the context before the graph it
        // borrows. The arena has nothing to drop.
        unsafe {
            (&raw mut (*nn).magic).write(0);
            ptr::drop_in_place(&raw mut (*nn).context);
            ptr::drop_in_place(&raw mut (*nn).graph);
        }
        Ok(())
    }))
}

/// Number of inputs of the model, `0` for a null or released handle.
///
/// # Safety
///
/// `nn` must be null or a handle written by nanoglia_load() whose buffer is still valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_input_count(nn: *const Nanoglia) -> usize {
    // SAFETY: caller contract.
    guard(|| Ok(unsafe { context(nn) }?.graph().input_count())).unwrap_or(0)
}

/// Number of outputs of the model, `0` for a null or released handle.
///
/// # Safety
///
/// `nn` must be null or a handle written by nanoglia_load() whose buffer is still valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_output_count(nn: *const Nanoglia) -> usize {
    // SAFETY: caller contract.
    guard(|| Ok(unsafe { context(nn) }?.graph().output_count())).unwrap_or(0)
}

/// Number of operators of the model, which is the number of nanoglia_step() calls of an
/// inference; `0` for a null or released handle.
///
/// # Safety
///
/// `nn` must be null or a handle written by nanoglia_load() whose buffer is still valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_operator_count(nn: *const Nanoglia) -> usize {
    // SAFETY: caller contract.
    guard(|| Ok(unsafe { context(nn) }?.graph().operator_count())).unwrap_or(0)
}

/// Describes input `index` and gives its writable contents, in the buffer of the handle.
///
/// Write the inputs before each inference (nanoglia_compute(), or the first
/// nanoglia_step()): the memory plan reuses their memory once the operators have read them.
/// The pointer stays the same until the handle is released.
///
/// # Safety
///
/// `nn` must be null or a handle written by nanoglia_load() whose buffer is still valid, and
/// `tensor` must point to a writable `nanoglia_tensor` or be null. A null argument returns
/// `NANOGLIA_STATUS_INVALID_ARGUMENT`, as does an `index` out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_input(
    nn: *mut Nanoglia,
    index: usize,
    tensor: *mut Tensor,
) -> Status {
    status(guard(|| {
        if tensor.is_null() {
            return Err(INVALID);
        }
        // SAFETY: caller contract.
        let context = unsafe { context_mut(nn) }?;
        let mut input = context.input_mut(index)?;
        let (dtype, dims, quantization, name) = (
            input.dtype(),
            input.dims(),
            input.quantization(),
            input.name(),
        );
        let bytes = input.bytes_mut();
        let described = Tensor::new(
            bytes.as_mut_ptr().cast(),
            bytes.len(),
            dtype,
            dims.as_slice(),
            quantization,
            name,
        );
        // SAFETY: valid for writes (caller contract).
        unsafe { tensor.write_unaligned(described) };
        Ok(())
    }))
}

/// Describes output `index` and gives its read-only contents.
///
/// The contents hold the result once nanoglia_compute() succeeds, or nanoglia_step()
/// reports the end of the inference, and until the next inference starts.
///
/// # Safety
///
/// `nn` must be null or a handle written by nanoglia_load() whose buffer is still valid, and
/// `tensor` must point to a writable `nanoglia_tensor` or be null. A null argument returns
/// `NANOGLIA_STATUS_INVALID_ARGUMENT`, as does an `index` out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_output(
    nn: *const Nanoglia,
    index: usize,
    tensor: *mut Tensor,
) -> Status {
    status(guard(|| {
        if tensor.is_null() {
            return Err(INVALID);
        }
        // SAFETY: caller contract.
        let output = unsafe { context(nn) }?.output(index)?;
        let bytes = output.bytes();
        let described = Tensor::new(
            bytes.as_ptr().cast_mut().cast(),
            bytes.len(),
            output.dtype(),
            output.dims().as_slice(),
            output.quantization(),
            output.name(),
        );
        // SAFETY: valid for writes (caller contract).
        unsafe { tensor.write_unaligned(described) };
        Ok(())
    }))
}

/// The name of `name_length` bytes at `name`, as UTF-8; a name that is not UTF-8 matches no
/// tensor.
///
/// # Safety
///
/// `name` points to `name_length` readable bytes.
unsafe fn tensor_name<'a>(name: *const c_char, name_length: usize) -> Result<&'a str, Error> {
    if name.is_null() {
        return Err(INVALID);
    }
    // SAFETY: caller contract.
    let bytes = unsafe { core::slice::from_raw_parts(name.cast::<u8>(), name_length) };
    core::str::from_utf8(bytes).map_err(|_| Error::new(ErrorCode::NotFound, Detail::TensorNotFound))
}

/// Finds the input named `name` (`name_length` bytes, not necessarily NUL-terminated) and
/// writes its index to `index`. Returns `NANOGLIA_STATUS_NOT_FOUND` if no input has this name.
///
/// # Safety
///
/// `nn` must be null or a handle written by nanoglia_load() whose buffer is still valid,
/// `name` must point to `name_length` readable bytes and `index` to a writable `size_t`. A null
/// argument returns `NANOGLIA_STATUS_INVALID_ARGUMENT`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_input_index(
    nn: *const Nanoglia,
    name: *const c_char,
    name_length: usize,
    index: *mut usize,
) -> Status {
    status(guard(|| {
        if index.is_null() {
            return Err(INVALID);
        }
        // SAFETY: caller contract.
        let context = unsafe { context(nn) }?;
        // SAFETY: caller contract.
        let found = context
            .graph()
            .input_index(unsafe { tensor_name(name, name_length) }?)?;
        // SAFETY: valid for writes (caller contract).
        unsafe { index.write_unaligned(found) };
        Ok(())
    }))
}

/// Finds the output named `name` (`name_length` bytes, not necessarily NUL-terminated) and
/// writes its index to `index`. Returns `NANOGLIA_STATUS_NOT_FOUND` if no output has this name.
///
/// # Safety
///
/// As nanoglia_input_index().
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_output_index(
    nn: *const Nanoglia,
    name: *const c_char,
    name_length: usize,
    index: *mut usize,
) -> Status {
    status(guard(|| {
        if index.is_null() {
            return Err(INVALID);
        }
        // SAFETY: caller contract.
        let context = unsafe { context(nn) }?;
        // SAFETY: caller contract.
        let found = context
            .graph()
            .output_index(unsafe { tensor_name(name, name_length) }?)?;
        // SAFETY: valid for writes (caller contract).
        unsafe { index.write_unaligned(found) };
        Ok(())
    }))
}

/// Runs a whole inference on the current inputs. Allocates nothing. Abandons an inference run
/// with nanoglia_step(), if any.
///
/// Writes the status and its detail to `error` unless it is null.
///
/// # Safety
///
/// `nn` must be null or a handle written by nanoglia_load() whose buffer is still valid, and
/// `error` must point to a writable `nanoglia_error` or be null. A null `nn` returns
/// `NANOGLIA_STATUS_INVALID_ARGUMENT`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_compute(nn: *mut Nanoglia, error: *mut ErrorInfo) -> Status {
    // SAFETY: caller contract.
    let outcome = guard(|| unsafe { context_mut(nn) }?.compute());
    // SAFETY: caller contract.
    unsafe { report(outcome, error) }
}

/// Runs the next operator of the current inference, starting one on the current inputs if
/// none is in progress, and writes to `done` whether it was the last one.
///
/// Call it until `done` is true to run a long inference in slices, between which the program
/// does something else; stop calling it, or call nanoglia_cancel(), to abandon the inference.
/// Do not change the inputs while an inference is in progress. If an operator fails, its error
/// is returned and the inference is abandoned: the next call starts a new one.
///
/// Writes the status and its detail to `error` unless it is null.
///
/// # Safety
///
/// `nn` must be null or a handle written by nanoglia_load() whose buffer is still valid,
/// `done` must point to a writable `bool`, and `error` to a writable `nanoglia_error` or be null.
/// A null `nn` or `done` returns `NANOGLIA_STATUS_INVALID_ARGUMENT`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_step(
    nn: *mut Nanoglia,
    done: *mut bool,
    error: *mut ErrorInfo,
) -> Status {
    let outcome = guard(|| {
        if done.is_null() {
            return Err(INVALID);
        }
        // SAFETY: caller contract.
        let step = unsafe { context_mut(nn) }?.step()?;
        // SAFETY: valid for writes (caller contract).
        unsafe { done.write_unaligned(step == Step::Done) };
        Ok(())
    });
    // SAFETY: caller contract.
    unsafe { report(outcome, error) }
}

/// Abandons the inference run with nanoglia_step(), if any: the next step starts a new one.
///
/// # Safety
///
/// `nn` must be null or a handle written by nanoglia_load() whose buffer is still valid. A
/// null `nn` returns `NANOGLIA_STATUS_INVALID_ARGUMENT`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nanoglia_cancel(nn: *mut Nanoglia) -> Status {
    status(guard(|| {
        // SAFETY: caller contract.
        unsafe { context_mut(nn) }?.cancel();
        Ok(())
    }))
}

/// Returns a short English description of a `nanoglia_status`, as a static NUL-terminated
/// string; `"unknown status"` for a value that is not one.
#[unsafe(no_mangle)]
pub extern "C" fn nanoglia_status_description(status: i32) -> *const c_char {
    let text = match status {
        0 => c"success",
        1 => c"invalid argument",
        2 => c"invalid encoding",
        3 => c"timeout",
        4 => c"runtime error",
        5 => c"unsupported operation",
        6 => c"too large",
        7 => c"not found",
        8 => c"security",
        9 => c"unknown error",
        _ => c"unknown status",
    };
    text.as_ptr()
}

/// Returns a short English description of a `nanoglia_detail`, as a static NUL-terminated
/// string; `"unknown detail"` for a value that is not one.
#[unsafe(no_mangle)]
pub extern "C" fn nanoglia_detail_description(detail: i32) -> *const c_char {
    let text = match detail {
        0 => c"no detail",
        1 => c"malformed model",
        2 => c"unsupported schema version",
        3 => c"unknown encoding",
        4 => c"unsupported model structure",
        5 => c"misaligned constant buffer",
        6 => c"unsupported operator",
        7 => c"unsupported type",
        8 => c"shape mismatch",
        9 => c"size mismatch",
        10 => c"index out of range",
        11 => c"out of memory",
        12 => c"tensor not found",
        13 => c"interrupted",
        14 => c"backend failure",
        _ => c"unknown detail",
    };
    text.as_ptr()
}

/// Loops forever: without the standard library there is nowhere to report a panic. nanoglia does
/// not panic on any model; this only runs on a bug.
#[cfg(all(feature = "panic-handler", not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
