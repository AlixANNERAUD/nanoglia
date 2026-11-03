//! Kernels receive tensors and their prepared parameters as byte slices; these helpers turn them
//! into typed slices after checking size and alignment, without any copy.

use crate::{Detail, Error, Result};
use core::mem::{align_of, size_of};

/// A type for which every bit pattern is valid and that can be copied byte-wise.
///
/// # Safety
///
/// Implementors must be `Copy`, have no padding bytes, accept any bit pattern, and have an
/// alignment of at most [`crate::ALIGN`]. Kernel parameter structs written with [`write()`] and read
/// back with [`read()`] only need to be `#[repr(C)]` and `Copy` with no padding.
pub unsafe trait Pod: Copy + 'static {}

macro_rules! pod {
    ($($t:ty),*) => {$(
        // SAFETY: primitive integers and floats accept any bit pattern and have no padding.
        unsafe impl Pod for $t {}
    )*};
}
pod!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

fn misaligned() -> Error {
    Error::invalid_argument(Detail::Misaligned { tensor: u32::MAX })
}

/// Views `bytes` as `&[T]`. Fails if `bytes` is not aligned for `T` or not a whole number of `T`.
pub fn cast_slice<T: Pod>(bytes: &[u8]) -> Result<&[T]> {
    if bytes.as_ptr().align_offset(align_of::<T>()) != 0
        || !bytes.len().is_multiple_of(size_of::<T>())
    {
        return Err(misaligned());
    }
    // SAFETY: alignment and length were checked; `T: Pod` accepts any bit pattern.
    Ok(unsafe {
        core::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), bytes.len() / size_of::<T>())
    })
}

/// Views `bytes` as `&mut [T]`. Fails if `bytes` is not aligned for `T` or not a whole number of `T`.
pub fn cast_slice_mut<T: Pod>(bytes: &mut [u8]) -> Result<&mut [T]> {
    if bytes.as_ptr().align_offset(align_of::<T>()) != 0
        || !bytes.len().is_multiple_of(size_of::<T>())
    {
        return Err(misaligned());
    }
    let len = bytes.len() / size_of::<T>();
    // SAFETY: alignment and length were checked; `T: Pod` accepts any bit pattern; the borrow is unique.
    Ok(unsafe { core::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<T>(), len) })
}

/// Views the start of `bytes` as `&T`.
pub fn read<T: Pod>(bytes: &[u8]) -> Result<&T> {
    let bytes = bytes.get(..size_of::<T>()).ok_or_else(misaligned)?;
    Ok(&cast_slice::<T>(bytes)?[0])
}

/// Writes `value` at the start of `bytes`.
pub fn write<T: Pod>(bytes: &mut [u8], value: T) -> Result<()> {
    let bytes = bytes.get_mut(..size_of::<T>()).ok_or_else(misaligned)?;
    cast_slice_mut::<T>(bytes)?[0] = value;
    Ok(())
}

/// A typed region inside a kernel's persistent buffer, computed identically at plan, prepare and
/// eval time by [`Layout`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region<T> {
    offset: usize,
    len: usize,
    _type: core::marker::PhantomData<T>,
}

impl<T: Pod> Region<T> {
    /// The typed slice of this region in `buffer`.
    pub fn get<'b>(&self, buffer: &'b [u8]) -> Result<&'b [T]> {
        let end = self.offset + self.len * size_of::<T>();
        cast_slice(buffer.get(self.offset..end).ok_or_else(misaligned)?)
    }

    /// The mutable typed slice of this region in `buffer`.
    pub fn get_mut<'b>(&self, buffer: &'b mut [u8]) -> Result<&'b mut [T]> {
        let end = self.offset + self.len * size_of::<T>();
        cast_slice_mut(buffer.get_mut(self.offset..end).ok_or_else(misaligned)?)
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the region is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Lays out typed regions one after another in a buffer aligned to [`crate::ALIGN`].
///
/// A kernel builds the same layout in `plan` (to report its size), in `prepare` (to fill it) and in
/// `eval` (to read it back).
#[derive(Clone, Copy, Debug, Default)]
pub struct Layout {
    size: usize,
}

impl Layout {
    /// An empty layout.
    pub const fn new() -> Self {
        Self { size: 0 }
    }

    /// Appends `len` elements of `T`.
    pub fn push<T: Pod>(&mut self, len: usize) -> Region<T> {
        let align = align_of::<T>();
        let offset = (self.size + align - 1) & !(align - 1);
        self.size = offset + len * size_of::<T>();
        Region {
            offset,
            len,
            _type: core::marker::PhantomData,
        }
    }

    /// Total size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}
