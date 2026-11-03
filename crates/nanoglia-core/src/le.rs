//! Model files are little-endian and their metadata is not guaranteed to be aligned, so nanoglia
//! reads scalars byte by byte through these views instead of casting pointers.

use core::marker::PhantomData;

/// A scalar that can be decoded from little-endian bytes.
pub trait LeScalar: Copy + 'static {
    /// Encoded size in bytes.
    const SIZE: usize;
    /// Decodes a value from exactly [`Self::SIZE`] bytes.
    fn from_le(bytes: &[u8]) -> Self;
}

macro_rules! le_scalar {
    ($($t:ty),*) => {$(
        impl LeScalar for $t {
            const SIZE: usize = core::mem::size_of::<$t>();
            #[inline]
            fn from_le(bytes: &[u8]) -> Self {
                let mut raw = [0u8; core::mem::size_of::<$t>()];
                raw.copy_from_slice(&bytes[..Self::SIZE]);
                <$t>::from_le_bytes(raw)
            }
        }
    )*};
}
le_scalar!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

/// Reads a `T` at `offset` in `bytes`, or `None` if it does not fit.
#[inline]
pub fn read<T: LeScalar>(bytes: &[u8], offset: usize) -> Option<T> {
    let end = offset.checked_add(T::SIZE)?;
    bytes.get(offset..end).map(T::from_le)
}

/// A zero-copy array of little-endian scalars, borrowed from model bytes.
#[derive(Clone, Copy)]
pub struct LeSlice<'a, T> {
    bytes: &'a [u8],
    _type: PhantomData<T>,
}

impl<T> core::fmt::Debug for LeSlice<'_, T>
where
    T: LeScalar + core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> PartialEq for LeSlice<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl<'a, T: LeScalar> LeSlice<'a, T> {
    /// An empty array.
    pub const EMPTY: Self = Self {
        bytes: &[],
        _type: PhantomData,
    };

    /// Views `bytes` as an array of `T`; trailing bytes that do not form a whole `T` are ignored.
    pub fn new(bytes: &'a [u8]) -> Self {
        let len = bytes.len() / T::SIZE * T::SIZE;
        Self {
            bytes: &bytes[..len],
            _type: PhantomData,
        }
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes.len() / T::SIZE
    }

    /// Whether the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The element at `index`, if any.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        read(self.bytes, index.checked_mul(T::SIZE)?)
    }

    /// The underlying bytes.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Iterates over the elements.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = T> + 'a {
        self.bytes.chunks_exact(T::SIZE).map(T::from_le)
    }
}
