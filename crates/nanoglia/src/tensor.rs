use nanoglia_core::pod::{Pod, cast_slice, cast_slice_mut};
use nanoglia_core::{DataType, Detail, Dims, Error, Quantization, Result, TensorDesc};

/// Rust element types matching a [`DataType`].
pub trait Element: Pod {
    /// The matching data type.
    const DTYPE: DataType;
}

macro_rules! element {
    ($($t:ty => $d:ident),*) => {$(
        impl Element for $t {
            const DTYPE: DataType = DataType::$d;
        }
    )*};
}
element!(i8 => I8, u8 => U8, i16 => I16, u16 => U16, i32 => I32, u32 => U32, i64 => I64, u64 => U64, f32 => F32, f64 => F64);

fn type_mismatch(desc: &TensorDesc<'_>) -> Error {
    Error::invalid_argument(Detail::UnsupportedType { tensor: desc.index })
}

/// A read-only view of a tensor of an execution context.
#[derive(Clone, Copy, Debug)]
pub struct TensorRef<'a> {
    desc: TensorDesc<'a>,
    data: &'a [u8],
}

impl<'a> TensorRef<'a> {
    pub(crate) fn new(desc: TensorDesc<'a>, data: &'a [u8]) -> Self {
        Self { desc, data }
    }

    /// Name in the model.
    pub fn name(&self) -> &'a str {
        self.desc.name
    }

    /// Element type.
    pub fn dtype(&self) -> DataType {
        self.desc.dtype
    }

    /// Shape.
    pub fn dims(&self) -> Dims {
        self.desc.dims
    }

    /// Quantization parameters.
    pub fn quantization(&self) -> Quantization<'a> {
        self.desc.quant
    }

    /// Raw little-endian contents.
    pub fn bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Contents as a typed slice; fails if `T` does not match the tensor type.
    pub fn as_slice<T: Element>(&self) -> Result<&'a [T]> {
        if T::DTYPE != self.desc.dtype {
            return Err(type_mismatch(&self.desc));
        }
        cast_slice(self.data)
    }
}

impl AsRef<[u8]> for TensorRef<'_> {
    /// Raw little-endian contents, as [`TensorRef::bytes`].
    fn as_ref(&self) -> &[u8] {
        self.data
    }
}

/// A writable view of an input tensor of an execution context.
#[derive(Debug)]
pub struct TensorMut<'a> {
    desc: TensorDesc<'a>,
    data: &'a mut [u8],
}

impl<'a> TensorMut<'a> {
    pub(crate) fn new(desc: TensorDesc<'a>, data: &'a mut [u8]) -> Self {
        Self { desc, data }
    }

    /// Name in the model.
    pub fn name(&self) -> &'a str {
        self.desc.name
    }

    /// Element type.
    pub fn dtype(&self) -> DataType {
        self.desc.dtype
    }

    /// Shape.
    pub fn dims(&self) -> Dims {
        self.desc.dims
    }

    /// Quantization parameters.
    pub fn quantization(&self) -> Quantization<'a> {
        self.desc.quant
    }

    /// Raw little-endian contents.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// Contents as a typed slice; fails if `T` does not match the tensor type.
    pub fn as_mut_slice<T: Element>(&mut self) -> Result<&mut [T]> {
        if T::DTYPE != self.desc.dtype {
            return Err(type_mismatch(&self.desc));
        }
        cast_slice_mut(self.data)
    }
}

impl AsRef<[u8]> for TensorMut<'_> {
    /// Raw little-endian contents.
    fn as_ref(&self) -> &[u8] {
        self.data
    }
}

impl AsMut<[u8]> for TensorMut<'_> {
    /// Raw little-endian contents, as [`TensorMut::bytes_mut`].
    fn as_mut(&mut self) -> &mut [u8] {
        self.data
    }
}
