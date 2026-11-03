use crate::le::LeSlice;

/// The element type of a tensor. Discriminants are the TFLite `TensorType` codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[non_exhaustive]
pub enum DataType {
    /// 32-bit float.
    F32 = 0,
    /// 16-bit IEEE float.
    F16 = 1,
    /// 32-bit signed integer.
    I32 = 2,
    /// 8-bit unsigned integer.
    U8 = 3,
    /// 64-bit signed integer.
    I64 = 4,
    /// Boolean, one byte per element.
    Bool = 6,
    /// 16-bit signed integer.
    I16 = 7,
    /// 8-bit signed integer, the main quantized type.
    I8 = 9,
    /// 64-bit float.
    F64 = 10,
    /// 64-bit unsigned integer.
    U64 = 12,
    /// 32-bit unsigned integer.
    U32 = 15,
    /// 16-bit unsigned integer.
    U16 = 16,
    /// 4-bit signed integer, two elements per byte.
    I4 = 17,
    /// 16-bit brain float.
    BF16 = 18,
}

impl DataType {
    /// Decodes a TFLite `TensorType` code.
    pub const fn from_tflite(code: i8) -> Option<Self> {
        Some(match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::I32,
            3 => Self::U8,
            4 => Self::I64,
            6 => Self::Bool,
            7 => Self::I16,
            9 => Self::I8,
            10 => Self::F64,
            12 => Self::U64,
            15 => Self::U32,
            16 => Self::U16,
            17 => Self::I4,
            18 => Self::BF16,
            _ => return None,
        })
    }

    /// Size of one element in bytes, or `None` for packed sub-byte types.
    pub const fn size(self) -> Option<usize> {
        Some(match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::I4 => return None,
        })
    }

    /// Number of bytes needed to store `elements` elements.
    pub const fn bytes_for(self, elements: usize) -> Option<usize> {
        match self.size() {
            Some(size) => elements.checked_mul(size),
            None => Some(elements.div_ceil(2)),
        }
    }
}

/// The maximum rank of a tensor.
pub const MAX_RANK: usize = 6;

/// The shape of a tensor, stored inline (no allocation).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dims {
    rank: u8,
    dims: [i32; MAX_RANK],
}

impl core::fmt::Debug for Dims {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for Dims {
    fn format(&self, f: defmt::Formatter<'_>) {
        defmt::write!(f, "{=[?]}", self.as_slice())
    }
}

impl Default for Dims {
    fn default() -> Self {
        Self::scalar()
    }
}

impl Dims {
    /// The shape of a scalar (rank 0).
    pub const fn scalar() -> Self {
        Self {
            rank: 0,
            dims: [0; MAX_RANK],
        }
    }

    /// A shape from its dimensions, or `None` if the rank exceeds [`MAX_RANK`].
    pub fn new(dims: &[i32]) -> Option<Self> {
        if dims.len() > MAX_RANK {
            return None;
        }
        let mut out = Self {
            rank: dims.len() as u8,
            dims: [0; MAX_RANK],
        };
        out.dims[..dims.len()].copy_from_slice(dims);
        Some(out)
    }

    /// A shape from an iterator of dimensions, or `None` if the rank exceeds [`MAX_RANK`].
    pub fn collect(dims: impl IntoIterator<Item = i32>) -> Option<Self> {
        let mut out = Self::scalar();
        for d in dims {
            if out.rank as usize == MAX_RANK {
                return None;
            }
            out.dims[out.rank as usize] = d;
            out.rank += 1;
        }
        Some(out)
    }

    /// The rank (number of dimensions).
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank as usize
    }

    /// The dimensions.
    #[inline]
    pub fn as_slice(&self) -> &[i32] {
        &self.dims[..self.rank as usize]
    }

    /// The dimension at `axis`, if any.
    #[inline]
    pub fn get(&self, axis: usize) -> Option<i32> {
        self.as_slice().get(axis).copied()
    }

    /// The dimension at `axis` as `usize`, `0` if out of range or negative.
    #[inline]
    pub fn dim(&self, axis: usize) -> usize {
        self.get(axis).map_or(0, |d| d.max(0) as usize)
    }

    /// The number of elements, or `None` for negative (dynamic) dimensions or overflow.
    pub fn num_elements(&self) -> Option<usize> {
        self.as_slice().iter().try_fold(1usize, |acc, &d| {
            if d < 0 {
                None
            } else {
                acc.checked_mul(d as usize)
            }
        })
    }
}

/// The quantization parameters of a tensor, borrowed from the model.
///
/// `real = scale * (quantized - zero_point)`. Per-tensor quantization has one scale; per-axis
/// (per-channel) quantization has one scale per slice along [`Self::axis`].
#[derive(Clone, Copy, Debug)]
pub struct Quantization<'a> {
    /// The scales.
    pub scale: LeSlice<'a, f32>,
    /// The zero points.
    pub zero_point: LeSlice<'a, i64>,
    /// The quantized dimension for per-axis quantization.
    pub axis: i32,
}

impl Default for Quantization<'_> {
    fn default() -> Self {
        Self::NONE
    }
}

impl<'a> Quantization<'a> {
    /// No quantization.
    pub const NONE: Self = Self {
        scale: LeSlice::EMPTY,
        zero_point: LeSlice::EMPTY,
        axis: 0,
    };

    /// Whether the tensor carries quantization parameters.
    pub fn is_quantized(&self) -> bool {
        !self.scale.is_empty()
    }

    /// Whether there is a single scale for the whole tensor.
    pub fn is_per_tensor(&self) -> bool {
        self.scale.len() == 1
    }

    /// The first scale (the only one for per-tensor quantization), `0.0` if not quantized.
    pub fn scale(&self) -> f32 {
        self.scale.get(0).unwrap_or(0.0)
    }

    /// The first zero point, `0` if absent or out of `i32` range.
    pub fn zero_point(&self) -> i32 {
        self.zero_point
            .get(0)
            .and_then(|z| i32::try_from(z).ok())
            .unwrap_or(0)
    }
}

/// Everything known before execution about a tensor of the graph.
#[derive(Clone, Copy, Debug)]
pub struct TensorDesc<'a> {
    /// Index of the tensor in the graph.
    pub index: u32,
    /// Element type.
    pub dtype: DataType,
    /// Shape.
    pub dims: Dims,
    /// Quantization parameters.
    pub quant: Quantization<'a>,
    /// Constant contents (weights, biases...), borrowed from the model; `None` for activations.
    pub data: Option<&'a [u8]>,
    /// Name of the tensor in the model, possibly empty.
    pub name: &'a str,
}

impl TensorDesc<'_> {
    /// Size in bytes of the tensor contents, or `None` for dynamic shapes or overflow.
    pub fn byte_len(&self) -> Option<usize> {
        self.dtype.bytes_for(self.dims.num_elements()?)
    }

    /// Whether the tensor is a constant.
    pub fn is_constant(&self) -> bool {
        self.data.is_some()
    }
}
