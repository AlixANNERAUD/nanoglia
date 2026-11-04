use crate::TensorDesc;

/// A TFLite builtin operator code (`BuiltinOperator` in the TFLite schema).
///
/// nanoglia uses the TFLite operator vocabulary as its canonical operator set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct BuiltinOp(pub i32);

#[allow(missing_docs)]
impl BuiltinOp {
    pub const ADD: Self = Self(0);
    pub const AVERAGE_POOL_2D: Self = Self(1);
    pub const CONCATENATION: Self = Self(2);
    pub const CONV_2D: Self = Self(3);
    pub const DEPTHWISE_CONV_2D: Self = Self(4);
    pub const DEPTH_TO_SPACE: Self = Self(5);
    pub const DEQUANTIZE: Self = Self(6);
    pub const FLOOR: Self = Self(8);
    pub const FULLY_CONNECTED: Self = Self(9);
    pub const L2_NORMALIZATION: Self = Self(11);
    pub const LOGISTIC: Self = Self(14);
    pub const LSTM: Self = Self(16);
    pub const MAX_POOL_2D: Self = Self(17);
    pub const MUL: Self = Self(18);
    pub const RELU: Self = Self(19);
    pub const RELU_N1_TO_1: Self = Self(20);
    pub const RELU6: Self = Self(21);
    pub const RESHAPE: Self = Self(22);
    pub const RESIZE_BILINEAR: Self = Self(23);
    pub const SOFTMAX: Self = Self(25);
    pub const SPACE_TO_DEPTH: Self = Self(26);
    pub const SVDF: Self = Self(27);
    pub const TANH: Self = Self(28);
    pub const CUSTOM: Self = Self(32);
    pub const PAD: Self = Self(34);
    pub const GATHER: Self = Self(36);
    pub const TRANSPOSE: Self = Self(39);
    pub const MEAN: Self = Self(40);
    pub const SUB: Self = Self(41);
    pub const DIV: Self = Self(42);
    pub const SQUEEZE: Self = Self(43);
    pub const UNIDIRECTIONAL_SEQUENCE_LSTM: Self = Self(44);
    pub const STRIDED_SLICE: Self = Self(45);
    pub const EXP: Self = Self(47);
    pub const SPLIT: Self = Self(49);
    pub const LOG_SOFTMAX: Self = Self(50);
    pub const DELEGATE: Self = Self(51);
    pub const CAST: Self = Self(53);
    pub const PRELU: Self = Self(54);
    pub const MAXIMUM: Self = Self(55);
    pub const ARG_MAX: Self = Self(56);
    pub const MINIMUM: Self = Self(57);
    pub const PADV2: Self = Self(60);
    pub const SLICE: Self = Self(65);
    pub const TRANSPOSE_CONV: Self = Self(67);
    pub const EXPAND_DIMS: Self = Self(70);
    pub const SUM: Self = Self(74);
    pub const SQRT: Self = Self(75);
    pub const RSQRT: Self = Self(76);
    pub const SHAPE: Self = Self(77);
    pub const POW: Self = Self(78);
    pub const ARG_MIN: Self = Self(79);
    pub const REDUCE_PROD: Self = Self(81);
    pub const REDUCE_MAX: Self = Self(82);
    pub const PACK: Self = Self(83);
    pub const UNPACK: Self = Self(88);
    pub const REDUCE_MIN: Self = Self(89);
    pub const SQUARE: Self = Self(92);
    pub const LEAKY_RELU: Self = Self(98);
    pub const SQUARED_DIFFERENCE: Self = Self(99);
    pub const MIRROR_PAD: Self = Self(100);
    pub const ABS: Self = Self(101);
    pub const SPLIT_V: Self = Self(102);
    pub const ADD_N: Self = Self(106);
    pub const RESIZE_NEAREST_NEIGHBOR: Self = Self(97);
    pub const ELU: Self = Self(111);
    pub const QUANTIZE: Self = Self(114);
    pub const HARD_SWISH: Self = Self(117);
    pub const BATCH_MATMUL: Self = Self(126);
}

/// The code identifying an operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpCode<'a> {
    /// A builtin operator.
    Builtin(BuiltinOp),
    /// A custom operator, identified by its `custom_code` string.
    Custom(&'a str),
}

/// Padding scheme of convolutions and pooling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum Padding {
    /// Output has the same spatial size as the input divided by the stride (TFLite `SAME`).
    #[default]
    Same,
    /// No padding (TFLite `VALID`).
    Valid,
}

/// Activation fused into an operator (TFLite `ActivationFunctionType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum Activation {
    /// No activation.
    #[default]
    None,
    /// `max(x, 0)`.
    Relu,
    /// `clamp(x, -1, 1)`.
    ReluN1To1,
    /// `clamp(x, 0, 6)`.
    Relu6,
    /// `tanh(x)`.
    Tanh,
    /// Sign bit.
    SignBit,
}

impl Activation {
    /// Decodes a TFLite `ActivationFunctionType` code.
    pub const fn from_tflite(code: i8) -> Option<Self> {
        Some(match code {
            0 => Self::None,
            1 => Self::Relu,
            2 => Self::ReluN1To1,
            3 => Self::Relu6,
            4 => Self::Tanh,
            5 => Self::SignBit,
            _ => return None,
        })
    }
}

/// Options of `CONV_2D`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Conv2DOptions {
    /// Padding scheme.
    pub padding: Padding,
    /// Horizontal stride.
    pub stride_w: i32,
    /// Vertical stride.
    pub stride_h: i32,
    /// Horizontal dilation.
    pub dilation_w: i32,
    /// Vertical dilation.
    pub dilation_h: i32,
    /// Fused activation.
    pub activation: Activation,
}

/// Options of `DEPTHWISE_CONV_2D`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct DepthwiseConv2DOptions {
    /// Padding scheme.
    pub padding: Padding,
    /// Horizontal stride.
    pub stride_w: i32,
    /// Vertical stride.
    pub stride_h: i32,
    /// Output channels per input channel.
    pub depth_multiplier: i32,
    /// Horizontal dilation.
    pub dilation_w: i32,
    /// Vertical dilation.
    pub dilation_h: i32,
    /// Fused activation.
    pub activation: Activation,
}

/// Options of `FULLY_CONNECTED`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FullyConnectedOptions {
    /// Fused activation.
    pub activation: Activation,
    /// Whether the weights use the default layout (`false` for shuffled formats, unsupported).
    pub default_weights_format: bool,
    /// Whether the output keeps the input's leading dimensions.
    pub keep_num_dims: bool,
    /// Whether float inputs are quantized on the fly (hybrid models).
    pub asymmetric_quantize_inputs: bool,
}

/// Options of `AVERAGE_POOL_2D` and `MAX_POOL_2D`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Pool2DOptions {
    /// Padding scheme.
    pub padding: Padding,
    /// Horizontal stride.
    pub stride_w: i32,
    /// Vertical stride.
    pub stride_h: i32,
    /// Window width.
    pub filter_w: i32,
    /// Window height.
    pub filter_h: i32,
    /// Fused activation.
    pub activation: Activation,
}

/// Options of `STRIDED_SLICE`.
///
/// Bit `i` of a mask refers to element `i` of the `begin`, `end` and `strides` inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct StridedSliceOptions {
    /// Axes whose `begin` is ignored: the slice starts at the first element (the last one for a
    /// negative stride).
    pub begin_mask: i32,
    /// Axes whose `end` is ignored: the slice runs to the last element (the first one for a
    /// negative stride).
    pub end_mask: i32,
    /// The axis standing for as many full axes as needed to cover the input (`...`).
    pub ellipsis_mask: i32,
    /// Axes inserted with size 1 (`tf.newaxis`).
    pub new_axis_mask: i32,
    /// Axes taken at index `begin` and removed from the output.
    pub shrink_axis_mask: i32,
    /// Whether `end` holds sizes relative to `begin` instead of indices.
    pub offset: bool,
}

/// Options decoded from the model for the operators nanoglia knows.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[non_exhaustive]
pub enum OpOptions<'a> {
    /// The operator has no options, or they are not needed.
    #[default]
    None,
    /// `CONV_2D`.
    Conv2D(Conv2DOptions),
    /// `DEPTHWISE_CONV_2D`.
    DepthwiseConv2D(DepthwiseConv2DOptions),
    /// `FULLY_CONNECTED`.
    FullyConnected(FullyConnectedOptions),
    /// `AVERAGE_POOL_2D`, `MAX_POOL_2D`.
    Pool2D(Pool2DOptions),
    /// `ADD`, `SUB`, `MUL`, `DIV`: the fused activation.
    Elementwise {
        /// Fused activation.
        activation: Activation,
    },
    /// `SOFTMAX`.
    Softmax {
        /// Inverse temperature.
        beta: f32,
    },
    /// `CONCATENATION`.
    Concatenation {
        /// Concatenation axis (may be negative).
        axis: i32,
        /// Fused activation.
        activation: Activation,
    },
    /// `MEAN`, `SUM`, `REDUCE_MAX`...
    Reducer {
        /// Whether reduced dimensions are kept with size 1.
        keep_dims: bool,
    },
    /// `LEAKY_RELU`.
    LeakyRelu {
        /// Slope for negative inputs.
        alpha: f32,
    },
    /// `SQUEEZE`: the axes to squeeze, as a little-endian `i32` array.
    Squeeze {
        /// The axes to squeeze; empty means all dimensions of size 1.
        dims: crate::le::LeSlice<'a, i32>,
    },
    /// `STRIDED_SLICE`.
    StridedSlice(StridedSliceOptions),
    /// `GATHER`.
    Gather {
        /// The axis of the input the indices select along (may be negative).
        axis: i32,
        /// The number of leading batch dimensions shared by the input and the indices (may be
        /// negative).
        batch_dims: i32,
    },
    /// `SPLIT`, `SPLIT_V`.
    Split {
        /// The number of outputs.
        num_splits: i32,
    },
    /// Raw options of a custom operator.
    Custom(&'a [u8]),
}

/// The maximum number of inputs of an operator nanoglia can run.
pub const MAX_OP_INPUTS: usize = 16;
/// The maximum number of outputs of an operator nanoglia can run.
pub const MAX_OP_OUTPUTS: usize = 8;

/// Everything a backend needs to decide whether it supports an operator and to prepare it.
#[derive(Clone, Copy, Debug)]
pub struct OpDesc<'a, 'b> {
    /// Index of the operator in the graph (execution order).
    pub index: u32,
    /// Operator code.
    pub code: OpCode<'a>,
    /// Decoded options.
    pub options: OpOptions<'a>,
    /// Inputs in operator order; `None` for omitted optional inputs (TFLite index `-1`).
    pub inputs: &'b [Option<TensorDesc<'a>>],
    /// Outputs in operator order.
    pub outputs: &'b [TensorDesc<'a>],
}

impl<'a> OpDesc<'a, '_> {
    /// The builtin code, if the operator is a builtin.
    pub fn builtin(&self) -> Option<BuiltinOp> {
        match self.code {
            OpCode::Builtin(op) => Some(op),
            OpCode::Custom(_) => None,
        }
    }

    /// The input at `i`, if present.
    pub fn input(&self, i: usize) -> Option<&TensorDesc<'a>> {
        self.inputs.get(i).and_then(Option::as_ref)
    }

    /// The output at `i`, if present.
    pub fn output(&self, i: usize) -> Option<&TensorDesc<'a>> {
        self.outputs.get(i)
    }

    /// The TFLite opcode number used in error details (`32` for custom operators).
    pub fn opcode_number(&self) -> i32 {
        match self.code {
            OpCode::Builtin(op) => op.0,
            OpCode::Custom(_) => BuiltinOp::CUSTOM.0,
        }
    }
}
