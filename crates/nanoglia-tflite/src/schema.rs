//! Field ids of the TFLite schema (`tensorflow/compiler/mlir/lite/schema/schema.fbs`).
//!
//! A union occupies two ids: its type (`*_TYPE`) then its value.

/// `table Model`.
pub(crate) mod model {
    pub(crate) const VERSION: u16 = 0;
    pub(crate) const OPERATOR_CODES: u16 = 1;
    pub(crate) const SUBGRAPHS: u16 = 2;
    pub(crate) const DESCRIPTION: u16 = 3;
    pub(crate) const BUFFERS: u16 = 4;
    pub(crate) const METADATA_BUFFER: u16 = 5;
    pub(crate) const METADATA: u16 = 6;
    pub(crate) const SIGNATURE_DEFS: u16 = 7;
    pub(crate) const EXTERNAL_BUFFER_GROUPS: u16 = 8;
    pub(crate) const EXTERNAL_BUFFERS: u16 = 9;
}

/// `table OperatorCode`.
pub(crate) mod operator_code {
    pub(crate) const DEPRECATED_BUILTIN_CODE: u16 = 0;
    pub(crate) const CUSTOM_CODE: u16 = 1;
    pub(crate) const VERSION: u16 = 2;
    pub(crate) const BUILTIN_CODE: u16 = 3;
}

/// `table SubGraph`.
pub(crate) mod subgraph {
    pub(crate) const TENSORS: u16 = 0;
    pub(crate) const INPUTS: u16 = 1;
    pub(crate) const OUTPUTS: u16 = 2;
    pub(crate) const OPERATORS: u16 = 3;
    pub(crate) const NAME: u16 = 4;
    pub(crate) const DEBUG_METADATA_INDEX: u16 = 5;
}

/// `table Tensor`.
pub(crate) mod tensor {
    pub(crate) const SHAPE: u16 = 0;
    pub(crate) const TYPE: u16 = 1;
    pub(crate) const BUFFER: u16 = 2;
    pub(crate) const NAME: u16 = 3;
    pub(crate) const QUANTIZATION: u16 = 4;
    pub(crate) const IS_VARIABLE: u16 = 5;
    pub(crate) const SPARSITY: u16 = 6;
    pub(crate) const SHAPE_SIGNATURE: u16 = 7;
    pub(crate) const HAS_RANK: u16 = 8;
    pub(crate) const VARIANT_TENSORS: u16 = 9;
    pub(crate) const EXTERNAL_BUFFER: u16 = 10;
}

/// `table VariantSubType`.
pub(crate) mod variant_sub_type {
    pub(crate) const SHAPE: u16 = 0;
    pub(crate) const TYPE: u16 = 1;
    pub(crate) const HAS_RANK: u16 = 2;
}

/// `table QuantizationParameters`.
pub(crate) mod quantization {
    pub(crate) const MIN: u16 = 0;
    pub(crate) const MAX: u16 = 1;
    pub(crate) const SCALE: u16 = 2;
    pub(crate) const ZERO_POINT: u16 = 3;
    pub(crate) const DETAILS_TYPE: u16 = 4;
    pub(crate) const QUANTIZED_DIMENSION: u16 = 6;
}

/// `table SparsityParameters`.
pub(crate) mod sparsity {
    pub(crate) const TRAVERSAL_ORDER: u16 = 0;
    pub(crate) const BLOCK_MAP: u16 = 1;
    pub(crate) const DIM_METADATA: u16 = 2;
}

/// `table DimensionMetadata`.
pub(crate) mod dimension_metadata {
    pub(crate) const FORMAT: u16 = 0;
    pub(crate) const DENSE_SIZE: u16 = 1;
    pub(crate) const ARRAY_SEGMENTS_TYPE: u16 = 2;
    pub(crate) const ARRAY_INDICES_TYPE: u16 = 4;
}

/// `union SparseIndexVector`: `Int32Vector`, `Uint16Vector`, `Uint8Vector`, each with a single
/// `values` vector field.
pub(crate) mod sparse_index_vector {
    pub(crate) const INT32: u8 = 1;
    pub(crate) const UINT16: u8 = 2;
    pub(crate) const UINT8: u8 = 3;
    pub(crate) const VALUES: u16 = 0;
}

/// `table Operator`.
pub(crate) mod operator {
    pub(crate) const OPCODE_INDEX: u16 = 0;
    pub(crate) const INPUTS: u16 = 1;
    pub(crate) const OUTPUTS: u16 = 2;
    pub(crate) const BUILTIN_OPTIONS_TYPE: u16 = 3;
    pub(crate) const CUSTOM_OPTIONS: u16 = 5;
    pub(crate) const CUSTOM_OPTIONS_FORMAT: u16 = 6;
    pub(crate) const MUTATING_VARIABLE_INPUTS: u16 = 7;
    pub(crate) const INTERMEDIATES: u16 = 8;
    pub(crate) const LARGE_CUSTOM_OPTIONS_OFFSET: u16 = 9;
    pub(crate) const LARGE_CUSTOM_OPTIONS_SIZE: u16 = 10;
    pub(crate) const BUILTIN_OPTIONS_2_TYPE: u16 = 11;
    pub(crate) const DEBUG_METADATA_INDEX: u16 = 13;
}

/// `table Buffer`.
pub(crate) mod buffer {
    pub(crate) const DATA: u16 = 0;
    pub(crate) const OFFSET: u16 = 1;
    pub(crate) const SIZE: u16 = 2;
}

/// `table Metadata`.
pub(crate) mod metadata {
    pub(crate) const NAME: u16 = 0;
    pub(crate) const BUFFER: u16 = 1;
}

/// `table SignatureDef`.
pub(crate) mod signature_def {
    pub(crate) const INPUTS: u16 = 0;
    pub(crate) const OUTPUTS: u16 = 1;
    pub(crate) const SIGNATURE_KEY: u16 = 2;
    pub(crate) const DEPRECATED_TAG: u16 = 3;
    pub(crate) const SUBGRAPH_INDEX: u16 = 4;
}

/// `table TensorMap`.
pub(crate) mod tensor_map {
    pub(crate) const NAME: u16 = 0;
    pub(crate) const TENSOR_INDEX: u16 = 1;
}

/// `table ExternalBufferGroup`.
pub(crate) mod external_buffer_group {
    pub(crate) const NAME: u16 = 0;
}

/// `table ExternalBuffer`.
pub(crate) mod external_buffer {
    pub(crate) const ID: u16 = 0;
    pub(crate) const GROUP: u16 = 1;
    pub(crate) const OFFSET: u16 = 2;
    pub(crate) const LENGTH: u16 = 3;
    pub(crate) const PACKING: u16 = 4;
}

/// `union BuiltinOptions` type codes of the options nanoglia decodes.
pub(crate) mod builtin_options {
    pub(crate) const CONV_2D: u8 = 1;
    pub(crate) const DEPTHWISE_CONV_2D: u8 = 2;
    pub(crate) const POOL_2D: u8 = 5;
    pub(crate) const FULLY_CONNECTED: u8 = 8;
    pub(crate) const SOFTMAX: u8 = 9;
    pub(crate) const CONCATENATION: u8 = 10;
    pub(crate) const ADD: u8 = 11;
    pub(crate) const MUL: u8 = 21;
    pub(crate) const REDUCER: u8 = 27;
    pub(crate) const SUB: u8 = 28;
    pub(crate) const DIV: u8 = 29;
    pub(crate) const SQUEEZE: u8 = 30;
    pub(crate) const LEAKY_RELU: u8 = 75;
}

/// `table Conv2DOptions`.
pub(crate) mod conv_2d {
    pub(crate) const PADDING: u16 = 0;
    pub(crate) const STRIDE_W: u16 = 1;
    pub(crate) const STRIDE_H: u16 = 2;
    pub(crate) const FUSED_ACTIVATION_FUNCTION: u16 = 3;
    pub(crate) const DILATION_W_FACTOR: u16 = 4;
    pub(crate) const DILATION_H_FACTOR: u16 = 5;
    pub(crate) const QUANTIZED_BIAS_TYPE: u16 = 6;
}

/// `table DepthwiseConv2DOptions`.
pub(crate) mod depthwise_conv_2d {
    pub(crate) const PADDING: u16 = 0;
    pub(crate) const STRIDE_W: u16 = 1;
    pub(crate) const STRIDE_H: u16 = 2;
    pub(crate) const DEPTH_MULTIPLIER: u16 = 3;
    pub(crate) const FUSED_ACTIVATION_FUNCTION: u16 = 4;
    pub(crate) const DILATION_W_FACTOR: u16 = 5;
    pub(crate) const DILATION_H_FACTOR: u16 = 6;
}

/// `table Pool2DOptions`.
pub(crate) mod pool_2d {
    pub(crate) const PADDING: u16 = 0;
    pub(crate) const STRIDE_W: u16 = 1;
    pub(crate) const STRIDE_H: u16 = 2;
    pub(crate) const FILTER_WIDTH: u16 = 3;
    pub(crate) const FILTER_HEIGHT: u16 = 4;
    pub(crate) const FUSED_ACTIVATION_FUNCTION: u16 = 5;
}

/// `table FullyConnectedOptions`.
pub(crate) mod fully_connected {
    pub(crate) const FUSED_ACTIVATION_FUNCTION: u16 = 0;
    pub(crate) const WEIGHTS_FORMAT: u16 = 1;
    pub(crate) const KEEP_NUM_DIMS: u16 = 2;
    pub(crate) const ASYMMETRIC_QUANTIZE_INPUTS: u16 = 3;
    pub(crate) const QUANTIZED_BIAS_TYPE: u16 = 4;
    pub(crate) const QUANT_SPEC: u16 = 5;
}

/// `table SoftmaxOptions`.
pub(crate) mod softmax {
    pub(crate) const BETA: u16 = 0;
}

/// `table ConcatenationOptions`.
pub(crate) mod concatenation {
    pub(crate) const AXIS: u16 = 0;
    pub(crate) const FUSED_ACTIVATION_FUNCTION: u16 = 1;
}

/// `table AddOptions`, `table SubOptions` (also `MulOptions`, `DivOptions` without the second field).
pub(crate) mod elementwise {
    pub(crate) const FUSED_ACTIVATION_FUNCTION: u16 = 0;
    pub(crate) const POT_SCALE_INT16: u16 = 1;
}

/// `table ReducerOptions`.
pub(crate) mod reducer {
    pub(crate) const KEEP_DIMS: u16 = 0;
}

/// `table SqueezeOptions`.
pub(crate) mod squeeze {
    pub(crate) const SQUEEZE_DIMS: u16 = 0;
}

/// `table LeakyReluOptions`.
pub(crate) mod leaky_relu {
    pub(crate) const ALPHA: u16 = 0;
}
