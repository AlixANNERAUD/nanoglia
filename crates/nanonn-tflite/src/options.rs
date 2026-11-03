//! Decoding of the `builtin_options` of the operators nanonn knows into [`OpOptions`].

use nanonn_core::le::LeScalar;
use nanonn_core::{
    Activation, BuiltinOp, Conv2DOptions, DepthwiseConv2DOptions, Detail, Error,
    FullyConnectedOptions, OpOptions, Padding, Pool2DOptions, Result,
};

use crate::fb::Table;
use crate::schema::{
    builtin_options as ty, concatenation, conv_2d, depthwise_conv_2d, elementwise, fully_connected,
    leaky_relu, operator, pool_2d, reducer, softmax, squeeze,
};

/// The `BuiltinOptions` union type carried by `op`, for the operators whose options are decoded.
pub(crate) fn expected_type(op: BuiltinOp) -> Option<u8> {
    Some(match op {
        BuiltinOp::CONV_2D => ty::CONV_2D,
        BuiltinOp::DEPTHWISE_CONV_2D => ty::DEPTHWISE_CONV_2D,
        BuiltinOp::FULLY_CONNECTED => ty::FULLY_CONNECTED,
        BuiltinOp::AVERAGE_POOL_2D | BuiltinOp::MAX_POOL_2D => ty::POOL_2D,
        BuiltinOp::ADD => ty::ADD,
        BuiltinOp::SUB => ty::SUB,
        BuiltinOp::MUL => ty::MUL,
        BuiltinOp::DIV => ty::DIV,
        BuiltinOp::SOFTMAX => ty::SOFTMAX,
        BuiltinOp::CONCATENATION => ty::CONCATENATION,
        BuiltinOp::MEAN
        | BuiltinOp::SUM
        | BuiltinOp::REDUCE_MAX
        | BuiltinOp::REDUCE_MIN
        | BuiltinOp::REDUCE_PROD => ty::REDUCER,
        BuiltinOp::LEAKY_RELU => ty::LEAKY_RELU,
        BuiltinOp::SQUEEZE => ty::SQUEEZE,
        _ => return None,
    })
}

/// Checks that the fields `decode` reads from an options table of union type `ty` are in bounds.
///
/// Tables of other types are only checked as tables by the caller: nanonn never reads them.
pub(crate) fn verify(ty: u8, table: &Table<'_>) -> Result<()> {
    const BYTE: usize = 1;
    const INT: usize = 4;
    let scalars: &[(u16, usize)] = match ty {
        ty::CONV_2D => &[
            (conv_2d::PADDING, BYTE),
            (conv_2d::STRIDE_W, INT),
            (conv_2d::STRIDE_H, INT),
            (conv_2d::FUSED_ACTIVATION_FUNCTION, BYTE),
            (conv_2d::DILATION_W_FACTOR, INT),
            (conv_2d::DILATION_H_FACTOR, INT),
            (conv_2d::QUANTIZED_BIAS_TYPE, BYTE),
        ],
        ty::DEPTHWISE_CONV_2D => &[
            (depthwise_conv_2d::PADDING, BYTE),
            (depthwise_conv_2d::STRIDE_W, INT),
            (depthwise_conv_2d::STRIDE_H, INT),
            (depthwise_conv_2d::DEPTH_MULTIPLIER, INT),
            (depthwise_conv_2d::FUSED_ACTIVATION_FUNCTION, BYTE),
            (depthwise_conv_2d::DILATION_W_FACTOR, INT),
            (depthwise_conv_2d::DILATION_H_FACTOR, INT),
        ],
        ty::POOL_2D => &[
            (pool_2d::PADDING, BYTE),
            (pool_2d::STRIDE_W, INT),
            (pool_2d::STRIDE_H, INT),
            (pool_2d::FILTER_WIDTH, INT),
            (pool_2d::FILTER_HEIGHT, INT),
            (pool_2d::FUSED_ACTIVATION_FUNCTION, BYTE),
        ],
        ty::FULLY_CONNECTED => &[
            (fully_connected::FUSED_ACTIVATION_FUNCTION, BYTE),
            (fully_connected::WEIGHTS_FORMAT, BYTE),
            (fully_connected::KEEP_NUM_DIMS, BYTE),
            (fully_connected::ASYMMETRIC_QUANTIZE_INPUTS, BYTE),
            (fully_connected::QUANTIZED_BIAS_TYPE, BYTE),
        ],
        ty::SOFTMAX => &[(softmax::BETA, INT)],
        ty::CONCATENATION => &[
            (concatenation::AXIS, INT),
            (concatenation::FUSED_ACTIVATION_FUNCTION, BYTE),
        ],
        ty::ADD | ty::SUB => &[
            (elementwise::FUSED_ACTIVATION_FUNCTION, BYTE),
            (elementwise::POT_SCALE_INT16, BYTE),
        ],
        ty::MUL | ty::DIV => &[(elementwise::FUSED_ACTIVATION_FUNCTION, BYTE)],
        ty::REDUCER => &[(reducer::KEEP_DIMS, BYTE)],
        ty::LEAKY_RELU => &[(leaky_relu::ALPHA, INT)],
        _ => &[],
    };
    for &(id, size) in scalars {
        table.field(id, size)?;
    }
    match ty {
        ty::FULLY_CONNECTED => {
            table.bytes(fully_connected::QUANT_SPEC)?;
        }
        ty::SQUEEZE => {
            table.slice::<i32>(squeeze::SQUEEZE_DIMS)?;
        }
        _ => {}
    }
    Ok(())
}

/// Reads the fields of an options table, falling back to the schema defaults when the table or
/// a field is absent.
struct Fields<'m> {
    table: Option<Table<'m>>,
    /// The error returned for enum values nanonn does not know.
    unsupported: Error,
}

impl<'m> Fields<'m> {
    fn scalar<T: LeScalar>(&self, id: u16, default: T) -> Result<T> {
        match &self.table {
            Some(table) => table.scalar(id, default),
            None => Ok(default),
        }
    }

    fn flag(&self, id: u16) -> Result<bool> {
        Ok(self.scalar::<u8>(id, 0)? != 0)
    }

    fn padding(&self, id: u16) -> Result<Padding> {
        match self.scalar::<i8>(id, 0)? {
            0 => Ok(Padding::Same),
            1 => Ok(Padding::Valid),
            _ => Err(self.unsupported),
        }
    }

    fn activation(&self, id: u16) -> Result<Activation> {
        Activation::from_tflite(self.scalar(id, 0)?).ok_or(self.unsupported)
    }
}

/// Decodes the options of the builtin operator `op` stored in the `Operator` table `operator`.
///
/// Errors: [`Error::malformed`] if the options table has the type of another operator,
/// [`Detail::UnsupportedOp`] for enum values or fields nanonn does not know.
pub(crate) fn decode<'m>(
    op: BuiltinOp,
    operator: &Table<'m>,
    op_index: u32,
) -> Result<OpOptions<'m>> {
    let Some(expected) = expected_type(op) else {
        return Ok(OpOptions::None);
    };
    let (actual, value) = operator.union(operator::BUILTIN_OPTIONS_TYPE)?;
    let table = match actual {
        0 => None,
        actual if actual == expected => value,
        _ => return Err(Error::malformed(operator.pos())),
    };
    let f = Fields {
        table,
        unsupported: Error::unsupported(Detail::UnsupportedOp {
            op_index,
            opcode: op.0,
        }),
    };
    Ok(match expected {
        ty::CONV_2D => OpOptions::Conv2D(Conv2DOptions {
            padding: f.padding(conv_2d::PADDING)?,
            stride_w: f.scalar(conv_2d::STRIDE_W, 0)?,
            stride_h: f.scalar(conv_2d::STRIDE_H, 0)?,
            dilation_w: f.scalar(conv_2d::DILATION_W_FACTOR, 1)?,
            dilation_h: f.scalar(conv_2d::DILATION_H_FACTOR, 1)?,
            activation: f.activation(conv_2d::FUSED_ACTIVATION_FUNCTION)?,
        }),
        ty::DEPTHWISE_CONV_2D => OpOptions::DepthwiseConv2D(DepthwiseConv2DOptions {
            padding: f.padding(depthwise_conv_2d::PADDING)?,
            stride_w: f.scalar(depthwise_conv_2d::STRIDE_W, 0)?,
            stride_h: f.scalar(depthwise_conv_2d::STRIDE_H, 0)?,
            depth_multiplier: f.scalar(depthwise_conv_2d::DEPTH_MULTIPLIER, 0)?,
            dilation_w: f.scalar(depthwise_conv_2d::DILATION_W_FACTOR, 1)?,
            dilation_h: f.scalar(depthwise_conv_2d::DILATION_H_FACTOR, 1)?,
            activation: f.activation(depthwise_conv_2d::FUSED_ACTIVATION_FUNCTION)?,
        }),
        ty::FULLY_CONNECTED => {
            // A non-standard quantization contract must be rejected by runtimes that do not
            // understand it (see the schema).
            if let Some(table) = &f.table
                && !table.bytes(fully_connected::QUANT_SPEC)?.is_empty()
            {
                return Err(f.unsupported);
            }
            OpOptions::FullyConnected(FullyConnectedOptions {
                activation: f.activation(fully_connected::FUSED_ACTIVATION_FUNCTION)?,
                default_weights_format: match f.scalar::<i8>(fully_connected::WEIGHTS_FORMAT, 0)? {
                    0 => true,  // DEFAULT
                    1 => false, // SHUFFLED4x16INT8
                    _ => return Err(f.unsupported),
                },
                keep_num_dims: f.flag(fully_connected::KEEP_NUM_DIMS)?,
                asymmetric_quantize_inputs: f.flag(fully_connected::ASYMMETRIC_QUANTIZE_INPUTS)?,
            })
        }
        ty::POOL_2D => OpOptions::Pool2D(Pool2DOptions {
            padding: f.padding(pool_2d::PADDING)?,
            stride_w: f.scalar(pool_2d::STRIDE_W, 0)?,
            stride_h: f.scalar(pool_2d::STRIDE_H, 0)?,
            filter_w: f.scalar(pool_2d::FILTER_WIDTH, 0)?,
            filter_h: f.scalar(pool_2d::FILTER_HEIGHT, 0)?,
            activation: f.activation(pool_2d::FUSED_ACTIVATION_FUNCTION)?,
        }),
        ty::ADD | ty::SUB | ty::MUL | ty::DIV => OpOptions::Elementwise {
            activation: f.activation(elementwise::FUSED_ACTIVATION_FUNCTION)?,
        },
        ty::SOFTMAX => OpOptions::Softmax {
            beta: f.scalar(softmax::BETA, 0.0)?,
        },
        ty::CONCATENATION => OpOptions::Concatenation {
            axis: f.scalar(concatenation::AXIS, 0)?,
            activation: f.activation(concatenation::FUSED_ACTIVATION_FUNCTION)?,
        },
        ty::REDUCER => OpOptions::Reducer {
            keep_dims: f.flag(reducer::KEEP_DIMS)?,
        },
        ty::LEAKY_RELU => OpOptions::LeakyRelu {
            alpha: f.scalar(leaky_relu::ALPHA, 0.0)?,
        },
        ty::SQUEEZE => OpOptions::Squeeze {
            dims: match &f.table {
                Some(table) => table.slice(squeeze::SQUEEZE_DIMS)?,
                None => nanonn_core::le::LeSlice::EMPTY,
            },
        },
        _ => OpOptions::None,
    })
}
