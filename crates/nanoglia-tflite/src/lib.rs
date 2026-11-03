#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

use core::fmt;

use nanoglia_core::le::LeSlice;
use nanoglia_core::{
    BuiltinOp, DataType, Detail, Dims, Error, ErrorCode, OpCode, OpOptions, Quantization, Result,
    TensorDesc,
};

mod fb;
mod options;
mod schema;
mod verify;

use fb::{Table, Tables};
use schema::{buffer, model, operator, operator_code, quantization, subgraph, tensor};

/// The flatbuffer file identifier of TFLite models, at byte offset 4.
pub const FILE_IDENTIFIER: &[u8; 4] = b"TFL3";

/// The only TFLite schema version nanoglia reads.
pub const SCHEMA_VERSION: u32 = 3;

/// A verified TFLite model, borrowing the model bytes.
///
/// Created by [`Model::parse`], which verifies the whole flatbuffer once. Accessors then return
/// views into the model bytes without copying them.
#[derive(Clone, Copy)]
pub struct Model<'m> {
    bytes: &'m [u8],
    version: u32,
    description: &'m str,
    opcodes: Tables<'m>,
    subgraphs: Tables<'m>,
    buffers: Tables<'m>,
}

impl fmt::Debug for Model<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("version", &self.version)
            .field("description", &self.description)
            .field("subgraphs", &self.subgraphs.len())
            .field("buffers", &self.buffers.len())
            .finish_non_exhaustive()
    }
}

impl<'m> Model<'m> {
    /// Whether `bytes` carries the TFLite file identifier (`TFL3` at offset 4).
    pub fn is_tflite(bytes: &[u8]) -> bool {
        bytes.get(4..8) == Some(&FILE_IDENTIFIER[..])
    }

    /// Verifies `bytes` as a TFLite model and returns a view over it.
    ///
    /// Verification covers every table reachable from the root: offsets, vtables, vectors and
    /// strings are in bounds, strings are UTF-8, external buffers (`Buffer.offset/size`) fit in
    /// `bytes`, and operator codes, buffers and tensor indices referenced by subgraphs and
    /// operators are in range. Operator options must have the type of their operator.
    ///
    /// # Errors
    ///
    /// - [`Detail::UnknownEncoding`] (code [`ErrorCode::InvalidEncoding`]) if `bytes` does not
    ///   carry the `TFL3` identifier;
    /// - [`Detail::MalformedModel`] with the offending offset if verification fails;
    /// - [`Detail::UnsupportedSchema`] if the schema version is not [`SCHEMA_VERSION`].
    pub fn parse(bytes: &'m [u8]) -> Result<Self> {
        if !Self::is_tflite(bytes) {
            return Err(Error::new(
                ErrorCode::InvalidEncoding,
                Detail::UnknownEncoding,
            ));
        }
        let root = Table::at(bytes, fb::follow(bytes, 0)?)?;
        let version: u32 = root.scalar(model::VERSION, 0)?;
        if version != SCHEMA_VERSION {
            return Err(Error::unsupported(Detail::UnsupportedSchema { version }));
        }
        verify::model(&root)?;
        Ok(Self {
            bytes,
            version,
            description: root.str(model::DESCRIPTION)?.unwrap_or(""),
            opcodes: root.tables(model::OPERATOR_CODES)?,
            subgraphs: root.tables(model::SUBGRAPHS)?,
            buffers: root.tables(model::BUFFERS)?,
        })
    }

    /// The schema version, always [`SCHEMA_VERSION`].
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The model description, empty if absent.
    pub fn description(&self) -> &'m str {
        self.description
    }

    /// The model bytes.
    pub fn bytes(&self) -> &'m [u8] {
        self.bytes
    }

    /// The number of subgraphs; subgraph 0 is the main graph.
    pub fn subgraph_count(&self) -> usize {
        self.subgraphs.len()
    }

    /// The subgraph at `index`.
    ///
    /// # Errors
    ///
    /// [`Detail::IndexOutOfRange`] if `index >= self.subgraph_count()`.
    pub fn subgraph(&self, index: usize) -> Result<SubGraph<'m>> {
        let table = get(&self.subgraphs, index)?;
        Ok(SubGraph {
            model: *self,
            name: table.str(subgraph::NAME)?.unwrap_or(""),
            tensors: table.tables(subgraph::TENSORS)?,
            operators: table.tables(subgraph::OPERATORS)?,
            inputs: table.slice(subgraph::INPUTS)?,
            outputs: table.slice(subgraph::OUTPUTS)?,
        })
    }

    /// The contents of buffer `index`, `None` for buffer 0 (the empty sentinel) and empty buffers.
    fn buffer_data(&self, index: u32) -> Result<Option<&'m [u8]>> {
        if index == 0 {
            return Ok(None);
        }
        let table = self
            .buffers
            .get(index as usize)
            .map_err(|_| Error::malformed(0))?;
        let offset: u64 = table.scalar(buffer::OFFSET, 0)?;
        let data = if offset > 1 {
            // Data stored after the flatbuffer, at an offset from the start of the file.
            let size: u64 = table.scalar(buffer::SIZE, 0)?;
            external(self.bytes, offset, size).ok_or(Error::malformed(table.pos()))?
        } else {
            table.bytes(buffer::DATA)?
        };
        Ok((!data.is_empty()).then_some(data))
    }
}

/// A subgraph of a [`Model`]: tensors and operators in execution order.
#[derive(Clone, Copy)]
pub struct SubGraph<'m> {
    model: Model<'m>,
    name: &'m str,
    tensors: Tables<'m>,
    operators: Tables<'m>,
    inputs: LeSlice<'m, i32>,
    outputs: LeSlice<'m, i32>,
}

impl fmt::Debug for SubGraph<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubGraph")
            .field("name", &self.name)
            .field("tensors", &self.tensors.len())
            .field("operators", &self.operators.len())
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish()
    }
}

impl<'m> SubGraph<'m> {
    /// The subgraph name, empty if absent.
    pub fn name(&self) -> &'m str {
        self.name
    }

    /// The number of tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// The tensor at `index`, with [`TensorDesc::index`] set to `index`.
    ///
    /// The shape is the `shape` field (`shape_signature` is ignored). Constant data is borrowed
    /// from the model; buffer 0 and empty buffers give `data: None`.
    ///
    /// # Errors
    ///
    /// - [`Detail::IndexOutOfRange`] if `index >= self.tensor_count()`;
    /// - [`Detail::UnsupportedType`] (code [`ErrorCode::UnsupportedOperation`]) if the tensor
    ///   cannot be represented: unknown or unsupported element type (string, complex, resource,
    ///   variant...), rank above [`nanoglia_core::MAX_RANK`], sparse tensor, data in an external
    ///   file, or a non-affine quantization scheme (`QuantizationParameters.details`).
    pub fn tensor(&self, index: usize) -> Result<TensorDesc<'m>> {
        let t = get(&self.tensors, index)?;
        let id = u32::try_from(index).unwrap_or(u32::MAX);
        let unsupported = Error::unsupported(Detail::UnsupportedType { tensor: id });

        let dtype = DataType::from_tflite(t.scalar(tensor::TYPE, 0)?).ok_or(unsupported)?;
        let shape = t.slice::<i32>(tensor::SHAPE)?;
        let dims = Dims::collect(shape.iter()).ok_or(unsupported)?;
        if t.has(tensor::SPARSITY)? || t.scalar::<u32>(tensor::EXTERNAL_BUFFER, 0)? != 0 {
            return Err(unsupported);
        }
        let quant = match t.table(tensor::QUANTIZATION)? {
            None => Quantization::NONE,
            Some(q) => {
                if q.scalar::<u8>(quantization::DETAILS_TYPE, 0)? != 0 {
                    return Err(unsupported);
                }
                Quantization {
                    scale: q.slice(quantization::SCALE)?,
                    zero_point: q.slice(quantization::ZERO_POINT)?,
                    axis: q.scalar(quantization::QUANTIZED_DIMENSION, 0)?,
                }
            }
        };
        Ok(TensorDesc {
            index: id,
            dtype,
            dims,
            quant,
            data: self.model.buffer_data(t.scalar(tensor::BUFFER, 0)?)?,
            name: t.str(tensor::NAME)?.unwrap_or(""),
        })
    }

    /// Whether the tensor at `index` is a variable (state kept between invocations).
    ///
    /// # Errors
    ///
    /// [`Detail::IndexOutOfRange`] if `index >= self.tensor_count()`.
    pub fn tensor_is_variable(&self, index: usize) -> Result<bool> {
        get(&self.tensors, index)?.flag(tensor::IS_VARIABLE, false)
    }

    /// The indices of the input tensors, all in range.
    pub fn inputs(&self) -> LeSlice<'m, i32> {
        self.inputs
    }

    /// The indices of the output tensors, all in range.
    pub fn outputs(&self) -> LeSlice<'m, i32> {
        self.outputs
    }

    /// The number of operators.
    pub fn operator_count(&self) -> usize {
        self.operators.len()
    }

    /// The operator at `index`, operators being in execution order.
    ///
    /// # Errors
    ///
    /// - [`Detail::IndexOutOfRange`] if `index >= self.operator_count()`;
    /// - [`Detail::UnsupportedOp`] (code [`ErrorCode::UnsupportedOperation`]) if the options of
    ///   a decoded operator hold an enum value nanoglia does not know (padding, activation,
    ///   weights format) or a non-standard quantization contract (`FullyConnectedOptions.quant_spec`).
    pub fn operator(&self, index: usize) -> Result<Operator<'m>> {
        let op = get(&self.operators, index)?;
        let opcode_index: u32 = op.scalar(operator::OPCODE_INDEX, 0)?;
        let code_table = self
            .model
            .opcodes
            .get(opcode_index as usize)
            .map_err(|_| Error::malformed(op.pos()))?;
        let code = op_code(&code_table)?;
        let options = match code {
            OpCode::Builtin(builtin) => {
                options::decode(builtin, &op, u32::try_from(index).unwrap_or(u32::MAX))?
            }
            OpCode::Custom(_) => OpOptions::Custom(match large_custom_options(&op)? {
                Some(options) => options,
                None => op.bytes(operator::CUSTOM_OPTIONS)?,
            }),
        };
        Ok(Operator {
            code,
            version: code_table.scalar(operator_code::VERSION, 1)?,
            options,
            inputs: op.slice(operator::INPUTS)?,
            outputs: op.slice(operator::OUTPUTS)?,
        })
    }
}

/// An operator of a [`SubGraph`].
#[derive(Clone, Copy, Debug)]
pub struct Operator<'m> {
    /// The operator code: `max(deprecated_builtin_code, builtin_code)`, or the custom code.
    pub code: OpCode<'m>,
    /// The operator version.
    pub version: i32,
    /// The decoded options; schema defaults if the options table is absent.
    pub options: OpOptions<'m>,
    /// Input tensor indices; `-1` marks an omitted optional input.
    pub inputs: LeSlice<'m, i32>,
    /// Output tensor indices.
    pub outputs: LeSlice<'m, i32>,
}

/// The table at `index` of `tables`, or [`Detail::IndexOutOfRange`].
fn get<'m>(tables: &Tables<'m>, index: usize) -> Result<Table<'m>> {
    if index >= tables.len() {
        return Err(Error::invalid_argument(Detail::IndexOutOfRange {
            index: u32::try_from(index).unwrap_or(u32::MAX),
        }));
    }
    tables.get(index)
}

/// The `size` bytes at `offset` from the start of the file, if in bounds.
pub(crate) fn external(bytes: &[u8], offset: u64, size: u64) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(usize::try_from(size).ok()?)?;
    bytes.get(start..end)
}

/// Decodes an `OperatorCode` table.
pub(crate) fn op_code<'m>(code: &Table<'m>) -> Result<OpCode<'m>> {
    let deprecated: i8 = code.scalar(operator_code::DEPRECATED_BUILTIN_CODE, 0)?;
    let builtin: i32 = code.scalar(operator_code::BUILTIN_CODE, 0)?;
    let builtin = BuiltinOp(i32::from(deprecated).max(builtin));
    if builtin == BuiltinOp::CUSTOM {
        let name = code
            .str(operator_code::CUSTOM_CODE)?
            .ok_or(Error::malformed(code.pos()))?;
        Ok(OpCode::Custom(name))
    } else {
        Ok(OpCode::Builtin(builtin))
    }
}

/// The custom options an operator stores after the flatbuffer (`large_custom_options_*`), if any.
pub(crate) fn large_custom_options<'m>(op: &Table<'m>) -> Result<Option<&'m [u8]>> {
    let offset: u64 = op.scalar(operator::LARGE_CUSTOM_OPTIONS_OFFSET, 0)?;
    if offset <= 1 {
        return Ok(None);
    }
    let size: u64 = op.scalar(operator::LARGE_CUSTOM_OPTIONS_SIZE, 0)?;
    external(op.buf(), offset, size)
        .map(Some)
        .ok_or(Error::malformed(op.pos()))
}
