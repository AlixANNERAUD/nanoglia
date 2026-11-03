"""Write `.tflite` flatbuffers directly, with full control of every field.

The models have one subgraph. Constant tensors get their own buffer, whose data
starts on a 16-byte boundary of the file (as `force_align: 16` in the TFLite
schema asks); every other tensor points at buffer 0, the empty buffer.

The schema classes come from the `schema_py_generated` module that the
`ai_edge_litert` package ships.

Example::

    b = ModelBuilder()
    x = b.add_tensor("x", np.int8, [1, 4], scale=0.1, zero_point=-3)
    y = b.add_tensor("y", np.int8, [1, 4], scale=0.1, zero_point=-3)
    b.add_operator(schema.BuiltinOperator.RELU, [x], [y], version=2)
    model_bytes = b.build(inputs=[x], outputs=[y])
"""

from __future__ import annotations

from collections.abc import Sequence

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema

__all__ = ["ModelBuilder", "schema", "tensor_type", "SCHEMA_VERSION", "FILE_IDENTIFIER"]

SCHEMA_VERSION = 3
FILE_IDENTIFIER = b"TFL3"
BUFFER_ALIGNMENT = 16
# Builtin codes above this value are stored in `builtin_code` only; the
# deprecated int8 field then holds PLACEHOLDER_FOR_GREATER_OP_CODES.
_DEPRECATED_CODE_MAX = 127

_TENSOR_TYPES = {
    np.dtype(np.float32): schema.TensorType.FLOAT32,
    np.dtype(np.int8): schema.TensorType.INT8,
    np.dtype(np.uint8): schema.TensorType.UINT8,
    np.dtype(np.int16): schema.TensorType.INT16,
    np.dtype(np.int32): schema.TensorType.INT32,
    np.dtype(np.int64): schema.TensorType.INT64,
    np.dtype(np.bool_): schema.TensorType.BOOL,
}


def tensor_type(dtype) -> int:
    """The `TensorType` of a NumPy dtype."""
    return _TENSOR_TYPES[np.dtype(dtype)]


class _AlignedBufferT(schema.BufferT):
    """A `BufferT` whose data vector starts on a `BUFFER_ALIGNMENT` boundary."""

    def Pack(self, builder):  # noqa: N802 (generated-code naming)
        data = None
        if self.data is not None:
            raw = bytes(self.data)
            builder.StartVector(1, len(raw), BUFFER_ALIGNMENT)
            builder.head = builder.head - len(raw)
            builder.Bytes[builder.head : builder.head + len(raw)] = raw
            data = builder.EndVector()
        schema.BufferStart(builder)
        if data is not None:
            schema.BufferAddData(builder, data)
        return schema.BufferEnd(builder)


def _options_type(options) -> int:
    """The `BuiltinOptions` union tag of an options object (`Conv2DOptionsT`...)."""
    name = type(options).__name__
    if not name.endswith("T"):
        raise TypeError(f"not an object-API options table: {name}")
    return getattr(schema.BuiltinOptions, name[:-1])


class ModelBuilder:
    """Accumulates tensors and operators of one subgraph, then serializes them."""

    def __init__(self, description: str = "nanoglia oracle"):
        self.description = description
        self._tensors: list[schema.TensorT] = []
        self._buffers: list[schema.BufferT] = [_AlignedBufferT()]  # buffer 0: empty
        self._opcodes: list[tuple[int, int]] = []
        self._operators: list[schema.OperatorT] = []

    def add_tensor(
        self,
        name: str,
        dtype,
        shape: Sequence[int],
        *,
        data: np.ndarray | None = None,
        scale: float | Sequence[float] | None = None,
        zero_point: int | Sequence[int] | None = None,
        quantized_dimension: int = 0,
    ) -> int:
        """Add a tensor and return its index.

        `data` makes it a constant tensor (stored little-endian in its own
        buffer). `scale` and `zero_point` are scalars for per-tensor
        quantization, or sequences for per-axis quantization along
        `quantized_dimension`.
        """
        dtype = np.dtype(dtype)
        t = schema.TensorT()
        t.name = name
        t.type = tensor_type(dtype)
        t.shape = [int(d) for d in shape]
        t.hasRank = True
        if data is not None:
            data = np.asarray(data)
            if data.dtype != dtype:
                raise TypeError(f"{name}: data is {data.dtype}, tensor is {dtype}")
            if list(data.shape) != t.shape:
                raise ValueError(f"{name}: data shape {data.shape} != {t.shape}")
            buf = _AlignedBufferT()
            buf.data = data.astype(dtype.newbyteorder("<"), copy=False).tobytes()
            t.buffer = len(self._buffers)
            self._buffers.append(buf)
        else:
            t.buffer = 0
        if scale is not None:
            scales = np.atleast_1d(np.asarray(scale, dtype=np.float32))
            zps = np.atleast_1d(
                np.asarray(0 if zero_point is None else zero_point, dtype=np.int64)
            )
            if zps.size == 1 and scales.size > 1:
                zps = np.repeat(zps, scales.size)
            if zps.size != scales.size:
                raise ValueError(f"{name}: {scales.size} scales but {zps.size} zero points")
            q = schema.QuantizationParametersT()
            q.scale = [float(s) for s in scales]
            q.zeroPoint = [int(z) for z in zps]
            q.quantizedDimension = int(quantized_dimension)
            t.quantization = q
        self._tensors.append(t)
        return len(self._tensors) - 1

    def add_operator(
        self,
        opcode: int,
        inputs: Sequence[int],
        outputs: Sequence[int],
        options=None,
        *,
        version: int = 1,
    ) -> int:
        """Add a builtin operator; `options` is an object-API table (`AddOptionsT`...).

        An input index of -1 marks an omitted optional input.
        """
        key = (int(opcode), int(version))
        if key not in self._opcodes:
            self._opcodes.append(key)
        op = schema.OperatorT()
        op.opcodeIndex = self._opcodes.index(key)
        op.inputs = [int(i) for i in inputs]
        op.outputs = [int(o) for o in outputs]
        if options is not None:
            op.builtinOptionsType = _options_type(options)
            op.builtinOptions = options
        self._operators.append(op)
        return len(self._operators) - 1

    def build(self, inputs: Sequence[int], outputs: Sequence[int]) -> bytes:
        """Serialize the model with the given subgraph inputs and outputs."""
        subgraph = schema.SubGraphT()
        subgraph.tensors = self._tensors
        subgraph.inputs = [int(i) for i in inputs]
        subgraph.outputs = [int(o) for o in outputs]
        subgraph.operators = self._operators
        subgraph.name = "main"

        opcodes = []
        for code, version in self._opcodes:
            oc = schema.OperatorCodeT()
            oc.deprecatedBuiltinCode = min(code, _DEPRECATED_CODE_MAX)
            oc.builtinCode = code
            oc.version = version
            opcodes.append(oc)

        model = schema.ModelT()
        model.version = SCHEMA_VERSION
        model.operatorCodes = opcodes
        model.subgraphs = [subgraph]
        model.description = self.description
        model.buffers = self._buffers

        builder = flatbuffers.Builder(1024)
        builder.Finish(model.Pack(builder), file_identifier=FILE_IDENTIFIER)
        out = bytes(builder.Output())
        _check_alignment(out)
        return out


def _check_alignment(model_bytes: bytes) -> None:
    """Assert that every buffer's data starts on a `BUFFER_ALIGNMENT` boundary."""
    model = schema.Model.GetRootAs(model_bytes, 0)
    for i in range(model.BuffersLength()):
        buf = model.Buffers(i)
        if buf.DataLength() == 0:
            continue
        o = buf._tab.Offset(4)  # field `data`
        start = buf._tab.Vector(o)
        if start % BUFFER_ALIGNMENT:
            raise AssertionError(f"buffer {i} data at offset {start} is misaligned")
