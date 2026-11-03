"""Generate the test models and golden outputs of `testdata/`.

Golden outputs come from the LiteRT interpreter with the reference kernels
(`OpResolverType.BUILTIN_REF`), which compute the same results as TFLite Micro.

Every case is a directory with:

- `model.tflite` (per-op cases only; MLPerf Tiny cases point at the shared model),
- `input_<i>.bin` / `output_<i>.bin`: raw little-endian bytes of subgraph input
  / output tensor `i`,
- `case.json`: the model path (relative to the repository root), the input and
  output tensors (name, dtype, shape, scale, zero point) and the tolerance.

Generation is deterministic: each case seeds its own random generator from its
name, so re-running this script rewrites identical bytes.

Run with `mise run oracle`, or `uv run python -I generate.py` from this directory.
"""

from __future__ import annotations

import json
import shutil
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tflite_builder import ModelBuilder, schema  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OPS_DIR = REPO / "testdata" / "ops"
MLPERF_DIR = REPO / "testdata" / "mlperf-tiny"
GOLDEN_DIR = MLPERF_DIR / "golden"

MLPERF_MODELS = [
    "ad01_int8",
    "kws_ref_model",
    "pretrainedResnet_quant",
    "vww_96_int8",
]
MLPERF_SAMPLES = 3

# Max allowed difference with the golden float outputs, relative to
# max(1, |expected|). Quantized outputs must match exactly (tolerance 0).
FLOAT_TOLERANCE = 1e-5

Op = schema.BuiltinOperator
Act = schema.ActivationFunctionType
Pad = schema.Padding

F32, I8, I16 = "float32", "int8", "int16"

# Operator versions, per activation type, as the TFLite converter sets them.
VERSIONS = {
    Op.CONV_2D: {F32: 1, I8: 3, I16: 4},
    Op.DEPTHWISE_CONV_2D: {F32: 1, I8: 3, I16: 5},
    Op.FULLY_CONNECTED: {F32: 1, I8: 4, I16: 7},
    Op.AVERAGE_POOL_2D: {F32: 1, I8: 2, I16: 3},
    Op.MAX_POOL_2D: {F32: 1, I8: 2, I16: 3},
    Op.SOFTMAX: {F32: 1, I8: 2, I16: 3},
    Op.ADD: {F32: 1, I8: 2, I16: 3},
    Op.SUB: {F32: 1, I8: 2, I16: 3},
    Op.MUL: {F32: 1, I8: 2, I16: 4},
    Op.RESHAPE: {F32: 1, I8: 1, I16: 1},
    Op.PAD: {F32: 1, I8: 2, I16: 4},
    Op.MEAN: {F32: 1, I8: 2, I16: 3},
    Op.LOGISTIC: {F32: 1, I8: 2, I16: 3},
    Op.TANH: {F32: 1, I8: 2, I16: 3},
    Op.RELU: {F32: 1, I8: 2, I16: 3},
    Op.RELU6: {F32: 1, I8: 2, I16: 3},
    Op.CONCATENATION: {F32: 1, I8: 2, I16: 3},
    Op.MAXIMUM: {F32: 1, I8: 2, I16: 4},
    Op.MINIMUM: {F32: 1, I8: 2, I16: 4},
    Op.QUANTIZE: {F32: 1, I8: 2, I16: 3},
    Op.DEQUANTIZE: {F32: 1, I8: 2, I16: 4},
}


# --------------------------------------------------------------------------
# Quantization helpers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """Type and quantization of an activation tensor (scale None: float)."""

    dtype: str
    scale: float | None = None
    zero_point: int | None = None

    def quantize(self, x: np.ndarray) -> np.ndarray:
        if self.scale is None:
            return x.astype(np.float32)
        info = np.iinfo(self.dtype)
        q = np.round(x / self.scale) + self.zero_point
        return np.clip(q, info.min, info.max).astype(self.dtype)


FLOAT = Spec(F32)


def act_spec(mode: str, lo: float, hi: float) -> Spec:
    """Quantization of an activation spanning [lo, hi].

    int8: asymmetric (the range always contains 0, so the zero point is
    usually nonzero). int16: symmetric, zero point 0 (16x8 quantization).
    """
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    if mode == F32:
        return FLOAT
    if mode == I8:
        scale = np.float32((hi - lo) / 255.0)
        zp = int(np.clip(np.round(-128 - lo / float(scale)), -128, 127))
        return Spec(I8, float(scale), zp)
    if mode == I16:
        scale = np.float32(max(-lo, hi) / 32767.0)
        return Spec(I16, float(scale), 0)
    raise ValueError(mode)


def range_spec(mode: str, *arrays: np.ndarray) -> Spec:
    """Quantization covering the range of all `arrays`."""
    lo = min(float(a.min()) for a in arrays)
    hi = max(float(a.max()) for a in arrays)
    return act_spec(mode, lo, hi)


def quantize_weights(w: np.ndarray, axis: int | None):
    """Symmetric int8 weights, per channel along `axis` (None: per tensor)."""
    if axis is None:
        absmax = np.array([np.abs(w).max()])
        scale = np.maximum(absmax / 127.0, 1e-9).astype(np.float32)
        q = np.round(w / scale[0])
    else:
        other = tuple(i for i in range(w.ndim) if i != axis)
        absmax = np.abs(w).max(axis=other)
        scale = np.maximum(absmax / 127.0, 1e-9).astype(np.float32)
        shape = [1] * w.ndim
        shape[axis] = -1
        q = np.round(w / scale.reshape(shape))
    return np.clip(q, -127, 127).astype(np.int8), scale


def add_weights(b: ModelBuilder, name, w, x: Spec, axis, per_channel=True):
    """Add a weight tensor: float, or int8 symmetric quantized (zero point 0)."""
    if x.scale is None:
        return b.add_tensor(name, np.float32, w.shape, data=w.astype(np.float32)), None
    q, scale = quantize_weights(w, axis if per_channel else None)
    t = b.add_tensor(
        name,
        np.int8,
        w.shape,
        data=q,
        scale=scale,
        zero_point=[0] * scale.size,
        quantized_dimension=axis if per_channel else 0,
    )
    return t, scale


def add_bias(b: ModelBuilder, name, bias, x: Spec, w_scale):
    """Add a bias: float, int32 (int8 activations) or int64 (int16 activations),
    with scale input_scale * weight_scale and zero point 0."""
    if bias is None:
        return -1
    if x.scale is None:
        return b.add_tensor(name, np.float32, bias.shape, data=bias.astype(np.float32))
    scale = (np.float32(x.scale) * w_scale).astype(np.float32)
    if scale.size == 1:
        scale = np.repeat(scale, bias.size)
        q_scale = scale[:1]
    else:
        q_scale = scale
    dtype = np.int32 if x.dtype == I8 else np.int64
    q = np.round(bias / scale.astype(np.float64)).astype(dtype)
    return b.add_tensor(
        name, dtype, bias.shape, data=q, scale=q_scale, zero_point=[0] * q_scale.size
    )


def add_act(b: ModelBuilder, name, spec: Spec, shape):
    return b.add_tensor(name, spec.dtype, shape, scale=spec.scale, zero_point=spec.zero_point)


def version(op: int, spec: Spec) -> int:
    return VERSIONS[op][spec.dtype]


def rng_for(name: str) -> np.random.Generator:
    """The random generator of a case, seeded from its name without the type
    suffix: the float32, int8 and int16 variants of a case share their data."""
    base, _, suffix = name.rpartition("_")
    if suffix in ("f32", "int8", "int16") and base:
        name = base
    return np.random.default_rng(zlib.crc32(name.encode()))


def conv_out(size, k, stride, dilation, padding):
    eff = (k - 1) * dilation + 1
    if padding == Pad.SAME:
        return (size + stride - 1) // stride
    return (size - eff + stride) // stride


# --------------------------------------------------------------------------
# Interpreter
# --------------------------------------------------------------------------


def make_interpreter(model_path=None, model_content=None) -> Interpreter:
    it = Interpreter(
        model_path=None if model_path is None else str(model_path),
        model_content=model_content,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    it.allocate_tensors()
    return it


def invoke(it: Interpreter, inputs):
    for d, x in zip(it.get_input_details(), inputs, strict=True):
        if x.dtype != d["dtype"] or list(x.shape) != list(d["shape"]):
            raise ValueError(f"input {d['name']}: {x.dtype}{x.shape} != {d['dtype']}{d['shape']}")
        it.set_tensor(d["index"], x)
    it.invoke()
    return [it.get_tensor(d["index"]).copy() for d in it.get_output_details()]


def run(model: bytes, inputs) -> list[np.ndarray]:
    """Run a model held in memory (calibration of the quantized cases)."""
    return invoke(make_interpreter(model_content=model), inputs)


def tensor_json(d) -> dict:
    scale, zp = d["quantization"]
    return {
        "name": d["name"],
        "dtype": np.dtype(d["dtype"]).name,
        "shape": [int(s) for s in d["shape"]],
        "scale": float(np.float32(scale)),
        "zero_point": int(zp),
    }


def write_case(case_dir: Path, model_path: Path, inputs, tolerance: float) -> None:
    """Run `model_path` on `inputs` and write the case files into `case_dir`."""
    case_dir.mkdir(parents=True, exist_ok=True)
    it = make_interpreter(model_path=model_path)
    outputs = invoke(it, inputs)
    for i, x in enumerate(inputs):
        (case_dir / f"input_{i}.bin").write_bytes(x.astype(x.dtype.newbyteorder("<")).tobytes())
    for i, y in enumerate(outputs):
        (case_dir / f"output_{i}.bin").write_bytes(y.astype(y.dtype.newbyteorder("<")).tobytes())
    case = {
        "model": model_path.relative_to(REPO).as_posix(),
        "inputs": [tensor_json(d) for d in it.get_input_details()],
        "outputs": [tensor_json(d) for d in it.get_output_details()],
        "tolerance": tolerance,
    }
    (case_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n")


CASES: list[str] = []


def save_op_case(name: str, model: bytes, inputs, tolerance=None) -> None:
    if name in CASES:
        raise ValueError(f"duplicate case {name}")
    CASES.append(name)
    case_dir = OPS_DIR / name
    case_dir.mkdir(parents=True)
    model_path = case_dir / "model.tflite"
    model_path.write_bytes(model)
    if tolerance is None:
        outputs = make_interpreter(model_content=model).get_output_details()
        tolerance = FLOAT_TOLERANCE if any(d["dtype"] == np.float32 for d in outputs) else 0
    write_case(case_dir, model_path, inputs, tolerance)


def single_op(op, in_specs, in_shapes, out_spec, out_shape, options=None, consts=(), ver_spec=None):
    """A model with one operator: activation inputs, then constant inputs.

    `consts` holds (name, array) pairs appended to the operator inputs.
    """
    b = ModelBuilder()
    ins = [add_act(b, f"input_{i}", s, shp) for i, (s, shp) in enumerate(zip(in_specs, in_shapes))]
    cs = [b.add_tensor(n, a.dtype, a.shape, data=a) for n, a in consts]
    y = add_act(b, "output", out_spec, out_shape)
    b.add_operator(op, ins + cs, [y], options, version=version(op, ver_spec or in_specs[0]))
    return b.build(ins, [y])


def calibrated_case(name, mode, xs, build, *, out=None, same_in=False, tolerance=None):
    """Save a case whose quantization is calibrated on a float run.

    `build(in_specs, out_spec)` returns the model bytes; `xs` are the float
    inputs. In quantized modes, the input ranges come from `xs` (one shared
    range if `same_in`) and the output range from the float model's output,
    unless `out` fixes the output: a Spec, or "same" for the input 0 params.
    """
    float_model = build([FLOAT] * len(xs), FLOAT)
    if mode == F32:
        save_op_case(name, float_model, xs, tolerance)
        return
    y = run(float_model, xs)[0]
    if same_in:
        in_specs = [range_spec(mode, *xs)] * len(xs)
    else:
        in_specs = [range_spec(mode, x) for x in xs]
    if out is None:
        out_spec = range_spec(mode, y)
    elif out == "same":
        out_spec = in_specs[0]
    else:
        out_spec = out
    model = build(in_specs, out_spec)
    save_op_case(name, model, [s.quantize(x) for s, x in zip(in_specs, xs)], tolerance)


def uniform(rng, shape, lo=-1.0, hi=3.0):
    """Float activations; the default range is asymmetric (nonzero zero points)."""
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


def weights(rng, shape, fan_in, channel_axis):
    """Random weights whose channels have different magnitudes."""
    w = rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=shape)
    gain_shape = [1] * len(shape)
    gain_shape[channel_axis] = shape[channel_axis]
    w *= rng.uniform(0.25, 2.0, size=gain_shape)
    return w.astype(np.float32)


# --------------------------------------------------------------------------
# Per-op cases
# --------------------------------------------------------------------------


def conv2d(name, mode, in_shape, out_ch, k, stride=1, padding=Pad.SAME, dilation=1,
           act=Act.NONE):
    rng = rng_for(name)
    n, h, w_, c = in_shape
    x = uniform(rng, in_shape)
    w = weights(rng, (out_ch, k, k, c), k * k * c, 0)
    bias = rng.normal(0.0, 0.2, size=out_ch).astype(np.float32)
    out_shape = [n, conv_out(h, k, stride, dilation, padding),
                 conv_out(w_, k, stride, dilation, padding), out_ch]
    opts = schema.Conv2DOptionsT(padding=padding, strideW=stride, strideH=stride,
                                 fusedActivationFunction=act, dilationWFactor=dilation,
                                 dilationHFactor=dilation)

    def build(ins, out):
        b = ModelBuilder()
        xi = add_act(b, "input", ins[0], in_shape)
        wi, ws = add_weights(b, "filter", w, ins[0], axis=0)
        bi = add_bias(b, "bias", bias, ins[0], ws)
        y = add_act(b, "output", out, out_shape)
        b.add_operator(Op.CONV_2D, [xi, wi, bi], [y], opts, version=version(Op.CONV_2D, ins[0]))
        return b.build([xi], [y])

    calibrated_case(name, mode, [x], build)


def depthwise_conv2d(name, mode, in_shape, multiplier, k=3, stride=1, padding=Pad.SAME,
                     dilation=1, act=Act.NONE):
    rng = rng_for(name)
    n, h, w_, c = in_shape
    out_ch = c * multiplier
    x = uniform(rng, in_shape)
    w = weights(rng, (1, k, k, out_ch), k * k, 3)
    bias = rng.normal(0.0, 0.2, size=out_ch).astype(np.float32)
    out_shape = [n, conv_out(h, k, stride, dilation, padding),
                 conv_out(w_, k, stride, dilation, padding), out_ch]
    opts = schema.DepthwiseConv2DOptionsT(padding=padding, strideW=stride, strideH=stride,
                                          depthMultiplier=multiplier,
                                          fusedActivationFunction=act,
                                          dilationWFactor=dilation, dilationHFactor=dilation)

    def build(ins, out):
        b = ModelBuilder()
        xi = add_act(b, "input", ins[0], in_shape)
        wi, ws = add_weights(b, "filter", w, ins[0], axis=3)
        bi = add_bias(b, "bias", bias, ins[0], ws)
        y = add_act(b, "output", out, out_shape)
        b.add_operator(Op.DEPTHWISE_CONV_2D, [xi, wi, bi], [y], opts,
                       version=version(Op.DEPTHWISE_CONV_2D, ins[0]))
        return b.build([xi], [y])

    calibrated_case(name, mode, [x], build)


def fully_connected(name, mode, in_shape, units, *, in_features=None, per_channel=True,
                    use_bias=True, keep_num_dims=False, act=Act.NONE):
    """`in_features` defaults to the last input dimension; a multiple of it
    makes the operator flatten the input into [-1, in_features]."""
    rng = rng_for(name)
    x = uniform(rng, in_shape)
    in_features = in_features or in_shape[-1]
    w = weights(rng, (units, in_features), in_features, 0)
    bias = rng.normal(0.0, 0.2, size=units).astype(np.float32) if use_bias else None
    if keep_num_dims:
        out_shape = list(in_shape[:-1]) + [units]
    else:
        out_shape = [int(np.prod(in_shape)) // in_features, units]
    opts = schema.FullyConnectedOptionsT(fusedActivationFunction=act, keepNumDims=keep_num_dims)

    def build(ins, out):
        b = ModelBuilder()
        xi = add_act(b, "input", ins[0], in_shape)
        wi, ws = add_weights(b, "weights", w, ins[0], axis=0, per_channel=per_channel)
        bi = add_bias(b, "bias", bias, ins[0], ws)
        y = add_act(b, "output", out, out_shape)
        b.add_operator(Op.FULLY_CONNECTED, [xi, wi, bi], [y], opts,
                       version=version(Op.FULLY_CONNECTED, ins[0]))
        return b.build([xi], [y])

    calibrated_case(name, mode, [x], build)


def pool2d(name, mode, op, in_shape, k, stride, padding, act=Act.NONE):
    rng = rng_for(name)
    n, h, w_, c = in_shape
    x = uniform(rng, in_shape)
    out_shape = [n, conv_out(h, k, stride, 1, padding), conv_out(w_, k, stride, 1, padding), c]
    opts = schema.Pool2DOptionsT(padding=padding, strideW=stride, strideH=stride,
                                 filterWidth=k, filterHeight=k, fusedActivationFunction=act)

    def build(ins, out):
        return single_op(op, ins, [in_shape], out, out_shape, opts)

    # Quantized pooling keeps the input quantization.
    calibrated_case(name, mode, [x], build, out="same")


def softmax(name, mode, shape, beta=1.0):
    rng = rng_for(name)
    x = uniform(rng, shape, -4.0, 4.0)
    out = {I8: Spec(I8, 1.0 / 256.0, -128), I16: Spec(I16, 1.0 / 32768.0, 0)}.get(mode)
    opts = schema.SoftmaxOptionsT(beta=beta)

    def build(ins, o):
        return single_op(Op.SOFTMAX, ins, [shape], o, shape, opts)

    calibrated_case(name, mode, [x], build, out=out)


def binary(name, mode, op, shape_a, shape_b, act=Act.NONE):
    rng = rng_for(name)
    a = uniform(rng, shape_a, -1.0, 3.0)
    bb = uniform(rng, shape_b, -2.0, 1.5)
    out_shape = list(np.broadcast_shapes(tuple(shape_a), tuple(shape_b)))
    if op == Op.ADD:
        opts = schema.AddOptionsT(fusedActivationFunction=act, potScaleInt16=False)
    elif op == Op.SUB:
        opts = schema.SubOptionsT(fusedActivationFunction=act, potScaleInt16=False)
    elif op == Op.MUL:
        opts = schema.MulOptionsT(fusedActivationFunction=act)
    else:
        opts = None

    def build(ins, out):
        return single_op(op, ins, [shape_a, shape_b], out, out_shape, opts)

    # MAXIMUM / MINIMUM compare raw quantized values: one quantization for all.
    shared = op in (Op.MAXIMUM, Op.MINIMUM)
    calibrated_case(name, mode, [a, bb], build, same_in=shared, out="same" if shared else None)


def unary(name, mode, op, shape, lo=-4.0, hi=5.0, out=None):
    rng = rng_for(name)
    x = uniform(rng, shape, lo, hi)

    def build(ins, o):
        return single_op(op, ins, [shape], o, shape)

    calibrated_case(name, mode, [x], build, out=out)


def reshape(name, mode, in_shape, new_shape):
    rng = rng_for(name)
    x = uniform(rng, in_shape)
    opts = schema.ReshapeOptionsT(newShape=list(new_shape))
    shape_t = np.array(new_shape, dtype=np.int32)

    def build(ins, out):
        return single_op(Op.RESHAPE, ins, [in_shape], out, new_shape, opts,
                         consts=[("shape", shape_t)])

    calibrated_case(name, mode, [x], build, out="same")


def pad(name, mode, in_shape, paddings):
    rng = rng_for(name)
    x = uniform(rng, in_shape)
    p = np.array(paddings, dtype=np.int32)
    out_shape = [d + int(lo) + int(hi) for d, (lo, hi) in zip(in_shape, paddings)]

    def build(ins, out):
        return single_op(Op.PAD, ins, [in_shape], out, out_shape, schema.PadOptionsT(),
                         consts=[("paddings", p)])

    # Quantized PAD fills with the zero point: same quantization in and out.
    calibrated_case(name, mode, [x], build, out="same")


def mean(name, mode, in_shape, keep_dims, same_quant):
    rng = rng_for(name)
    x = uniform(rng, in_shape)
    axis = np.array([1, 2], dtype=np.int32)
    if keep_dims:
        out_shape = [in_shape[0], 1, 1, in_shape[3]]
    else:
        out_shape = [in_shape[0], in_shape[3]]
    opts = schema.ReducerOptionsT(keepDims=keep_dims)

    def build(ins, out):
        return single_op(Op.MEAN, ins, [in_shape], out, out_shape, opts,
                         consts=[("axis", axis)])

    calibrated_case(name, mode, [x], build, out="same" if same_quant else None)


def concatenation(name, mode, shapes, axis, same_quant):
    rng = rng_for(name)
    xs = [uniform(rng, s, -1.0 - i, 2.0 + 2 * i) for i, s in enumerate(shapes)]
    out_shape = list(shapes[0])
    out_shape[axis] = sum(s[axis] for s in shapes)
    opts = schema.ConcatenationOptionsT(axis=axis)

    def build(ins, out):
        return single_op(Op.CONCATENATION, ins, shapes, out, out_shape, opts)

    calibrated_case(name, mode, xs, build, same_in=same_quant,
                    out="same" if same_quant else None)


def quantize_cases():
    # float32 -> int8
    name = "quantize_f32_int8"
    rng = rng_for(name)
    shape = [1, 4, 4, 8]
    x = uniform(rng, shape)
    q = range_spec(I8, x)
    model = single_op(Op.QUANTIZE, [FLOAT], [shape], q, shape, ver_spec=q)
    save_op_case(name, model, [x])

    # int8 -> int8 requantization
    name = "quantize_int8_int8"
    rng = rng_for(name)
    x = uniform(rng, shape)
    qi = range_spec(I8, x)
    qo = act_spec(I8, -2.5, 4.0)
    model = single_op(Op.QUANTIZE, [qi], [shape], qo, shape)
    save_op_case(name, model, [qi.quantize(x)])

    # int8 -> float32: bit exact (one correctly rounded multiplication).
    name = "dequantize_int8_f32"
    rng = rng_for(name)
    x = uniform(rng, shape)
    qi = range_spec(I8, x)
    model = single_op(Op.DEQUANTIZE, [qi], [shape], FLOAT, shape)
    save_op_case(name, model, [qi.quantize(x)], tolerance=0)


def op_cases():
    for mode in (I8, F32):
        s = "_" + {I8: "int8", F32: "f32"}[mode]
        # CONV_2D
        conv2d("conv2d_1x1" + s, mode, [1, 6, 6, 8], 16, 1, padding=Pad.VALID)
        conv2d("conv2d_3x3_same" + s, mode, [1, 8, 8, 4], 8, 3)
        conv2d("conv2d_3x3_stride2_valid" + s, mode, [1, 9, 9, 4], 8, 3, stride=2,
               padding=Pad.VALID)
        conv2d("conv2d_3x3_dilation2" + s, mode, [1, 9, 9, 3], 4, 3, dilation=2)
        conv2d("conv2d_relu" + s, mode, [1, 7, 7, 4], 8, 3, act=Act.RELU)
        conv2d("conv2d_relu6" + s, mode, [1, 7, 7, 4], 8, 3, stride=2, act=Act.RELU6)
        conv2d("conv2d_batch2" + s, mode, [2, 5, 5, 3], 6, 3)
        # DEPTHWISE_CONV_2D
        depthwise_conv2d("depthwise_conv2d_m1_same" + s, mode, [1, 8, 8, 8], 1)
        depthwise_conv2d("depthwise_conv2d_m2_valid" + s, mode, [1, 7, 7, 4], 2,
                         padding=Pad.VALID)
        depthwise_conv2d("depthwise_conv2d_stride2_same_relu6" + s, mode, [1, 9, 9, 8], 1,
                         stride=2, act=Act.RELU6)
        depthwise_conv2d("depthwise_conv2d_stride2_valid" + s, mode, [1, 8, 8, 8], 1,
                         stride=2, padding=Pad.VALID)
        depthwise_conv2d("depthwise_conv2d_m2_dilation2" + s, mode, [1, 8, 8, 3], 2,
                         dilation=2)
        # FULLY_CONNECTED
        fully_connected("fully_connected_per_tensor" + s, mode, [2, 16], 8, per_channel=False)
        fully_connected("fully_connected_per_channel" + s, mode, [2, 16], 8)
        fully_connected("fully_connected_no_bias" + s, mode, [2, 16], 8, use_bias=False)
        fully_connected("fully_connected_keep_num_dims" + s, mode, [1, 3, 16], 8,
                        keep_num_dims=True)
        fully_connected("fully_connected_4d_input" + s, mode, [1, 2, 2, 8], 10, in_features=32)
        fully_connected("fully_connected_relu" + s, mode, [3, 12], 6, act=Act.RELU)
        # Pooling
        pool2d("average_pool2d_2x2_stride2_valid" + s, mode, Op.AVERAGE_POOL_2D,
               [1, 8, 8, 4], 2, 2, Pad.VALID)
        pool2d("average_pool2d_3x3_stride1_same" + s, mode, Op.AVERAGE_POOL_2D,
               [1, 6, 6, 4], 3, 1, Pad.SAME)
        pool2d("average_pool2d_3x3_stride2_same" + s, mode, Op.AVERAGE_POOL_2D,
               [2, 7, 7, 3], 3, 2, Pad.SAME)
        pool2d("max_pool2d_2x2_stride2_valid" + s, mode, Op.MAX_POOL_2D,
               [1, 8, 8, 4], 2, 2, Pad.VALID)
        pool2d("max_pool2d_3x3_stride1_same" + s, mode, Op.MAX_POOL_2D,
               [1, 6, 6, 4], 3, 1, Pad.SAME)
        pool2d("max_pool2d_3x3_stride2_same" + s, mode, Op.MAX_POOL_2D,
               [2, 7, 7, 3], 3, 2, Pad.SAME)
        # SOFTMAX
        softmax("softmax_2d" + s, mode, [2, 10])
        softmax("softmax_4d" + s, mode, [1, 2, 3, 10])
        softmax("softmax_2d_beta2" + s, mode, [3, 7], beta=2.0)
        # ADD / SUB / MUL
        for op, opname in ((Op.ADD, "add"), (Op.SUB, "sub"), (Op.MUL, "mul")):
            binary(f"{opname}" + s, mode, op, [1, 4, 4, 8], [1, 4, 4, 8])
            binary(f"{opname}_relu" + s, mode, op, [1, 4, 4, 8], [1, 4, 4, 8], act=Act.RELU)
            binary(f"{opname}_broadcast" + s, mode, op, [1, 4, 4, 8], [1, 1, 1, 8])
            binary(f"{opname}_broadcast_relu6" + s, mode, op, [1, 1, 4, 8], [1, 3, 1, 8],
                   act=Act.RELU6)
        # MAXIMUM / MINIMUM
        binary("maximum" + s, mode, Op.MAXIMUM, [1, 4, 4, 8], [1, 4, 4, 8])
        binary("maximum_broadcast" + s, mode, Op.MAXIMUM, [1, 4, 4, 8], [1, 1, 1, 8])
        binary("minimum" + s, mode, Op.MINIMUM, [1, 4, 4, 8], [1, 4, 4, 8])
        binary("minimum_broadcast" + s, mode, Op.MINIMUM, [1, 4, 4, 8], [1, 1, 1, 8])
        # RESHAPE / PAD / MEAN
        reshape("reshape" + s, mode, [1, 2, 2, 8], [1, 32])
        pad("pad" + s, mode, [1, 4, 4, 3], [[0, 0], [1, 2], [2, 1], [0, 0]])
        mean("mean_keep_dims" + s, mode, [1, 4, 4, 8], True, same_quant=True)
        mean("mean" + s, mode, [1, 4, 4, 8], False, same_quant=False)
        # Activations
        logistic_out = Spec(I8, 1.0 / 256.0, -128) if mode == I8 else None
        tanh_out = Spec(I8, 1.0 / 128.0, 0) if mode == I8 else None
        unary("logistic" + s, mode, Op.LOGISTIC, [1, 4, 4, 8], -6.0, 5.0, out=logistic_out)
        unary("tanh" + s, mode, Op.TANH, [1, 4, 4, 8], -3.0, 2.5, out=tanh_out)
        unary("relu" + s, mode, Op.RELU, [1, 4, 4, 8])
        unary("relu6" + s, mode, Op.RELU6, [1, 4, 4, 8], -3.0, 8.0)
        # CONCATENATION
        concatenation("concatenation_same_quant" + s, mode, [[1, 2, 2, 3], [1, 2, 2, 5]], 3,
                      same_quant=True)
        concatenation("concatenation_axis1" + s, mode,
                      [[1, 2, 3, 4], [1, 1, 3, 4], [1, 3, 3, 4]], 1, same_quant=True)
    # Not generated: int8 CONCATENATION with different input and output
    # quantization. The reference kernel rejects it (concatenation.cc requires
    # equal scales and zero points for int8 and int16), as does TFLite Micro.

    quantize_cases()

    # 16x8 quantization: int16 activations (zero point 0), int8 weights, int64 bias.
    conv2d("conv2d_3x3_same_int16", I16, [1, 8, 8, 4], 8, 3)
    fully_connected("fully_connected_per_channel_int16", I16, [2, 16], 8)
    softmax("softmax_2d_int16", I16, [2, 10])
    binary("add_int16", I16, Op.ADD, [1, 4, 4, 8], [1, 4, 4, 8])
    binary("add_broadcast_int16", I16, Op.ADD, [1, 4, 4, 8], [1, 1, 1, 8])


# --------------------------------------------------------------------------
# MLPerf Tiny
# --------------------------------------------------------------------------


def mlperf_cases():
    for model in MLPERF_MODELS:
        model_path = MLPERF_DIR / f"{model}.tflite"
        rng = rng_for(model)
        details = make_interpreter(model_path=model_path).get_input_details()
        for sample in range(MLPERF_SAMPLES):
            inputs = []
            for d in details:
                info = np.iinfo(d["dtype"])
                inputs.append(
                    rng.integers(info.min, info.max, size=d["shape"], dtype=d["dtype"],
                                 endpoint=True)
                )
            write_case(GOLDEN_DIR / model / f"sample_{sample}", model_path, inputs, 0)


def main():
    for d in (OPS_DIR, GOLDEN_DIR):
        if d.exists():
            shutil.rmtree(d)
    op_cases()
    mlperf_cases()
    print(f"{len(CASES)} op cases in {OPS_DIR.relative_to(REPO)}")
    print(f"{len(MLPERF_MODELS)} x {MLPERF_SAMPLES} MLPerf Tiny cases in "
          f"{GOLDEN_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
