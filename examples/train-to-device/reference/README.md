# Reference gesture model

`model.tflite` is the gesture classifier of the
[train-and-deploy tutorial](../../../documentation/tutorials/train-and-deploy.md), exported by the
Keras track, and `sample_<k>/` its six test vectors, in the golden-case format of
[`testdata/`](../../../testdata/README.md) plus `window.bin` (the float32 window before
quantization). They let the Rust and C programs of this example build and run without training
anything: `mise run example-train-to-device` checks them on every change.

They were written on 2026-11-10 by

```sh
uv run --locked --project examples/train-to-device/training --group keras \
    python examples/train-to-device/training/train_keras.py examples/train-to-device/reference
```

on a GitHub Actions `ubuntu-latest` runner (Ubuntu 26.04, x86_64, AMD EPYC 7763), with the
versions of `training/uv.lock`: TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.4, ai-edge-litert
2.3.0, Python 3.13.18. The model reaches 96.33 % on the 600 test windows, in float32 and in int8.

The training is not reproducible to the bit across processors: two runs on that runner gave the
same file, a run on a runner with AVX-512 gave a model with 46 different bytes out of 5048, the
same accuracy and the same outputs on the six vectors. A freshly exported model is therefore not
expected to equal this file.

Models exported from a framework change with its versions, so this is the only one committed
(12 KiB with its vectors). `mise run train-to-device` trains and exports both tracks again and runs
the programs on the fresh models; to refresh this directory, run the command above and commit the
result, with the outputs quoted in the tutorial.
