# From training to the device: a gesture classifier

The code of the tutorial
[From training to the device](../../documentation/tutorials/train-and-deploy.md): a small
classifier of accelerometer windows (`idle`, `shake`, `circle`), trained on synthetic data with
PyTorch or with Keras, exported to a full-integer int8 `.tflite` model, and run with nanoglia from
Rust and from C on the same test vectors, whose outputs must be the bytes LiteRT computes.

| Path | Contents |
| --- | --- |
| `training/` | A [uv](https://docs.astral.sh/uv/) project: `gesture.py` (data, LiteRT evaluation, test vectors), `train_pytorch.py` (dependency group `pytorch`: PT2E quantization, litert-torch), `train_keras.py` (group `keras`: `tf.lite.TFLiteConverter`) |
| `src/main.rs`, `build.rs` | The Rust program, its own Cargo workspace: the model embedded with `include_bytes!`, all memory in a static `Arena` |
| `c/` | The C program through the C API, with its CMake project; `embed.cmake` turns the model into a C initializer list |
| `reference/` | One committed model with its test vectors (see its README), so that the programs run without training |

```sh
mise run example-train-to-device   # the Rust and C programs on the reference model
mise run train-to-device           # train and export both tracks, run the programs on each
```

The second task needs x86_64 Linux or macOS on Apple silicon: litert-torch's converter has no
other wheels. Each training script takes an output directory and writes `model.tflite` and
`sample_<k>/` there; the programs take that directory, and the model to embed through
`GESTURE_MODEL` (Rust) or `-DGESTURE_DIRECTORY` (CMake):

```sh
uv run --locked --project examples/train-to-device/training --group keras \
    python examples/train-to-device/training/train_keras.py target/train-to-device/keras
GESTURE_MODEL=$PWD/target/train-to-device/keras/model.tflite \
    cargo run --release --manifest-path examples/train-to-device/Cargo.toml -- target/train-to-device/keras
```
