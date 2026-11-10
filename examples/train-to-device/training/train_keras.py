"""Train the gesture classifier with Keras and export it to a full-integer int8 LiteRT model.

    uv run --project examples/train-to-device/training --group keras \
        python examples/train-to-device/training/train_keras.py target/train-to-device/keras

Trains the network on the synthetic windows of `gesture.py`, converts it with
`tf.lite.TFLiteConverter` (post-training full-integer quantization, calibrated
on training windows, int8 input and output), then checks the `.tflite` model
with LiteRT and writes it with its test vectors (see `gesture.export`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

import gesture


def build() -> tf.keras.Model:
    """Two convolutions over time, a global average, a dense layer: 835 parameters."""
    return tf.keras.Sequential([
        # 50 samples of 3 axes, as one channel: [50, 3, 1].
        tf.keras.Input(shape=(gesture.SAMPLES, gesture.AXES, 1)),
        tf.keras.layers.Conv2D(8, (5, 3), activation="relu"),  # mixes the axes: [46, 1, 8]
        tf.keras.layers.MaxPooling2D((2, 1)),  # [23, 1, 8]
        tf.keras.layers.Conv2D(16, (5, 1), activation="relu"),  # [19, 1, 16]
        tf.keras.layers.GlobalAveragePooling2D(),  # [16]
        tf.keras.layers.Dense(len(gesture.LABELS)),
        tf.keras.layers.Softmax(),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, help="output directory (deleted first)")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(0)
    (train_x, train_y), (test_x, test_y) = gesture.datasets()
    train_x, test_x = train_x[..., None], test_x[..., None]  # one channel, last

    model = build()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_x, train_y, epochs=gesture.EPOCHS, batch_size=gesture.BATCH, verbose=2)
    _, float_accuracy = model.evaluate(test_x, test_y, verbose=0)

    def representative_dataset():
        for window in train_x[: gesture.CALIBRATION_WINDOWS]:
            yield [window[None]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Integer kernels only, and an int8 interface: no float operator may remain.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    gesture.export(args.out_dir.resolve(), converter.convert(), float(float_accuracy))


if __name__ == "__main__":
    main()
