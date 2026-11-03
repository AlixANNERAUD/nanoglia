# MLPerf Tiny reference models

The int8 reference models of the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark
(Apache-2.0, see `LICENSE`), used as end-to-end test cases:

| File | Task | Operators |
|---|---|---|
| `kws_ref_model.tflite` | Keyword spotting (DS-CNN) | CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, RESHAPE, FULLY_CONNECTED, SOFTMAX |
| `vww_96_int8.tflite` | Visual wake words (MobileNetV1 0.25) | same |
| `pretrainedResnet_quant.tflite` | Image classification (ResNet-8) | CONV_2D, ADD, AVERAGE_POOL_2D, RESHAPE, FULLY_CONNECTED, SOFTMAX |
| `ad01_int8.tflite` | Anomaly detection (FC autoencoder) | FULLY_CONNECTED |
