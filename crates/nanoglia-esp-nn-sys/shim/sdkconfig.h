/*
 * Stand-in for the `sdkconfig.h` that ESP-IDF generates from Kconfig. nanoglia builds ESP-NN
 * without ESP-IDF: the few options ESP-NN reads (`CONFIG_NN_OPTIMIZED`,
 * `CONFIG_IDF_TARGET_ESP32S3`...) are passed by the build script on the command line, the others
 * keep their defaults.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
