/*
 * Included before every ESP-NN source file of the optimized builds: ESP-NN prints a message on
 * some error paths (a scratch buffer that was not set), which nanoglia never takes. The calls
 * become no-ops so that nothing references the C library's `printf`.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/* The assembly sources are preprocessed with this header too. */
#ifndef __ASSEMBLER__
#include <stdio.h>

#undef printf
#define printf(...) ((void)0)
#endif
