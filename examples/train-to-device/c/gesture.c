/*
 * Runs the gesture classifier on the test vectors exported with it and checks every output
 * against the one LiteRT computed: the C counterpart of ../src/main.rs.
 *
 *     gesture VECTORS_DIRECTORY
 *
 * The argument is a directory written by training/train_pytorch.py or training/train_keras.py:
 * sample_<k>/window.bin holds an accelerometer window (float32, in g), input_0.bin the int8 input
 * and output_0.bin the int8 output of LiteRT.
 *
 * The model is compiled into the program, as in a firmware: model.inc holds its bytes as a C
 * initializer list (`xxd -i < model.tflite > model.inc`, or cmake/embed.cmake). Every byte
 * nanoglia uses is in the two static arrays below.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "nanoglia.h"

/* The model bytes, on a 16-byte boundary (NANOGLIA_MODEL_ALIGNMENT): const, so in flash. */
static _Alignas(NANOGLIA_MODEL_ALIGNMENT) const uint8_t model[] = {
#include "model.inc"
};
/* The handle, the graph and the execution context: nanoglia_memory_requirements() gives the size. */
static unsigned char memory[8 * 1024];

/* The classes of the model, in the order of its outputs. */
static const char *const labels[] = {"idle", "shake", "circle"};

enum { WINDOW = 50 * 3, CLASSES = 3 };

/* Reads exactly size bytes of DIRECTORY/sample_<sample>/NAME; false if the file is missing or short. */
static bool read_vector(const char *directory, unsigned sample, const char *name, void *data,
                        size_t size) {
    char path[1024];
    snprintf(path, sizeof path, "%s/sample_%u/%s", directory, sample, name);
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return false;
    }
    bool complete = fread(data, 1, size, file) == size && fgetc(file) == EOF;
    fclose(file);
    return complete;
}

static int fail(const char *what, const nanoglia_error *error) {
    fprintf(stderr, "%s: %s (%s)\n", what, nanoglia_status_description(error->status),
            nanoglia_detail_description(error->detail));
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s VECTORS_DIRECTORY\n", argv[0]);
        return 2;
    }

    /* How much memory the model needs; the buffer serves as scratch memory meanwhile. */
    nanoglia_error error;
    nanoglia_memory needs;
    if (nanoglia_memory_requirements(model, sizeof model, memory, sizeof memory, &needs, &error) !=
        NANOGLIA_STATUS_OK) {
        return fail("memory requirements", &error);
    }
    printf("model %zu B, graph %zu B, context %zu B, buffer %zu of %zu bytes\n", sizeof model,
           needs.graph_bytes, needs.context_bytes, needs.memory_bytes, sizeof memory);

    nanoglia *nn;
    if (nanoglia_load(model, sizeof model, memory, sizeof memory, &nn, &error) != NANOGLIA_STATUS_OK) {
        return fail("load", &error);
    }
    nanoglia_tensor input;
    nanoglia_tensor output;
    nanoglia_input(nn, 0, &input);
    nanoglia_output(nn, 0, &output);
    if (input.element_type != NANOGLIA_TYPE_INT8 || input.size != WINDOW ||
        output.element_type != NANOGLIA_TYPE_INT8 || output.size != CLASSES) {
        fprintf(stderr, "not the gesture model\n");
        nanoglia_release(nn);
        return 1;
    }
    printf("%zu operators, input scale %.9g, zero point %d\n", nanoglia_operator_count(nn),
           (double)input.scale, (int)input.zero_point);

    unsigned sample = 0;
    unsigned mismatches = 0;
    float window[WINDOW]; /* 50 samples of 3 axes, in g */
    while (read_vector(argv[1], sample, "window.bin", window, sizeof window)) {
        /* Quantize the window into the input tensor: q = round(x / scale) + zero_point. */
        int8_t *quantized = input.data;
        for (size_t i = 0; i < WINDOW; i++) {
            long q = lroundf(window[i] / input.scale) + input.zero_point;
            quantized[i] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
        }
        int8_t expected_input[WINDOW];
        bool same = read_vector(argv[1], sample, "input_0.bin", expected_input, WINDOW) &&
                    memcmp(quantized, expected_input, WINDOW) == 0;

        if (nanoglia_compute(nn, &error) != NANOGLIA_STATUS_OK) {
            nanoglia_release(nn);
            return fail("compute", &error);
        }

        /* Dequantize the best score: probability = scale * (q - zero_point). */
        const int8_t *scores = output.data;
        size_t best = 0;
        for (size_t i = 1; i < CLASSES; i++) {
            if (scores[i] > scores[best]) {
                best = i;
            }
        }
        float probability = output.scale * (float)(scores[best] - output.zero_point);

        int8_t expected_output[CLASSES];
        same = same && read_vector(argv[1], sample, "output_0.bin", expected_output, CLASSES) &&
               memcmp(scores, expected_output, CLASSES) == 0;
        printf("sample_%u: [%4d, %4d, %4d] -> %-6s %.3f  %s\n", sample, scores[0], scores[1],
               scores[2], labels[best], (double)probability,
               same ? "same as LiteRT" : "DIFFERS from LiteRT");
        mismatches += same ? 0 : 1;
        sample++;
    }
    nanoglia_release(nn);

    if (sample == 0) {
        fprintf(stderr, "%s: no sample_0/window.bin\n", argv[1]);
        return 1;
    }
    printf("%u of %u samples match LiteRT\n", sample - mismatches, sample);
    return mismatches == 0 ? 0 : 1;
}
