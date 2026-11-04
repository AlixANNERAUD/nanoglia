/*
 * Runs the MLPerf Tiny keyword-spotting model on one input and prints the most likely class.
 *
 *     nanoglia_keyword_spotting kws_ref_model.tflite input_0.bin
 *
 * Every byte nanoglia uses is in the two static buffers below: the model, and the memory of the
 * loaded model. A firmware would keep the model in flash and fill the input from a microphone.
 */

#include <stdio.h>
#include <string.h>

#include "nanoglia.h"

/* The model bytes, on a 16-byte boundary (NANOGLIA_MODEL_ALIGNMENT). */
static _Alignas(NANOGLIA_MODEL_ALIGNMENT) unsigned char model[64 * 1024];
/* The handle, the graph and the execution context: nanoglia_memory_requirements() gives the size. */
static unsigned char memory[32 * 1024];

/* Reads a whole file into buffer, returns its size, or 0 on failure. */
static size_t read_file(const char *path, unsigned char *buffer, size_t capacity) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror(path);
        return 0;
    }
    size_t size = fread(buffer, 1, capacity, file);
    if (fgetc(file) != EOF) {
        fprintf(stderr, "%s: larger than %zu bytes\n", path, capacity);
        size = 0;
    }
    fclose(file);
    return size;
}

static int fail(const char *what, const nanoglia_error *error) {
    fprintf(stderr, "%s: %s (%s", what, nanoglia_status_description(error->status),
            nanoglia_detail_description(error->detail));
    if (error->detail == NANOGLIA_DETAIL_OUT_OF_MEMORY) {
        fprintf(stderr, ", %zu bytes", error->size);
    }
    fprintf(stderr, ")\n");
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.tflite INPUT.bin\n", argv[0]);
        return 2;
    }
    size_t model_length = read_file(argv[1], model, sizeof model);
    if (model_length == 0) {
        return 1;
    }

    /* How much memory the model needs; the buffer serves as scratch memory meanwhile. */
    nanoglia_error error;
    nanoglia_memory needs;
    if (nanoglia_memory_requirements(model, model_length, memory, sizeof memory, &needs, &error) !=
        NANOGLIA_STATUS_OK) {
        return fail("memory requirements", &error);
    }
    printf("graph %zu bytes, context %zu bytes, buffer %zu of %zu bytes\n", needs.graph_bytes,
           needs.context_bytes, needs.memory_bytes, sizeof memory);

    nanoglia *nn;
    if (nanoglia_load(model, model_length, memory, sizeof memory, &nn, &error) != NANOGLIA_STATUS_OK) {
        return fail("load", &error);
    }

    /* Write the input straight into the memory of the model. */
    nanoglia_tensor input;
    nanoglia_input(nn, 0, &input);
    if (read_file(argv[2], input.data, input.size) != input.size) {
        fprintf(stderr, "%s: expected %zu bytes\n", argv[2], input.size);
        nanoglia_release(nn);
        return 1;
    }

    /* One operator per call: a firmware would do other work between two steps. */
    size_t steps = 0;
    bool done = false;
    while (!done) {
        if (nanoglia_step(nn, &done, &error) != NANOGLIA_STATUS_OK) {
            nanoglia_release(nn);
            return fail("step", &error);
        }
        steps++;
    }

    nanoglia_tensor output;
    nanoglia_output(nn, 0, &output);
    const int8_t *scores = output.data;
    size_t best = 0;
    for (size_t i = 1; i < output.size; i++) {
        if (scores[i] > scores[best]) {
            best = i;
        }
    }
    printf("%zu steps: class %zu, probability %.3f\n", steps, best,
           output.scale * (float)(scores[best] - output.zero_point));

    nanoglia_release(nn);
    return 0;
}
