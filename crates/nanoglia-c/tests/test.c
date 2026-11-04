/*
 * Tests of the C API, from C: the golden keyword-spotting samples with nanoglia_compute() and
 * nanoglia_step(), cancellation, the exact buffer size, buffers too small, corrupted models and
 * invalid arguments.
 *
 *     nanoglia_test TESTDATA_DIRECTORY
 */

#undef NDEBUG

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanoglia.h"

#define MODEL_CAPACITY (64 * 1024)
#define MEMORY_CAPACITY (64 * 1024)

static const char *testdata;
static _Alignas(16) unsigned char model[MODEL_CAPACITY];
static size_t model_length;
static _Alignas(16) unsigned char memory[MEMORY_CAPACITY + 16];
static _Alignas(16) unsigned char corrupted[MODEL_CAPACITY];
static nanoglia_memory needs;

static size_t read_file(const char *path, unsigned char *buffer, size_t capacity) {
    char full[1024];
    snprintf(full, sizeof full, "%s/%s", testdata, path);
    FILE *file = fopen(full, "rb");
    if (file == NULL) {
        perror(full);
        exit(1);
    }
    size_t size = fread(buffer, 1, capacity, file);
    assert(fgetc(file) == EOF); /* the whole file fits */
    fclose(file);
    return size;
}

/* Reads file `name` of golden sample `k` of the keyword-spotting model. */
static size_t read_sample(int k, const char *name, unsigned char *buffer, size_t capacity) {
    char path[256];
    snprintf(path, sizeof path, "mlperf-tiny/golden/kws_ref_model/sample_%d/%s", k, name);
    return read_file(path, buffer, capacity);
}

static nanoglia *load_at(size_t offset, size_t length, nanoglia_error *error) {
    nanoglia *nn = NULL;
    nanoglia_status status = nanoglia_load(model, model_length, memory + offset, length, &nn, error);
    assert(status == error->status);
    assert((status == NANOGLIA_STATUS_OK) == (nn != NULL));
    return nn;
}

static nanoglia *load(void) {
    nanoglia_error error;
    nanoglia *nn = load_at(0, needs.memory_bytes, &error);
    assert(nn != NULL);
    return nn;
}

static void write_input(nanoglia *nn, int k) {
    nanoglia_tensor input;
    assert(nanoglia_input(nn, 0, &input) == NANOGLIA_STATUS_OK);
    assert(read_sample(k, "input_0.bin", input.data, input.size) == input.size);
}

static void check_output(const nanoglia *nn, int k) {
    unsigned char expected[64];
    size_t length = read_sample(k, "output_0.bin", expected, sizeof expected);
    nanoglia_tensor output;
    assert(nanoglia_output(nn, 0, &output) == NANOGLIA_STATUS_OK);
    assert(output.size == length);
    assert(memcmp(output.data, expected, length) == 0);
}

static size_t run_steps(nanoglia *nn) {
    size_t steps = 0;
    bool done = false;
    while (!done) {
        nanoglia_error error;
        assert(nanoglia_step(nn, &done, &error) == NANOGLIA_STATUS_OK);
        assert(error.status == NANOGLIA_STATUS_OK);
        steps++;
    }
    return steps;
}

static void computes_memory_requirements(void) {
    nanoglia_error error;
    assert(nanoglia_memory_requirements(model, model_length, memory, MEMORY_CAPACITY, &needs, &error) ==
           NANOGLIA_STATUS_OK);
    assert(error.status == NANOGLIA_STATUS_OK && error.detail == NANOGLIA_DETAIL_NONE);
    assert(needs.graph_bytes > 0 && needs.context_bytes > 0);
    assert(needs.memory_bytes > needs.graph_bytes + needs.context_bytes);
    assert(needs.memory_bytes <= MEMORY_CAPACITY);
    printf("graph %zu, context %zu, memory %zu bytes\n", needs.graph_bytes, needs.context_bytes,
           needs.memory_bytes);

    /* Scratch memory too small for the planner. */
    nanoglia_memory unused;
    assert(nanoglia_memory_requirements(model, model_length, memory, 8, &unused, &error) ==
           NANOGLIA_STATUS_TOO_LARGE);
    assert(error.detail == NANOGLIA_DETAIL_OUT_OF_MEMORY && error.size > 8);
}

static void runs_the_golden_samples(void) {
    nanoglia *nn = load();
    assert(nanoglia_input_count(nn) == 1);
    assert(nanoglia_output_count(nn) == 1);
    size_t operators = nanoglia_operator_count(nn);
    assert(operators > 1);
    for (int k = 0; k < 3; k++) {
        nanoglia_error error;
        write_input(nn, k);
        assert(nanoglia_compute(nn, &error) == NANOGLIA_STATUS_OK);
        assert(error.status == NANOGLIA_STATUS_OK);
        check_output(nn, k);

        write_input(nn, k);
        assert(run_steps(nn) == operators);
        check_output(nn, k);
    }
    assert(nanoglia_release(nn) == NANOGLIA_STATUS_OK);
}

static void cancels(void) {
    nanoglia *nn = load();
    write_input(nn, 2);
    bool done = false;
    for (int i = 0; i < 3; i++) {
        assert(nanoglia_step(nn, &done, NULL) == NANOGLIA_STATUS_OK);
        assert(!done);
    }
    assert(nanoglia_cancel(nn) == NANOGLIA_STATUS_OK);
    /* The operators that ran may have reused the memory of the input: write it again. */
    write_input(nn, 2);
    assert(run_steps(nn) == nanoglia_operator_count(nn));
    check_output(nn, 2);

    /* nanoglia_compute() abandons an inference in progress too. */
    write_input(nn, 1);
    assert(nanoglia_step(nn, &done, NULL) == NANOGLIA_STATUS_OK);
    write_input(nn, 1);
    assert(nanoglia_compute(nn, NULL) == NANOGLIA_STATUS_OK);
    check_output(nn, 1);
    assert(nanoglia_release(nn) == NANOGLIA_STATUS_OK);
}

static void needs_exactly_memory_bytes(void) {
    /* Exactly memory_bytes, at every alignment of the buffer. */
    for (size_t offset = 0; offset < 16; offset++) {
        nanoglia_error error;
        nanoglia *nn = load_at(offset, needs.memory_bytes, &error);
        assert(nn != NULL);
        write_input(nn, 0);
        assert(nanoglia_compute(nn, NULL) == NANOGLIA_STATUS_OK);
        check_output(nn, 0);
        assert(nanoglia_release(nn) == NANOGLIA_STATUS_OK);
    }

    /* One byte less, one byte past a 16-byte boundary: too small. */
    nanoglia_error error;
    assert(load_at(1, needs.memory_bytes - 1, &error) == NULL);
    assert(error.status == NANOGLIA_STATUS_TOO_LARGE);
    assert(error.detail == NANOGLIA_DETAIL_OUT_OF_MEMORY);

    const size_t small[] = {0, 1, 64, needs.graph_bytes, needs.memory_bytes / 2};
    for (size_t i = 0; i < sizeof small / sizeof small[0]; i++) {
        assert(load_at(0, small[i], &error) == NULL);
        assert(error.status == NANOGLIA_STATUS_TOO_LARGE);
        assert(error.detail == NANOGLIA_DETAIL_OUT_OF_MEMORY);
        assert(error.size > 0);
    }
}

static nanoglia_status load_corrupted(size_t length, nanoglia_error *error) {
    nanoglia *nn = NULL;
    nanoglia_status status = nanoglia_load(corrupted, length, memory, MEMORY_CAPACITY, &nn, error);
    if (status == NANOGLIA_STATUS_OK) {
        /* A corruption the model survives: it must still run, or fail cleanly. */
        nanoglia_compute(nn, NULL);
        assert(nanoglia_release(nn) == NANOGLIA_STATUS_OK);
    }
    return status;
}

static void rejects_corrupted_models(void) {
    nanoglia_error error;
    memcpy(corrupted, "not a model, not even close", 27);
    assert(load_corrupted(27, &error) == NANOGLIA_STATUS_INVALID_ENCODING);

    memcpy(corrupted, model, model_length);
    assert(load_corrupted(model_length / 2, &error) == NANOGLIA_STATUS_INVALID_ENCODING);
    assert(error.detail == NANOGLIA_DETAIL_MALFORMED_MODEL);

    /* Flip bytes all over the model: an error or a model that runs, never a crash. */
    uint64_t state = 0x9e3779b97f4a7c15u;
    for (int round = 0; round < 200; round++) {
        memcpy(corrupted, model, model_length);
        for (int flip = 0; flip < 8; flip++) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            corrupted[state % model_length] ^= (unsigned char)((state >> 32) | 1);
        }
        load_corrupted(model_length, &error);
    }
}

static void describes_tensors(void) {
    nanoglia *nn = load();
    nanoglia_tensor tensor;
    assert(nanoglia_input(nn, 0, &tensor) == NANOGLIA_STATUS_OK);
    assert(tensor.element_type == NANOGLIA_TYPE_INT8);
    assert(tensor.rank == 4);
    assert(tensor.dims[0] == 1 && tensor.dims[1] == 49 && tensor.dims[2] == 10 && tensor.dims[3] == 1);
    assert(tensor.size == 490);
    assert(tensor.zero_point == 83);
    assert(tensor.scale > 0.58f && tensor.scale < 0.59f);
    assert(tensor.name_length == 7 && memcmp(tensor.name, "input_1", 7) == 0);
    /* The input lies in the buffer of the handle. */
    assert((unsigned char *)tensor.data >= memory &&
           (unsigned char *)tensor.data + tensor.size <= memory + needs.memory_bytes);

    assert(nanoglia_output(nn, 0, &tensor) == NANOGLIA_STATUS_OK);
    assert(tensor.element_type == NANOGLIA_TYPE_INT8);
    assert(tensor.rank == 2 && tensor.dims[0] == 1 && tensor.dims[1] == 12);
    assert(tensor.zero_point == -128 && tensor.scale == 1.0f / 256);

    assert(nanoglia_input(nn, 1, &tensor) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_output(nn, 1, &tensor) == NANOGLIA_STATUS_INVALID_ARGUMENT);

    size_t index = 99;
    assert(nanoglia_input_index(nn, "input_1", 7, &index) == NANOGLIA_STATUS_OK && index == 0);
    assert(nanoglia_output_index(nn, "Identity", 8, &index) == NANOGLIA_STATUS_OK && index == 0);
    assert(nanoglia_output_index(nn, "Identity", 3, &index) == NANOGLIA_STATUS_NOT_FOUND);
    assert(nanoglia_input_index(nn, "output", 6, &index) == NANOGLIA_STATUS_NOT_FOUND);
    assert(nanoglia_release(nn) == NANOGLIA_STATUS_OK);
}

static void rejects_invalid_arguments(void) {
    nanoglia_error error;
    nanoglia_memory sizes;
    nanoglia_tensor tensor;
    nanoglia *nn = NULL;
    size_t index;
    bool done;
    static _Alignas(16) unsigned char scratch[1024];

    assert(nanoglia_memory_requirements(NULL, model_length, scratch, sizeof scratch, &sizes, &error) ==
           NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(error.status == NANOGLIA_STATUS_INVALID_ARGUMENT && error.detail == NANOGLIA_DETAIL_NONE);
    assert(nanoglia_memory_requirements(model, model_length, NULL, sizeof scratch, &sizes, &error) ==
           NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_memory_requirements(model, model_length, scratch, sizeof scratch, NULL, NULL) ==
           NANOGLIA_STATUS_INVALID_ARGUMENT);
    /* A model off its 16-byte boundary. */
    assert(nanoglia_memory_requirements(model + 1, model_length - 1, scratch, sizeof scratch, &sizes, NULL) ==
           NANOGLIA_STATUS_INVALID_ARGUMENT);

    assert(nanoglia_load(NULL, model_length, memory, MEMORY_CAPACITY, &nn, &error) ==
           NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_load(model, model_length, NULL, MEMORY_CAPACITY, &nn, &error) ==
           NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_load(model, model_length, memory, MEMORY_CAPACITY, NULL, &error) ==
           NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_load(model + 1, model_length - 1, memory, MEMORY_CAPACITY, &nn, NULL) ==
           NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nn == NULL);

    assert(nanoglia_release(NULL) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_input_count(NULL) == 0);
    assert(nanoglia_output_count(NULL) == 0);
    assert(nanoglia_operator_count(NULL) == 0);
    assert(nanoglia_input(NULL, 0, &tensor) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_output(NULL, 0, &tensor) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_input_index(NULL, "input_1", 7, &index) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_output_index(NULL, "Identity", 8, &index) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_compute(NULL, &error) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_step(NULL, &done, &error) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_cancel(NULL) == NANOGLIA_STATUS_INVALID_ARGUMENT);

    nn = load();
    assert(nanoglia_input(nn, 0, NULL) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_output(nn, 0, NULL) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_input_index(nn, NULL, 0, &index) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_output_index(nn, "Identity", 8, NULL) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_step(nn, NULL, NULL) == NANOGLIA_STATUS_INVALID_ARGUMENT);

    /* A released handle is no longer one, as long as its buffer is not reused. */
    assert(nanoglia_release(nn) == NANOGLIA_STATUS_OK);
    assert(nanoglia_release(nn) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_compute(nn, NULL) == NANOGLIA_STATUS_INVALID_ARGUMENT);
    assert(nanoglia_input_count(nn) == 0);
    /* Nor is memory that never held a handle. */
    assert(nanoglia_compute((nanoglia *)scratch, NULL) == NANOGLIA_STATUS_INVALID_ARGUMENT);
}

static void describes_statuses(void) {
    assert(strcmp(nanoglia_status_description(NANOGLIA_STATUS_OK), "success") == 0);
    assert(strcmp(nanoglia_status_description(NANOGLIA_STATUS_INVALID_ENCODING), "invalid encoding") == 0);
    assert(strcmp(nanoglia_status_description(1000), "unknown status") == 0);
    assert(strcmp(nanoglia_detail_description(NANOGLIA_DETAIL_MALFORMED_MODEL), "malformed model") == 0);
    assert(strcmp(nanoglia_detail_description(-5), "unknown detail") == 0);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s TESTDATA_DIRECTORY\n", argv[0]);
        return 2;
    }
    testdata = argv[1];
    model_length = read_file("mlperf-tiny/kws_ref_model.tflite", model, sizeof model);

    computes_memory_requirements();
    runs_the_golden_samples();
    cancels();
    needs_exactly_memory_bytes();
    rejects_corrupted_models();
    describes_tensors();
    rejects_invalid_arguments();
    describes_statuses();
    printf("all tests passed\n");
    return 0;
}
