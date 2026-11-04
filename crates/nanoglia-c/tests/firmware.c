/*
 * Links the static library into a bare-metal program without any C library (`mise run cross-c`:
 * arm-none-eabi-gcc -nostdlib): every symbol nanoglia needs must come from the archive itself, so
 * it cannot need malloc or free. The program is linked, never run.
 */

#include "nanoglia.h"

static _Alignas(NANOGLIA_MODEL_ALIGNMENT) const uint8_t model[1024];
static unsigned char memory[32 * 1024];

int main(void);

int main(void) {
    nanoglia_memory needs;
    nanoglia_error error;
    nanoglia *nn;
    nanoglia_tensor tensor;
    size_t index;
    bool done = false;
    int failures = 0;

    failures += nanoglia_memory_requirements(model, sizeof model, memory, sizeof memory, &needs, &error) != 0;
    if (nanoglia_load(model, sizeof model, memory, sizeof memory, &nn, &error) != NANOGLIA_STATUS_OK) {
        return (int)error.detail;
    }
    failures += nanoglia_input(nn, 0, &tensor) != 0;
    failures += nanoglia_input_index(nn, "input_1", 7, &index) != 0;
    failures += nanoglia_compute(nn, &error) != 0;
    while (!done && nanoglia_step(nn, &done, &error) == NANOGLIA_STATUS_OK) {
    }
    failures += nanoglia_cancel(nn) != 0;
    failures += nanoglia_output(nn, 0, &tensor) != 0;
    failures += nanoglia_output_index(nn, "Identity", 8, &index) != 0;
    failures += (int)(nanoglia_input_count(nn) + nanoglia_output_count(nn) + nanoglia_operator_count(nn));
    failures += nanoglia_status_description(error.status)[0] + nanoglia_detail_description(error.detail)[0];
    failures += nanoglia_release(nn) != 0;
    return failures;
}
