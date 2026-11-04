/* The header compiles on its own, in C99 with every warning (built by CMake, not linked). */

#include "nanoglia.h"

int nanoglia_header_compiles(void);

int nanoglia_header_compiles(void) {
    nanoglia_status status = NANOGLIA_STATUS_OK;
    nanoglia_detail detail = NANOGLIA_DETAIL_NONE;
    nanoglia_type type = NANOGLIA_TYPE_INT8;
    return (int)status + (int)detail + (int)type + NANOGLIA_MAX_RANK + NANOGLIA_MODEL_ALIGNMENT;
}
