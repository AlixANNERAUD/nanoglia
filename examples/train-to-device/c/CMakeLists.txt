cmake_minimum_required(VERSION 3.21)

# The C program of the train-and-deploy tutorial: the gesture classifier compiled into the program,
# run on the test vectors exported with it (`mise run example-train-to-device`).
project(nanoglia_gesture LANGUAGES C)

set(GESTURE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../reference" CACHE PATH
    "Directory written by a training script: model.tflite and its test vectors (default: the reference model)")

# Only the kernels of the operators of the model are built into the library: those of the Keras
# model, plus DEPTHWISE_CONV_2D and RESHAPE for the PyTorch one. `nanoglia check model.tflite` lists
# the operators of a model.
set(NANOGLIA_OPERATORS
    "CONV_2D;DEPTHWISE_CONV_2D;FULLY_CONNECTED;MAX_POOL_2D;MEAN;RESHAPE;SOFTMAX" CACHE STRING
    "Operators to build kernels for")
# int8 model: no float32 nor int16 kernel.
set(NANOGLIA_INT8_ONLY ON CACHE BOOL "Leave out the float32 and int16 operators")

# nanoglia's C library, built by Cargo, as the target nanoglia::nanoglia.
add_subdirectory(../../../crates/nanoglia-c nanoglia)

# model.inc: the bytes of the model as a C initializer list, which gesture.c includes.
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/model.inc"
    COMMAND "${CMAKE_COMMAND}"
        "-DINPUT=${GESTURE_DIRECTORY}/model.tflite"
        "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/model.inc"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/embed.cmake"
    DEPENDS "${GESTURE_DIRECTORY}/model.tflite" embed.cmake
    COMMENT "Embedding ${GESTURE_DIRECTORY}/model.tflite"
    VERBATIM
)

add_executable(gesture gesture.c "${CMAKE_CURRENT_BINARY_DIR}/model.inc")
target_include_directories(gesture PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(gesture PRIVATE nanoglia::nanoglia)
set_target_properties(gesture PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_compile_options(gesture PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Werror>)

# Fails unless every sample gives the bytes LiteRT computed.
enable_testing()
add_test(NAME gesture COMMAND gesture "${GESTURE_DIRECTORY}")
