cmake_minimum_required(VERSION 3.21)

# A project that uses an installed nanoglia through find_package: the example, linked to
# nanoglia::nanoglia, run on golden sample 0 (`mise run test-c`).
project(nanoglia_package_test LANGUAGES C)

find_package(nanoglia 0.1 REQUIRED)

add_executable(keyword_spotting ../../examples/keyword_spotting.c)
target_link_libraries(keyword_spotting PRIVATE nanoglia::nanoglia)
set_target_properties(keyword_spotting PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)

enable_testing()
set(testdata "${CMAKE_CURRENT_SOURCE_DIR}/../../../../testdata")
add_test(NAME keyword_spotting
    COMMAND keyword_spotting
        "${testdata}/mlperf-tiny/kws_ref_model.tflite"
        "${testdata}/mlperf-tiny/golden/kws_ref_model/sample_0/input_0.bin")
set_tests_properties(keyword_spotting PROPERTIES PASS_REGULAR_EXPRESSION "class 9,")
