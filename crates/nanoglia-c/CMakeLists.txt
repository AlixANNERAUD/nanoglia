cmake_minimum_required(VERSION 3.21)

# Builds libnanoglia_c.a with Cargo and exposes it as the `nanoglia` target (alias
# `nanoglia::nanoglia`), with the header include/nanoglia.h. Add this directory to your project
# (add_subdirectory) or install it and use find_package(nanoglia).
project(nanoglia VERSION 0.1.0 LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(NANOGLIA_BUILD_TESTS "Build the tests of nanoglia" ${PROJECT_IS_TOP_LEVEL})
option(NANOGLIA_BUILD_EXAMPLES "Build the examples of nanoglia" ${PROJECT_IS_TOP_LEVEL})
set(NANOGLIA_CARGO_TARGET "" CACHE STRING "Rust target triple to build nanoglia for, such as thumbv7em-none-eabihf; empty for the host")
if(NANOGLIA_CARGO_TARGET)
    set(nanoglia_std_default OFF)
else()
    set(nanoglia_std_default ON)
endif()
option(NANOGLIA_STD "Link the Rust standard library (needed on the host): its panic handler, and panics are caught at the C boundary" ${nanoglia_std_default})
option(NANOGLIA_PANIC_HANDLER "Without NANOGLIA_STD, include a panic handler that loops forever; turn it off if another Rust library of the program provides one" ON)
option(NANOGLIA_INT8_ONLY "Leave out the float32 and int16 operators to save flash" OFF)
option(NANOGLIA_ESP_NN "Use the ESP-NN kernels first (ESP32-S3 optimized on xtensa-esp32s3)" OFF)
set(NANOGLIA_CARGO_PROFILE "release" CACHE STRING "Cargo profile to build nanoglia with")
set(NANOGLIA_CARGO_FLAGS "" CACHE STRING "Extra arguments of cargo, such as --features single-rounding")
set(NANOGLIA_CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo" CACHE PATH "Cargo target directory")

find_program(CARGO cargo REQUIRED)

set(nanoglia_features "")
if(NANOGLIA_INT8_ONLY)
    list(APPEND nanoglia_features cmsis-nn kernels)
else()
    list(APPEND nanoglia_features cmsis-nn kernels float int16)
endif()
if(NANOGLIA_ESP_NN)
    list(APPEND nanoglia_features esp-nn)
endif()
if(NANOGLIA_STD)
    list(APPEND nanoglia_features std)
elseif(NANOGLIA_PANIC_HANDLER)
    list(APPEND nanoglia_features panic-handler)
endif()
list(JOIN nanoglia_features "," nanoglia_features)

# `cargo rustc --crate-type staticlib`: the crate is an rlib for Cargo, so that a Rust program can
# depend on it without its panic handler.
set(nanoglia_cargo_arguments
    rustc
    --package nanoglia-c
    --lib
    --crate-type staticlib
    --profile "${NANOGLIA_CARGO_PROFILE}"
    --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
    --target-dir "${NANOGLIA_CARGO_TARGET_DIR}"
    --no-default-features
    --features "${nanoglia_features}"
)
if(NANOGLIA_CARGO_PROFILE STREQUAL "dev")
    set(nanoglia_profile_directory debug)
else()
    set(nanoglia_profile_directory "${NANOGLIA_CARGO_PROFILE}")
endif()
if(NANOGLIA_CARGO_TARGET)
    list(APPEND nanoglia_cargo_arguments --target "${NANOGLIA_CARGO_TARGET}")
    set(nanoglia_cargo_output "${NANOGLIA_CARGO_TARGET_DIR}/${NANOGLIA_CARGO_TARGET}/${nanoglia_profile_directory}")
else()
    set(nanoglia_cargo_output "${NANOGLIA_CARGO_TARGET_DIR}/${nanoglia_profile_directory}")
endif()
separate_arguments(nanoglia_cargo_extra_flags NATIVE_COMMAND "${NANOGLIA_CARGO_FLAGS}")
list(APPEND nanoglia_cargo_arguments ${nanoglia_cargo_extra_flags})

# Rust names the archive libnanoglia_c.a, also with an MSVC target (nanoglia_c.lib).
if(MSVC)
    set(nanoglia_library "${nanoglia_cargo_output}/nanoglia_c.lib")
else()
    set(nanoglia_library "${nanoglia_cargo_output}/libnanoglia_c.a")
endif()

# Always run, and part of `all` (for install): Cargo knows when the library is up to date.
add_custom_target(nanoglia_cargo ALL
    COMMAND "${CARGO}" ${nanoglia_cargo_arguments}
    BYPRODUCTS "${nanoglia_library}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building nanoglia with cargo (features ${nanoglia_features})"
    USES_TERMINAL
    VERBATIM
)

# The system libraries the standard library needs, on a host.
set(nanoglia_system_libraries "")
if(NANOGLIA_STD)
    if(WIN32)
        list(APPEND nanoglia_system_libraries ws2_32 userenv ntdll bcrypt)
    elseif(APPLE)
        list(APPEND nanoglia_system_libraries m)
    elseif(UNIX)
        find_package(Threads REQUIRED)
        list(APPEND nanoglia_system_libraries Threads::Threads ${CMAKE_DL_LIBS} m)
    endif()
endif()

add_library(nanoglia STATIC IMPORTED GLOBAL)
add_library(nanoglia::nanoglia ALIAS nanoglia)
set_target_properties(nanoglia PROPERTIES
    IMPORTED_LOCATION "${nanoglia_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
    INTERFACE_LINK_LIBRARIES "${nanoglia_system_libraries}"
)
add_dependencies(nanoglia nanoglia_cargo)

# install: the archive, the header and a package configuration for find_package(nanoglia).
install(FILES "${nanoglia_library}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES include/nanoglia.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
get_filename_component(NANOGLIA_LIBRARY_NAME "${nanoglia_library}" NAME)
set(NANOGLIA_NEEDS_THREADS OFF)
if(NANOGLIA_STD AND UNIX AND NOT APPLE)
    set(NANOGLIA_NEEDS_THREADS ON)
endif()
string(REPLACE "Threads::Threads" "" NANOGLIA_SYSTEM_LIBRARIES "${nanoglia_system_libraries}")
configure_package_config_file(cmake/nanogliaConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/nanogliaConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nanoglia"
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/nanogliaConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/nanogliaConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/nanogliaConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nanoglia"
)

function(nanoglia_strict_c target)
    set_target_properties(${target} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    target_compile_options(${target} PRIVATE
        $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Werror>)
endfunction()

set(nanoglia_testdata "${CMAKE_CURRENT_SOURCE_DIR}/../../testdata")

if(NANOGLIA_BUILD_TESTS)
    enable_testing()
    add_executable(nanoglia_test tests/test.c)
    target_link_libraries(nanoglia_test PRIVATE nanoglia)
    nanoglia_strict_c(nanoglia_test)
    add_test(NAME nanoglia_test COMMAND nanoglia_test "${nanoglia_testdata}")

    # The header alone, as C99.
    add_library(nanoglia_header_c99 OBJECT tests/header.c)
    target_link_libraries(nanoglia_header_c99 PRIVATE nanoglia)
    set_target_properties(nanoglia_header_c99 PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    target_compile_options(nanoglia_header_c99 PRIVATE
        $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Werror>)
endif()

if(NANOGLIA_BUILD_EXAMPLES)
    add_executable(nanoglia_keyword_spotting examples/keyword_spotting.c)
    target_link_libraries(nanoglia_keyword_spotting PRIVATE nanoglia)
    nanoglia_strict_c(nanoglia_keyword_spotting)
    if(NANOGLIA_BUILD_TESTS)
        # Sample 0 of the golden data: its highest score is class 9.
        add_test(NAME nanoglia_keyword_spotting
            COMMAND nanoglia_keyword_spotting
                "${nanoglia_testdata}/mlperf-tiny/kws_ref_model.tflite"
                "${nanoglia_testdata}/mlperf-tiny/golden/kws_ref_model/sample_0/input_0.bin")
        set_tests_properties(nanoglia_keyword_spotting PROPERTIES PASS_REGULAR_EXPRESSION "class 9,")
    endif()
endif()
