//! Compiles ESP-NN (the `esp-nn` git submodule) into a static library.
//!
//! Only files of `esp-nn/src` and `esp-nn/include` are read: the tests, test application and
//! tools of the repository are never built. Keep the file selection below in sync with the
//! `include` list of `Cargo.toml`.
//!
//! The ANSI C reference kernels are always built, so the crate builds (and its users can be
//! tested) on any target. On ESP32-S3 (`xtensa-esp32s3-*` targets) the build also enables
//! ESP-NN's optimized kernels (C and assembly using the PIE SIMD instructions), selected by its
//! headers through `CONFIG_NN_OPTIMIZED` and `CONFIG_IDF_TARGET_ESP32S3`.
//!
//! ESP-NN is compiled without `ESP_PLATFORM`: no ESP-IDF, FreeRTOS or heap dependency, a single
//! core. `CONFIG_NN_SKIP_NUDGE` stays undefined, so the requantization is bit-exact with TFLite.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let esp_nn = root.join("esp-nn");
    let src = esp_nn.join("src");
    if !esp_nn.join("include").join("esp_nn.h").is_file() {
        panic!(
            "{} not found: the ESP-NN git submodule is not checked out; run \
             `git submodule update --init --recursive` (or `mise run submodules`)",
            esp_nn.join("include").display()
        );
    }
    let target = env::var("TARGET").unwrap();
    let esp32s3 = target.starts_with("xtensa-esp32s3-");

    let mut sources = Vec::new();
    collect(&src, &mut sources, &|name| {
        // The reference kernels, except the float LUT logistic (libm's `expf`, not TFLite's
        // fixed-point arithmetic), which nanoglia does not use.
        name.ends_with("_ansi.c") && name != "esp_nn_logistic_ansi.c"
    });
    if esp32s3 {
        collect(&src, &mut sources, &|name| {
            name.ends_with("_esp32s3.c")
                || name.ends_with("_esp32s3.S")
                // Generic C optimizations, which some ESP32-S3 kernels fall back to.
                || name.ends_with("_opt.c")
                // Single-core stubs of the dual-core API (`ESP_PLATFORM` is undefined) and the
                // grouped/multi-core fronts the ESP32-S3 kernels call.
                || name == "esp_nn_multicore.c"
                || name == "esp_nn_conv_mt_split.c"
                || name == "esp_nn_mul_mt.c"
        });
    }
    sources.sort();
    sources.push(root.join("src").join("dispatch.c"));
    sources.push(root.join("src").join("layout.c"));

    let mut build = cc::Build::new();
    build
        .include(root.join("shim"))
        .include(esp_nn.join("include"))
        .include(src.join("common"))
        .files(&sources)
        .warnings(false)
        .extra_warnings(false)
        .flag_if_supported("-Wno-unused-function");
    if esp32s3 {
        build
            .define("CONFIG_NN_OPTIMIZED", "1")
            .define("CONFIG_IDF_TARGET_ESP32S3", "1")
            .define("CONFIG_IDF_TARGET_ARCH_XTENSA", "1")
            // ESP-NN's own flags for this target (its CMakeLists.txt).
            .flag("-mlongcalls")
            .flag("-fno-unroll-loops")
            // Turns the `printf` of the error paths (scratch buffer not set) into no-ops: nothing
            // links the C library.
            .flag("-include")
            .flag(
                root.join("shim")
                    .join("nanoglia_esp_nn_prelude.h")
                    .to_str()
                    .unwrap(),
            );
    }

    // The kernels are far too slow unoptimized, even in debug builds (host tests). Keep the
    // size-oriented levels of release profiles.
    let opt_level = match env::var("OPT_LEVEL").as_deref() {
        Ok("s") => "s",
        Ok("z") => "z",
        Ok("3") => "3",
        _ => "2",
    };
    build.opt_level_str(opt_level);
    build.compile("esp_nn");

    // Lets the Rust code know which kernels the dispatch functions call.
    println!("cargo:rustc-check-cfg=cfg(esp_nn_esp32s3)");
    if esp32s3 {
        println!("cargo:rustc-cfg=esp_nn_esp32s3");
    }
    println!("cargo:include={}", esp_nn.join("include").display());
    println!("cargo:rerun-if-changed=esp-nn/include");
    println!("cargo:rerun-if-changed=esp-nn/src");
    println!("cargo:rerun-if-changed=shim");
    println!("cargo:rerun-if-changed=src/dispatch.c");
    println!("cargo:rerun-if-changed=src/layout.c");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Appends the files under `dir` whose name `keep` accepts.
fn collect(dir: &Path, out: &mut Vec<PathBuf>, keep: &dyn Fn(&str) -> bool) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(&path, out, keep);
        } else if path.file_name().and_then(|n| n.to_str()).is_some_and(keep) {
            out.push(path);
        }
    }
}
