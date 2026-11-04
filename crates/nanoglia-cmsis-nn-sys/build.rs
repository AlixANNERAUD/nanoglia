//! Compiles the integer kernels of CMSIS-NN (the `cmsis-nn` git submodule) into a static library.
//!
//! Only the integer C files of `cmsis-nn/Source` are built: the float ones (`*_f16.c`, `*_f32.c`,
//! `*_flt.c`) would pull in libm, and `Source/Bindings` is C++ for Python.
//!
//! On Arm M-profile targets the CPU flags expose the DSP (Armv7E-M, Armv8-M Mainline) and MVE
//! (Armv8.1-M with the `mve` target feature) extensions, so CMSIS-NN picks its optimized code
//! paths. Everywhere else (host, RISC-V, Xtensa...) CMSIS-NN compiles its portable C paths.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cmsis = root.join("cmsis-nn");
    let include = cmsis.join("Include");
    if !include.join("arm_nnfunctions.h").is_file() {
        panic!(
            "{} not found: the CMSIS-NN git submodule is not checked out; run \
             `git submodule update --init --recursive` (or `mise run submodules`)",
            include.display()
        );
    }

    let mut sources = Vec::new();
    collect_c_files(&cmsis.join("Source"), &mut sources);
    sources.sort();
    sources.push(root.join("src").join("layout.c"));

    let mut build = cc::Build::new();
    build
        .include(&include)
        .files(&sources)
        .warnings(false)
        .extra_warnings(false)
        // Integer kernels only: the float API (and its libm dependency) stays out.
        .define("ARM_NN_ENABLE_F32", "0")
        .define("ARM_NN_ENABLE_F16", "0");
    // Double rounding (CMSIS-NN's and TensorFlow Lite Micro's default) unless feature
    // `single-rounding` asks for `TFLITE_SINGLE_ROUNDING` semantics.
    if env::var_os("CARGO_FEATURE_SINGLE_ROUNDING").is_some() {
        build.define("CMSIS_NN_USE_SINGLE_ROUNDING", None);
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

    let target = env::var("TARGET").unwrap();
    let features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    for flag in arm_cpu_flags(&target, &features) {
        build.flag(flag);
    }

    build.compile("cmsis_nn");

    println!("cargo:include={}", include.display());
    println!("cargo:rerun-if-changed=cmsis-nn/Include");
    println!("cargo:rerun-if-changed=cmsis-nn/Source");
    println!("cargo:rerun-if-changed=src/layout.c");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Appends the integer C files under `dir`.
fn collect_c_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_c_files(&path, out);
        } else if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(is_integer_source)
        {
            out.push(path);
        }
    }
}

/// Whether a file of `Source/` is an integer C kernel: every `.c` file except the float ones.
/// Keep in sync with the `include` list of `Cargo.toml`, which leaves the float files out of the
/// package.
fn is_integer_source(name: &str) -> bool {
    name.ends_with(".c")
        && !name.ends_with("_f16.c")
        && !name.ends_with("_f32.c")
        && !name.ends_with("_flt.c")
}

/// CPU flags for Arm M-profile targets, so the compiler defines `__ARM_FEATURE_DSP` and
/// `__ARM_FEATURE_MVE` where the architecture has them. `cc` already passes `-march`, `-mthumb`,
/// `-mfloat-abi` and `-mfpu`; a later `-march` overrides it, so extensions are added the same way
/// and the scheduling model is picked with `-mtune` (mixing `-mcpu` with `-march` would warn).
fn arm_cpu_flags(target: &str, features: &str) -> Vec<&'static str> {
    let has = |f: &str| features.split(',').any(|x| x == f);
    let arch = target.split('-').next().unwrap_or_default();
    match arch {
        // Cortex-M0/M0+/M1: no DSP.
        "thumbv6m" => vec!["-mtune=cortex-m0plus"],
        // Cortex-M3: no DSP.
        "thumbv7m" => vec!["-mtune=cortex-m3"],
        // Cortex-M4/M7: Armv7E-M always has the DSP extension.
        "thumbv7em" => vec!["-march=armv7e-m", "-mtune=cortex-m4"],
        // Cortex-M23: no DSP.
        "thumbv8m.base" => vec!["-mtune=cortex-m23"],
        // Cortex-M33/M35P: the DSP extension is optional, so it is used only when the Rust
        // target enables it (`-C target-cpu=cortex-m33` or `-C target-feature=+dsp`).
        "thumbv8m.main" if has("dsp") => vec!["-march=armv8-m.main+dsp", "-mtune=cortex-m33"],
        "thumbv8m.main" => vec!["-march=armv8-m.main", "-mtune=cortex-m33"],
        // Cortex-M55/M85: Helium (MVE) when the Rust target enables it.
        "thumbv8.1m.main" if has("mve.fp") => {
            vec!["-march=armv8.1-m.main+mve.fp+fp.dp", "-mtune=cortex-m55"]
        }
        "thumbv8.1m.main" if has("mve") => vec!["-march=armv8.1-m.main+mve", "-mtune=cortex-m55"],
        "thumbv8.1m.main" if has("dsp") => vec!["-march=armv8.1-m.main+dsp", "-mtune=cortex-m55"],
        "thumbv8.1m.main" => vec!["-march=armv8.1-m.main", "-mtune=cortex-m55"],
        _ => Vec::new(),
    }
}
