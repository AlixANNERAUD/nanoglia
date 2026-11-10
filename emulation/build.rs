//! Puts the memory layout of the target on the linker search path and links with the runtime's
//! linker script. Passed here rather than in `.cargo/config.toml`, which a `RUSTFLAGS` environment
//! variable (CI sets one) would override.

use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let (layout, script) = match arch.as_str() {
        // cortex-m-rt's link.x includes memory.x.
        "arm" => ("memory-arm.x", "link.x"),
        // riscv-rt's link.x includes memory.x with its `memory` feature.
        "riscv32" => ("memory-riscv.x", "link.x"),
        other => panic!("no memory layout for {other}"),
    };
    let memory = manifest.join(layout);
    fs::copy(&memory, out.join("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-arg-bins=-T{script}");
    println!("cargo:rerun-if-changed={}", memory.display());
}
