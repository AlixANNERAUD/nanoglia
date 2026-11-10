//! Puts `memory.x` on the linker search path and links with cortex-m-rt's linker script. Passed
//! here rather than in `.cargo/config.toml`, which a `RUSTFLAGS` environment variable (CI sets
//! one) would override.

use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::copy("memory.x", out.join("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    println!("cargo:rerun-if-changed=memory.x");
}
