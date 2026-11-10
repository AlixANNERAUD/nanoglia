//! Puts `memory.x` on the linker search path and passes the linker scripts.
//!
//! The scripts are given here rather than in `.cargo/config.toml`: a `RUSTFLAGS` environment
//! variable (CI sets one) replaces the `rustflags` of the configuration file, and the firmware
//! would then link empty.

use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::copy("memory.x", out.join("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    println!("cargo:rustc-link-arg-bins=-Tdefmt.x");
}
