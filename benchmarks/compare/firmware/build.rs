//! Build script shared by the firmware crates (`build = "../build.rs"`): puts `firmware/memory.x`
//! on the linker search path, for cortex-m-rt's `link.x`.

use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let memory = manifest.join("../memory.x");
    fs::copy(&memory, out.join("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed={}", memory.display());
}
