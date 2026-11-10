//! Chooses the model `src/main.rs` embeds: the file named by the `GESTURE_MODEL` environment
//! variable (a model the training scripts just exported), or the committed reference model.

use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=GESTURE_MODEL");
    let model = match env::var_os("GESTURE_MODEL") {
        Some(path) => PathBuf::from(path),
        None => {
            PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("reference/model.tflite")
        }
    };
    let model = model
        .canonicalize()
        .unwrap_or_else(|e| panic!("{}: {e}", model.display()));
    println!("cargo:rerun-if-changed={}", model.display());
    println!("cargo:rustc-env=GESTURE_MODEL_PATH={}", model.display());
}
