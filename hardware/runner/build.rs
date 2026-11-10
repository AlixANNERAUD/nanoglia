//! Passes the version of nanoglia, the revision of the sources and the build settings to the
//! runner, which reports them in its answer to `INFO`.

use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let nanoglia = manifest_dir.join("../../crates/nanoglia/Cargo.toml");
    println!("cargo:rerun-if-changed={}", nanoglia.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=NANOGLIA_REVISION");

    // The first `version = "..."` of the manifest is the one of the package.
    let manifest = fs::read_to_string(&nanoglia).expect("crates/nanoglia/Cargo.toml");
    let version = manifest
        .lines()
        .find_map(|line| line.strip_prefix("version = \"")?.strip_suffix('"'))
        .expect("the version of nanoglia");
    println!("cargo:rustc-env=RUNNER_NANOGLIA_VERSION={version}");

    // The mise tasks pass the revision, so that a build is rerun when it changes; otherwise ask
    // git, once.
    let revision = env::var("NANOGLIA_REVISION")
        .ok()
        .filter(|r| !r.is_empty())
        .or_else(|| {
            let output = Command::new("git")
                .args([
                    "describe",
                    "--always",
                    "--dirty",
                    "--abbrev=12",
                    "--exclude=*",
                ])
                .current_dir(&manifest_dir)
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        });
    println!(
        "cargo:rustc-env=RUNNER_REVISION={}",
        revision.as_deref().unwrap_or("unknown")
    );

    println!(
        "cargo:rustc-env=RUNNER_BUILD=cmsis-nn kernels float int16 all-ops, {}",
        env::var("TARGET").unwrap()
    );
    println!(
        "cargo:rustc-env=RUNNER_OPT_LEVEL={}",
        env::var("OPT_LEVEL").unwrap()
    );

    // The compiler, as `rustc --version` names it.
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    println!(
        "cargo:rustc-env=RUNNER_RUSTC={}",
        version.as_deref().unwrap_or("unknown")
    );
}
