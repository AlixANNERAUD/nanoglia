//! The Arm extensions of the Rust target, and the matching C compiler flags: pure functions of the
//! build script's environment, unit-tested through `tests/build_target.rs`.
//!
//! Workaround (issue #28): `CARGO_CFG_TARGET_FEATURE` lists only the features stable Rust knows,
//! and the Arm `dsp` and `mve` features are unstable: stable Rust leaves them out even under
//! `-C target-cpu=cortex-m33`. [`target_features`] adds them from the `-C target-cpu` and
//! `-C target-feature` flags of `CARGO_ENCODED_RUSTFLAGS`. Drop the flag parsing once stable Rust
//! reports `dsp` and `mve` in `CARGO_CFG_TARGET_FEATURE` (rust-lang/rust's `arm_target_feature`,
//! unstable).

use std::collections::BTreeSet;

/// The DSP and MVE features of `cpu` (a `-C target-cpu` value), as LLVM defines the CPU:
/// `rustc +nightly --print cfg --target thumbv8m.main-none-eabihf -C target-cpu=<cpu>` lists
/// them. Unknown CPUs (and `native`) add nothing, so CMSIS-NN falls back to fewer extensions,
/// never more.
fn cpu_features(cpu: &str) -> &'static [&'static str] {
    match cpu {
        "cortex-m33" | "cortex-m35p" | "star-mc1" => &["dsp"],
        "cortex-m52" | "cortex-m55" | "cortex-m85" | "star-mc3" => &["dsp", "mve", "mve.fp"],
        _ => &[],
    }
}

/// The features that enabling `feature` also enables (LLVM: MVE needs DSP, MVE float needs MVE).
fn implied_by(feature: &str) -> &'static [&'static str] {
    match feature {
        "mve" => &["dsp"],
        "mve.fp" => &["mve", "dsp"],
        _ => &[],
    }
}

/// The features that disabling `feature` also disables (the reverse of [`implied_by`]).
fn disabled_with(feature: &str) -> &'static [&'static str] {
    match feature {
        "dsp" => &["mve", "mve.fp"],
        "mve" => &["mve.fp"],
        _ => &[],
    }
}

/// The values of the `-C <key>=…` codegen options of `rustflags` (`CARGO_ENCODED_RUSTFLAGS`,
/// arguments separated by `0x1f`), in order. Accepts every spelling rustc does: `-C key=v`,
/// `-Ckey=v`, `--codegen key=v` and `--codegen=key=v`.
fn codegen_values<'a>(rustflags: &'a str, key: &str) -> Vec<&'a str> {
    let mut values = Vec::new();
    let mut args = rustflags.split('\x1f');
    while let Some(arg) = args.next() {
        let option = match arg {
            "-C" | "--codegen" => match args.next() {
                Some(option) => option,
                None => break,
            },
            _ => match arg
                .strip_prefix("--codegen=")
                .or_else(|| arg.strip_prefix("-C"))
            {
                Some(option) => option,
                None => continue,
            },
        };
        if let Some((name, value)) = option.split_once('=')
            && name == key
        {
            values.push(value);
        }
    }
    values
}

/// The target features of the Rust code: `cfg` (`CARGO_CFG_TARGET_FEATURE`, comma-separated),
/// plus what the `-C target-cpu` and `-C target-feature` options of `rustflags` enable, as rustc
/// and LLVM compute them:
///
/// - the last `target-cpu` wins and gives the base features, wherever it sits among the
///   `target-feature` options;
/// - the `target-feature` entries (comma-separated `+name`/`-name`, over every `target-feature`
///   option) then apply in order: `+mve.fp` also enables `mve` and `dsp`, `-dsp` also disables
///   `mve` and `mve.fp`, and so on.
///
/// So `-C target-feature=-dsp -C target-cpu=cortex-m33` has no DSP, like
/// `-C target-cpu=cortex-m33 -C target-feature=-dsp`.
pub fn target_features(cfg: &str, rustflags: &str) -> BTreeSet<String> {
    let mut features: BTreeSet<String> = cfg
        .split(',')
        .filter(|f| !f.is_empty())
        .map(str::to_string)
        .collect();
    if let Some(cpu) = codegen_values(rustflags, "target-cpu").last() {
        features.extend(cpu_features(cpu).iter().map(|f| f.to_string()));
    }
    for list in codegen_values(rustflags, "target-feature") {
        for entry in list.split(',') {
            if let Some(name) = entry.strip_prefix('+') {
                features.insert(name.to_string());
                features.extend(implied_by(name).iter().map(|f| f.to_string()));
            } else if let Some(name) = entry.strip_prefix('-') {
                features.remove(name);
                for dependent in disabled_with(name) {
                    features.remove(*dependent);
                }
            }
        }
    }
    features
}

/// CPU flags for Arm M-profile targets, so the compiler defines `__ARM_FEATURE_DSP` and
/// `__ARM_FEATURE_MVE` where the architecture has them. `cc` already passes `-march`, `-mthumb`,
/// `-mfloat-abi` and `-mfpu`; a later `-march` overrides it, so extensions are added the same way
/// and the scheduling model is picked with `-mtune` (mixing `-mcpu` with `-march` would warn).
pub fn arm_cpu_flags(target: &str, features: &BTreeSet<String>) -> Vec<&'static str> {
    let has = |f: &str| features.contains(f);
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
        // Cortex-M52/M55/M85: Helium (MVE) when the Rust target enables it.
        "thumbv8.1m.main" if has("mve.fp") => {
            vec!["-march=armv8.1-m.main+mve.fp+fp.dp", "-mtune=cortex-m55"]
        }
        "thumbv8.1m.main" if has("mve") => vec!["-march=armv8.1-m.main+mve", "-mtune=cortex-m55"],
        "thumbv8.1m.main" if has("dsp") => vec!["-march=armv8.1-m.main+dsp", "-mtune=cortex-m55"],
        "thumbv8.1m.main" => vec!["-march=armv8.1-m.main", "-mtune=cortex-m55"],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `CARGO_ENCODED_RUSTFLAGS` for `args`.
    fn flags(args: &[&str]) -> String {
        args.join("\x1f")
    }

    /// The DSP and MVE features for `args`, with an empty cfg.
    fn arm(args: &[&str]) -> Vec<String> {
        target_features("", &flags(args))
            .into_iter()
            .filter(|f| matches!(f.as_str(), "dsp" | "mve" | "mve.fp"))
            .collect()
    }

    #[test]
    fn no_flags_keeps_the_cfg() {
        assert!(arm(&[]).is_empty());
        let cfg = target_features("thumb2,v7,dsp", "");
        assert_eq!(cfg, ["dsp", "thumb2", "v7"].map(String::from).into());
    }

    #[test]
    fn every_spelling() {
        for args in [
            &["-C", "target-cpu=cortex-m33"][..],
            &["-Ctarget-cpu=cortex-m33"],
            &["--codegen", "target-cpu=cortex-m33"],
            &["--codegen=target-cpu=cortex-m33"],
            &["-C", "target-feature=+dsp"],
            &["-Ctarget-feature=+dsp"],
            &["--codegen", "target-feature=+dsp"],
            &["--codegen=target-feature=+dsp"],
        ] {
            assert_eq!(arm(args), ["dsp"], "{args:?}");
        }
        // Other options and a dangling `-C` are ignored.
        assert!(
            arm(&[
                "-C",
                "opt-level=s",
                "-Ctarget-cpux=cortex-m33",
                "-D",
                "warnings"
            ])
            .is_empty()
        );
        assert!(arm(&["-C"]).is_empty());
        assert!(arm(&["--codegen"]).is_empty());
    }

    #[test]
    fn removal_regardless_of_order() {
        let cpu_first = arm(&["-C", "target-cpu=cortex-m33", "-C", "target-feature=-dsp"]);
        let feature_first = arm(&["-C", "target-feature=-dsp", "-C", "target-cpu=cortex-m33"]);
        assert!(cpu_first.is_empty());
        assert!(feature_first.is_empty());
    }

    #[test]
    fn removal_cascades() {
        assert!(arm(&["-Ctarget-cpu=cortex-m55", "-Ctarget-feature=-dsp"]).is_empty());
        assert!(arm(&["-Ctarget-feature=-dsp", "-Ctarget-cpu=cortex-m85"]).is_empty());
        assert_eq!(
            arm(&["-Ctarget-cpu=cortex-m55", "-Ctarget-feature=-mve"]),
            ["dsp"]
        );
        assert_eq!(
            arm(&["-Ctarget-cpu=cortex-m55", "-Ctarget-feature=-mve.fp"]),
            ["dsp", "mve"]
        );
    }

    #[test]
    fn features_apply_in_order() {
        // As LLVM: a later `+mve` enables DSP again, a later `-dsp` wins over `+mve`.
        assert_eq!(arm(&["-Ctarget-feature=-dsp,+mve"]), ["dsp", "mve"]);
        assert!(arm(&["-Ctarget-feature=+mve", "-Ctarget-feature=-dsp"]).is_empty());
        assert_eq!(arm(&["-Ctarget-feature=+mve.fp"]), ["dsp", "mve", "mve.fp"]);
        assert_eq!(arm(&["-Ctarget-feature=+dsp,-dsp,+dsp"]), ["dsp"]);
        // A comma-separated list, with features this module ignores.
        assert_eq!(arm(&["-Ctarget-feature=+v7,+dsp,-fpregs"]), ["dsp"]);
    }

    #[test]
    fn last_cpu_wins() {
        assert!(arm(&["-Ctarget-cpu=cortex-m33", "-Ctarget-cpu=cortex-m23"]).is_empty());
        assert_eq!(
            arm(&["-Ctarget-cpu=cortex-m55", "-Ctarget-cpu=cortex-m33"]),
            ["dsp"]
        );
        assert_eq!(
            arm(&["-Ctarget-cpu=cortex-m4", "-Ctarget-cpu=cortex-m55"]),
            ["dsp", "mve", "mve.fp"]
        );
    }

    #[test]
    fn cpus() {
        for cpu in ["cortex-m33", "cortex-m35p", "star-mc1"] {
            assert_eq!(arm(&["-C", &format!("target-cpu={cpu}")]), ["dsp"], "{cpu}");
        }
        for cpu in ["cortex-m52", "cortex-m55", "cortex-m85", "star-mc3"] {
            let args = ["-C", &format!("target-cpu={cpu}")];
            assert_eq!(arm(&args), ["dsp", "mve", "mve.fp"], "{cpu}");
        }
        for cpu in ["cortex-m0plus", "cortex-m23", "generic", "native"] {
            assert!(
                arm(&["-C", &format!("target-cpu={cpu}")]).is_empty(),
                "{cpu}"
            );
        }
    }

    #[test]
    fn march() {
        let m33 = target_features("", &flags(&["-C", "target-cpu=cortex-m33"]));
        let no_dsp = target_features(
            "",
            &flags(&["-C", "target-feature=-dsp", "-C", "target-cpu=cortex-m33"]),
        );
        assert_eq!(
            arm_cpu_flags("thumbv8m.main-none-eabihf", &m33)[0],
            "-march=armv8-m.main+dsp"
        );
        assert_eq!(
            arm_cpu_flags("thumbv8m.main-none-eabihf", &no_dsp)[0],
            "-march=armv8-m.main"
        );
        let m55 = target_features("", &flags(&["-Ctarget-cpu=cortex-m55"]));
        let m55_no_dsp = target_features(
            "",
            &flags(&["-Ctarget-cpu=cortex-m55", "-Ctarget-feature=-dsp"]),
        );
        assert_eq!(
            arm_cpu_flags("thumbv8.1m.main-none-eabihf", &m55)[0],
            "-march=armv8.1-m.main+mve.fp+fp.dp"
        );
        assert_eq!(
            arm_cpu_flags("thumbv8.1m.main-none-eabihf", &m55_no_dsp)[0],
            "-march=armv8.1-m.main"
        );
        assert_eq!(
            arm_cpu_flags("thumbv7em-none-eabihf", &no_dsp)[0],
            "-march=armv7e-m"
        );
        assert!(arm_cpu_flags("riscv32imc-unknown-none-elf", &m33).is_empty());
    }
}
