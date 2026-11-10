//! Command-line arguments, parsed by hand: one command and a few options do not need a crate.

use std::ffi::OsString;
use std::path::PathBuf;

/// The text of `nanoglia --help`.
pub const HELP: &str = "\
Check whether nanoglia runs a TensorFlow Lite model, before flashing anything.

Usage: nanoglia check [OPTIONS] <MODEL.tflite>...

Lists every operator of the model with the kernel that runs it, or the reason no
kernel does, then the memory the model needs and the Cargo features of nanoglia
a firmware needs to run it. With several models (a firmware that loads them
all), each gets its report, then the features of a firmware that runs them all.

Options:
      --backends <LIST>  Backends of the firmware, a comma-separated list of
                         cmsis-nn, esp-nn and kernels: the Cargo features of
                         nanoglia with the same names [default: cmsis-nn,kernels]
      --ram <SIZE>       RAM budget for the graph and one execution context of
                         each model (bytes, or with a binary suffix: 256k, 1M):
                         the check fails if they do not fit
      --json             Print one JSON document instead of the table
  -h, --help             Print this help
  -V, --version          Print the version

Exit status: 0 if every model runs (and fits the budget), 1 if one does not,
2 if the arguments are wrong or a file cannot be read.

The float32 and int16 kernels, the rounding mode and the operators are Cargo
features of this tool (float, int16, single-rounding, all-ops or the op-* feature
of each operator), as they are features of nanoglia: build it with the features
of your firmware. The output says which ones were built in.";

/// The backends a check considers: the Cargo features `esp-nn`, `cmsis-nn` and `kernels` of
/// `nanoglia`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Backends {
    pub esp_nn: bool,
    pub cmsis_nn: bool,
    pub kernels: bool,
}

impl Default for Backends {
    /// The default features of `nanoglia`.
    fn default() -> Self {
        Self {
            esp_nn: false,
            cmsis_nn: true,
            kernels: true,
        }
    }
}

impl Backends {
    fn parse(list: &str) -> Result<Self, String> {
        let mut backends = Self {
            esp_nn: false,
            cmsis_nn: false,
            kernels: false,
        };
        for name in list.split(',').map(str::trim) {
            match name {
                "esp-nn" => backends.esp_nn = true,
                "cmsis-nn" => backends.cmsis_nn = true,
                "kernels" => backends.kernels = true,
                other => {
                    return Err(format!(
                        "unknown backend `{other}` (expected cmsis-nn, esp-nn or kernels)"
                    ));
                }
            }
        }
        Ok(backends)
    }

    /// The feature names, in the order the registry tries the backends.
    pub fn names(&self) -> Vec<&'static str> {
        [
            (self.esp_nn, "esp-nn"),
            (self.cmsis_nn, "cmsis-nn"),
            (self.kernels, "kernels"),
        ]
        .into_iter()
        .filter_map(|(enabled, name)| enabled.then_some(name))
        .collect()
    }
}

/// The arguments of `nanoglia check`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Check {
    /// The models, at least one, in the order given.
    pub models: Vec<PathBuf>,
    pub json: bool,
    pub backends: Backends,
    /// RAM budget in bytes.
    pub ram: Option<usize>,
}

/// What the command line asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Check(Check),
    Help,
    Version,
}

/// A size in bytes: digits, optionally followed by a binary suffix (`k`, `K`, `kB`, `KiB`: 1024;
/// `M`, `MB`, `MiB`: 1024²), as RAM sizes of microcontrollers are given.
pub fn parse_size(text: &str) -> Result<usize, String> {
    let invalid = || format!("invalid size `{text}` (expected bytes, or a number with k or M)");
    let digits = text.bytes().take_while(u8::is_ascii_digit).count();
    let (number, suffix) = text.split_at(digits);
    let number: usize = number.parse().map_err(|_| invalid())?;
    let unit: usize = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1024,
        "m" | "mb" | "mib" => 1024 * 1024,
        _ => return Err(invalid()),
    };
    number.checked_mul(unit).ok_or_else(invalid)
}

/// Parses the arguments that follow the program name.
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let mut command = None;
    let mut models = Vec::new();
    let mut json = false;
    let mut backends = Backends::default();
    let mut ram = None;
    while let Some(arg) = args.next() {
        let Some(text) = arg.to_str() else {
            // Only a path may hold bytes that are not UTF-8.
            positional(&mut command, &mut models, arg);
            continue;
        };
        let (name, inline) = match text.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_owned())),
            _ => (text, None),
        };
        let mut value = |name: &str| -> Result<String, String> {
            match inline.clone().or_else(|| args.next()?.into_string().ok()) {
                Some(value) => Ok(value),
                None => Err(format!("`{name}` needs a value")),
            }
        };
        match name {
            "-h" | "--help" | "-V" | "--version" | "--json" if inline.is_some() => {
                return Err(format!("`{name}` takes no value"));
            }
            "-h" | "--help" => return Ok(Command::Help),
            "-V" | "--version" => return Ok(Command::Version),
            "--json" => json = true,
            "--backends" => backends = Backends::parse(&value(name)?)?,
            "--ram" => ram = Some(parse_size(&value(name)?)?),
            option if option.starts_with('-') && option.len() > 1 => {
                return Err(format!("unknown option `{option}`"));
            }
            _ => positional(&mut command, &mut models, arg),
        }
    }
    match (command.as_deref(), models.is_empty()) {
        (Some("check"), false) => Ok(Command::Check(Check {
            models,
            json,
            backends,
            ram,
        })),
        (Some("check"), true) => Err("`check` needs the path of a model".to_owned()),
        (Some(other), _) => Err(format!("unknown command `{other}` (expected `check`)")),
        (None, _) => Err("no command (expected `check <MODEL.tflite>`)".to_owned()),
    }
}

/// Takes the command, then the model paths.
fn positional(command: &mut Option<String>, models: &mut Vec<PathBuf>, arg: OsString) {
    if command.is_none() {
        *command = Some(arg.to_string_lossy().into_owned());
    } else {
        models.push(PathBuf::from(arg));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(args: &[&str]) -> Result<Command, String> {
        parse(args.iter().map(OsString::from))
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_size("4096"), Ok(4096));
        assert_eq!(parse_size("256k"), Ok(256 * 1024));
        assert_eq!(parse_size("256KiB"), Ok(256 * 1024));
        assert_eq!(parse_size("1M"), Ok(1024 * 1024));
        assert_eq!(parse_size("2 MB"), Ok(2 * 1024 * 1024));
        for bad in ["", "k", "12x", "-1", "1.5M", "99999999999999999999M"] {
            assert!(parse_size(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn check_with_defaults_and_options() {
        assert_eq!(
            parse_str(&["check", "model.tflite"]),
            Ok(Command::Check(Check {
                models: vec![PathBuf::from("model.tflite")],
                json: false,
                backends: Backends::default(),
                ram: None,
            }))
        );
        let Ok(Command::Check(check)) = parse_str(&[
            "check",
            "--backends=kernels, esp-nn",
            "model.tflite",
            "--ram",
            "64k",
            "--json",
        ]) else {
            panic!("not a check");
        };
        assert_eq!(check.backends.names(), ["esp-nn", "kernels"]);
        assert_eq!(check.ram, Some(65536));
        assert!(check.json);

        let Ok(Command::Check(check)) = parse_str(&["check", "a.tflite", "--json", "b.tflite"])
        else {
            panic!("not a check");
        };
        assert_eq!(
            check.models,
            [PathBuf::from("a.tflite"), PathBuf::from("b.tflite")]
        );
    }

    #[test]
    fn help_version_and_errors() {
        assert_eq!(parse_str(&["--help"]), Ok(Command::Help));
        assert_eq!(parse_str(&["check", "-h"]), Ok(Command::Help));
        assert_eq!(parse_str(&["-V"]), Ok(Command::Version));
        for bad in [
            &[][..],
            &["check"],
            &["run", "model.tflite"],
            &["check", "a.tflite", "--ram"],
            &["check", "a.tflite", "--backends", "tflm"],
            &["check", "a.tflite", "--backends", ""],
            &["check", "a.tflite", "--fast"],
        ] {
            assert!(parse_str(bad).is_err(), "{bad:?}");
        }
    }
}
