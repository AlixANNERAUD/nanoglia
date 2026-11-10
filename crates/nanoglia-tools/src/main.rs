#![forbid(unsafe_code)]
//! `nanoglia check <MODEL.tflite>`: whether nanoglia runs a model, with which kernels and how
//! much memory. See the README of the crate.

mod args;
mod output;
mod report;

use args::{Check, Command};
use output::Outcome;
use std::io::Write;
use std::process::ExitCode;

/// Exit status when the model does not run, or does not fit the RAM budget.
const DOES_NOT_RUN: u8 = 1;
/// Exit status when the arguments are wrong or the file cannot be read.
const USAGE: u8 = 2;

/// Model bytes on a 16-byte boundary, where the constants of a model are aligned for their
/// types, as a firmware places them.
struct AlignedModel {
    storage: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedModel {
    const ALIGN: usize = 16;

    fn read(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let mut storage = vec![0u8; bytes.len() + Self::ALIGN];
        // At most `ALIGN - 1` for bytes; a pointer that cannot be aligned would report more.
        let start = storage.as_ptr().align_offset(Self::ALIGN) % Self::ALIGN;
        storage[start..start + bytes.len()].copy_from_slice(&bytes);
        Ok(Self {
            storage,
            start,
            len: bytes.len(),
        })
    }

    fn bytes(&self) -> &[u8] {
        &self.storage[self.start..self.start + self.len]
    }
}

fn check(args: &Check) -> ExitCode {
    let model = match AlignedModel::read(&args.model) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("nanoglia: {}: {error}", args.model.display());
            return ExitCode::from(USAGE);
        }
    };
    let outcome = Outcome {
        args,
        model_bytes: model.len,
        report: report::check(model.bytes(), args.backends),
    };
    let text = if args.json {
        output::json(&outcome)
    } else {
        output::text(&outcome)
    };
    // A reader that closes the pipe early (`| head`) is not an error.
    if let Err(error) = std::io::stdout().write_all(text.as_bytes())
        && error.kind() != std::io::ErrorKind::BrokenPipe
    {
        eprintln!("nanoglia: {error}");
        return ExitCode::from(USAGE);
    }
    if outcome.passes() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(DOES_NOT_RUN)
    }
}

fn main() -> ExitCode {
    match args::parse(std::env::args_os().skip(1)) {
        Ok(Command::Check(args)) => check(&args),
        Ok(Command::Help) => {
            println!("{}", args::HELP);
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("nanoglia {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("nanoglia: {message}\n\nFor the usage: nanoglia --help");
            ExitCode::from(USAGE)
        }
    }
}
