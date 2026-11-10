//! The two renderings of a check: a table for people, JSON for programs.

use crate::args::Check;
use crate::report::{Operator, Report, Slot, Tensor, code_name, explain};
use nanoglia::{Detail, Error, MemoryBreakdown};
use std::fmt::Write as _;

/// Everything a rendering shows.
pub struct Outcome<'a> {
    pub args: &'a Check,
    /// Size of the model file.
    pub model_bytes: usize,
    /// The report, or why the operators could not be walked.
    pub report: Result<Report, Error>,
}

impl Outcome<'_> {
    /// What loading allocates, if the model loads.
    pub fn memory(&self) -> Option<MemoryBreakdown> {
        self.report.as_ref().ok()?.memory.ok()
    }

    /// Whether every operator has a kernel and the model loads.
    pub fn runs(&self) -> bool {
        self.memory().is_some()
    }

    /// The RAM of the graph and one execution context in one arena
    /// (`MemoryRequirements::arena_bytes(1)`).
    pub fn arena_bytes(&self) -> Option<usize> {
        Some(self.memory()?.requirements().arena_bytes(1))
    }

    /// Whether the model fits the RAM budget; `None` without a budget or when it does not run.
    pub fn fits(&self) -> Option<bool> {
        Some(self.arena_bytes()? <= self.args.ram?)
    }

    /// Whether the check passes: the model runs, within the budget if there is one.
    pub fn passes(&self) -> bool {
        self.runs() && self.fits() != Some(false)
    }
}

/// The Cargo features this build checks with, besides the backends.
fn features() -> [(&'static str, bool); 3] {
    [
        ("float", cfg!(feature = "float")),
        ("int16", cfg!(feature = "int16")),
        ("single-rounding", cfg!(feature = "single-rounding")),
    ]
}

fn tensor_text(tensor: &Tensor) -> String {
    let shape: Vec<String> = tensor.shape.iter().map(i32::to_string).collect();
    format!("{}[{}]", tensor.dtype, shape.join(","))
}

/// The tensors an operator computes on: constants are only counted.
fn slots_text(slots: &[Slot]) -> String {
    let mut parts = Vec::new();
    let mut constants = 0;
    for slot in slots {
        match slot {
            Slot::Tensor(tensor) if tensor.constant => constants += 1,
            Slot::Tensor(tensor) => parts.push(tensor_text(tensor)),
            Slot::Omitted => {}
            Slot::Unknown(_) => parts.push("?".to_owned()),
        }
    }
    if constants > 0 {
        parts.push(format!("+{constants} const"));
    }
    parts.join(" ")
}

fn operator_text(op: &Operator) -> String {
    if op.custom {
        format!("custom {} v{}", op.name, op.version)
    } else {
        format!("{} v{}", op.name, op.version)
    }
}

/// Where the operators, types and restrictions of each backend are listed.
const SUPPORTED_OPERATORS: &str = "https://github.com/AlixANNERAUD/nanoglia/blob/main/documentation/reference/supported-operators.md";

/// Why an operator has no kernel, short enough for a table cell.
fn reason(error: Error) -> String {
    match error.detail {
        Detail::UnsupportedOp { .. } => "no kernel for these types, shapes and options".to_owned(),
        _ => explain(error),
    }
}

/// The names of the operators without a kernel, with how many times each blocks.
fn blocking(operators: &[Operator]) -> String {
    let mut names: Vec<(String, usize)> = Vec::new();
    for op in operators.iter().filter(|op| op.kernel.is_err()) {
        let name = operator_text(op);
        match names.iter_mut().find(|(known, _)| *known == name) {
            Some((_, count)) => *count += 1,
            None => names.push((name, 1)),
        }
    }
    let parts: Vec<String> = names
        .iter()
        .map(|(name, count)| format!("{name} x{count}"))
        .collect();
    parts.join(", ")
}

/// The table and the summary.
pub fn text(outcome: &Outcome<'_>) -> String {
    let mut out = String::new();
    let args = outcome.args;
    let enabled: Vec<&str> = features()
        .iter()
        .filter_map(|(name, on)| on.then_some(*name))
        .collect();
    let _ = writeln!(
        out,
        "model: {} ({} bytes)",
        args.model.display(),
        outcome.model_bytes
    );
    let _ = writeln!(
        out,
        "checked with: backends {}; features {}",
        args.backends.names().join(", "),
        if enabled.is_empty() {
            "none".to_owned()
        } else {
            enabled.join(", ")
        },
    );
    let _ = writeln!(out);

    let report = match &outcome.report {
        Ok(report) => report,
        Err(error) => {
            let _ = writeln!(out, "DOES NOT RUN: {}", explain(*error));
            return out;
        }
    };

    let header = ["#", "operator", "inputs", "outputs", "kernel"].map(str::to_owned);
    let rows: Vec<[String; 5]> = report
        .operators
        .iter()
        .map(|op| {
            [
                op.index.to_string(),
                operator_text(op),
                slots_text(&op.inputs),
                slots_text(&op.outputs),
                match &op.kernel {
                    Ok(kernel) => kernel.name.to_owned(),
                    Err(error) => format!("NONE: {}", reason(*error)),
                },
            ]
        })
        .collect();
    let mut widths = header.each_ref().map(|cell| cell.chars().count());
    for row in &rows {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.chars().count());
        }
    }
    for row in std::iter::once(&header).chain(&rows) {
        let _ = write!(out, "{:>index$}", row[0], index = widths[0]);
        for (cell, width) in row[1..4].iter().zip(&widths[1..4]) {
            let _ = write!(out, "  {cell:<width$}");
        }
        let _ = writeln!(out, "  {}", row[4]);
    }
    let _ = writeln!(out);

    let count = report.operators.len();
    match report.memory {
        Ok(memory) => {
            let arena = memory.requirements().arena_bytes(1);
            match (outcome.fits(), args.ram) {
                (Some(false), Some(ram)) => {
                    let _ = writeln!(
                        out,
                        "DOES NOT FIT: every operator has a kernel, but the graph and one \
                         execution context need {arena} bytes, {} more than the budget of {ram}.",
                        arena - ram
                    );
                }
                _ => {
                    let _ = writeln!(out, "RUNS: {count} of {count} operators have a kernel.");
                }
            }
            let _ = writeln!(out);
            let _ = writeln!(out, "memory (bytes):");
            let _ = writeln!(
                out,
                "  graph block        {:>9}  of which prepared kernel parameters {}",
                memory.graph_bytes, memory.persistent_bytes
            );
            let _ = writeln!(
                out,
                "  execution context  {:>9}  activations {} + scratch {}",
                memory.context_bytes, memory.activation_bytes, memory.scratch_bytes
            );
            let _ = writeln!(
                out,
                "  one arena for both {arena:>9}  MemoryRequirements::arena_bytes(1)"
            );
            if let (Some(ram), Some(fits)) = (args.ram, outcome.fits()) {
                let _ = writeln!(
                    out,
                    "  RAM budget         {ram:>9}  {}",
                    if fits {
                        format!("fits, {} bytes left", ram - arena)
                    } else {
                        format!("{} bytes short", arena - ram)
                    }
                );
            }
            let _ = writeln!(
                out,
                "  model              {:>9}  borrowed where it is stored, never copied",
                outcome.model_bytes
            );
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "Kernel selection does not depend on the target: a firmware built with these \
                 backends and\nfeatures selects these kernels.{}",
                if args.backends.esp_nn {
                    " Except that ESP-NN claims no operator on a target without\natomic \
                     compare-and-swap (Cortex-M0+, riscv32imc): the next backend runs it there."
                } else {
                    ""
                }
            );
            let _ = writeln!(
                out,
                "The sizes are those of this {}-bit build: on a 32-bit target the graph block is \
                 a little\nsmaller, and CMSIS-NN with DSP or Helium, or ESP-NN on an ESP32-S3, \
                 may ask for another\nscratch size.",
                usize::BITS
            );
        }
        Err(_) if report.unsupported > 0 => {
            let _ = writeln!(
                out,
                "DOES NOT RUN: {} of {count} operators {} no kernel ({}).",
                report.unsupported,
                if report.unsupported == 1 {
                    "has"
                } else {
                    "have"
                },
                blocking(&report.operators)
            );
            let _ = writeln!(
                out,
                "What each backend runs, with which types and restrictions:\n{SUPPORTED_OPERATORS}"
            );
        }
        Err(error) => {
            let _ = writeln!(out, "DOES NOT RUN: {}", explain(error));
        }
    }
    out
}

/// A JSON string.
fn string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn array(items: impl IntoIterator<Item = String>) -> String {
    let items: Vec<String> = items.into_iter().collect();
    format!("[{}]", items.join(", "))
}

fn option(value: Option<String>) -> String {
    value.unwrap_or_else(|| "null".to_owned())
}

fn error_json(error: Error) -> String {
    format!(
        "{{\"code\": {}, \"detail\": {}, \"message\": {}}}",
        string(code_name(error.code)),
        string(&format!("{:?}", error.detail)),
        string(&explain(error))
    )
}

fn slot_json(slot: &Slot) -> String {
    match slot {
        Slot::Tensor(tensor) => format!(
            "{{\"tensor\": {}, \"type\": {}, \"shape\": {}, \"constant\": {}}}",
            tensor.index,
            string(tensor.dtype),
            array(tensor.shape.iter().map(i32::to_string)),
            tensor.constant
        ),
        Slot::Omitted => "null".to_owned(),
        Slot::Unknown(error) => format!("{{\"error\": {}}}", error_json(*error)),
    }
}

fn operator_json(op: &Operator) -> String {
    let kernel = op.kernel.as_ref().ok();
    format!(
        "{{\"index\": {}, \"opcode\": {}, \"name\": {}, \"custom\": {}, \"version\": {}, \
         \"inputs\": {}, \"outputs\": {}, \"kernel\": {}, \"backend\": {}, \
         \"persistent_bytes\": {}, \"scratch_bytes\": {}, \"error\": {}}}",
        op.index,
        op.opcode,
        string(&op.name),
        op.custom,
        op.version,
        array(op.inputs.iter().map(slot_json)),
        array(op.outputs.iter().map(slot_json)),
        option(kernel.map(|kernel| string(kernel.name))),
        option(kernel.map(|kernel| string(kernel.backend))),
        option(kernel.map(|kernel| kernel.plan.persistent.to_string())),
        option(kernel.map(|kernel| kernel.plan.scratch.to_string())),
        option(op.kernel.err().map(error_json)),
    )
}

/// One JSON document, on several lines (one per operator).
pub fn json(outcome: &Outcome<'_>) -> String {
    let args = outcome.args;
    let report = outcome.report.as_ref().ok();
    // The error that keeps the model from loading, when it is not the one of an operator.
    let model_error = match &outcome.report {
        Err(error) => Some(*error),
        Ok(report) if report.unsupported == 0 => report.memory.err(),
        Ok(_) => None,
    };
    let memory = outcome.memory().map(|memory| {
        format!(
            "{{\"graph_bytes\": {}, \"context_bytes\": {}, \"persistent_bytes\": {}, \
             \"activation_bytes\": {}, \"scratch_bytes\": {}, \"arena_bytes\": {}}}",
            memory.graph_bytes,
            memory.context_bytes,
            memory.persistent_bytes,
            memory.activation_bytes,
            memory.scratch_bytes,
            memory.requirements().arena_bytes(1)
        )
    });
    let operators = report.map_or_else(Vec::new, |report| {
        report.operators.iter().map(operator_json).collect()
    });
    let mut out = String::from("{\n");
    let mut field = |name: &str, value: String, last: bool| {
        let _ = writeln!(out, "  \"{name}\": {value}{}", if last { "" } else { "," });
    };
    field(
        "model",
        format!(
            "{{\"path\": {}, \"bytes\": {}}}",
            string(&args.model.to_string_lossy()),
            outcome.model_bytes
        ),
        false,
    );
    field(
        "backends",
        array(args.backends.names().into_iter().map(string)),
        false,
    );
    field(
        "features",
        format!(
            "{{{}}}",
            features()
                .map(|(name, on)| format!("{}: {on}", string(name)))
                .join(", ")
        ),
        false,
    );
    field("pointer_bits", usize::BITS.to_string(), false);
    field("passes", outcome.passes().to_string(), false);
    field("runs", outcome.runs().to_string(), false);
    field(
        "unsupported",
        option(report.map(|report| report.unsupported.to_string())),
        false,
    );
    field("error", option(model_error.map(error_json)), false);
    field("memory", option(memory), false);
    field(
        "ram_budget_bytes",
        option(args.ram.map(|ram| ram.to_string())),
        false,
    );
    field(
        "fits",
        option(outcome.fits().map(|fits| fits.to_string())),
        false,
    );
    field(
        "operators",
        if operators.is_empty() {
            "[]".to_owned()
        } else {
            format!("[\n    {}\n  ]", operators.join(",\n    "))
        },
        true,
    );
    out.push_str("}\n");
    out
}
