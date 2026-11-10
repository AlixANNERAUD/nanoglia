//! The Cargo features of `nanoglia` a firmware needs to run the models checked: the backends of
//! the selected kernels, the `op-*` feature of each operator, and `float` / `int16` when a
//! selected kernel exists only with them.

use crate::report::Report;
use nanoglia::BuiltinOp;
use std::collections::BTreeSet;

/// The kernels the `float` feature of `nanoglia` adds (`nanoglia-kernels`, the
/// `#[cfg(feature = "float")]` entries of its table). Other kernels also run `f32` tensors
/// without it, the binary operators for instance (`rust/binary`): the feature is needed only when
/// one of these is selected. The tests of this module check this list against the sources.
pub const FLOAT_KERNELS: &[&str] = &[
    "rust/conv_2d_f32",
    "rust/depthwise_conv_2d_f32",
    "rust/fully_connected_f32",
    "rust/hard_swish_f32",
    "rust/leaky_relu_f32",
    "rust/logistic_f32",
    "rust/pool_f32",
    "rust/prelu_f32",
    "rust/relu_f32",
    "rust/resize_bilinear_f32",
    "rust/softmax_f32",
    "rust/tanh_f32",
    "rust/transpose_conv_f32",
    "rust/unary_f32",
];

/// The kernels the `int16` feature of `nanoglia` adds: those of `nanoglia-kernels` behind
/// `#[cfg(feature = "int16")]`, and the 16x8 kernels of CMSIS-NN, which it refuses to select
/// without the feature. Other kernels also run `int16` tensors without it (`rust/reshape`,
/// `rust/binary_q`...). The tests of this module check this list against the sources.
pub const INT16_KERNELS: &[&str] = &[
    "cmsis-nn/add_s16",
    "cmsis-nn/average_pool2d_s16",
    "cmsis-nn/conv2d_s16",
    "cmsis-nn/depthwise_conv2d_s16",
    "cmsis-nn/fully_connected_s16",
    "cmsis-nn/max_pool2d_s16",
    "cmsis-nn/mul_s16",
    "rust/fully_connected_16x8",
    "rust/logistic_i16",
    "rust/softmax_i16",
    "rust/tanh_i16",
];

/// The backend features, in the order the registry tries them, with the name each backend
/// reports (`SelectedKernel::backend`).
const BACKENDS: [(&str, &str); 3] = [
    ("esp-nn", "esp-nn"),
    ("cmsis-nn", "cmsis-nn"),
    ("rust", "kernels"),
];

/// The features of a firmware that selects the kernels a check selected.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Firmware {
    /// Backend features, as names of [`BACKENDS`].
    backends: BTreeSet<&'static str>,
    float: bool,
    int16: bool,
    /// The `op-*` features.
    operators: BTreeSet<&'static str>,
}

impl Firmware {
    /// What a firmware needs for the model of `report`, `None` if the model does not run.
    pub fn of(report: &Report) -> Option<Self> {
        report.memory.ok()?;
        let mut firmware = Self::default();
        for op in &report.operators {
            let kernel = op.kernel.as_ref().ok()?;
            let (_, feature) = BACKENDS.iter().find(|(name, _)| *name == kernel.backend)?;
            firmware.backends.insert(feature);
            firmware.float |= FLOAT_KERNELS.contains(&kernel.name);
            firmware.int16 |= INT16_KERNELS.contains(&kernel.name);
            if !op.custom {
                // Every operator a backend runs has a feature.
                firmware.operators.extend(BuiltinOp(op.opcode).feature());
            }
        }
        Some(firmware)
    }

    /// Adds what `other` needs: a firmware that runs both models.
    pub fn add(&mut self, other: &Self) {
        self.backends.extend(&other.backends);
        self.float |= other.float;
        self.int16 |= other.int16;
        self.operators.extend(&other.operators);
    }

    /// The features: backends in the registry's order, `float`, `int16`, `single-rounding` when
    /// this tool was built with it (the kernels then differ), then the operators in alphabetical
    /// order.
    pub fn features(&self) -> Vec<&'static str> {
        let mut features: Vec<&'static str> = BACKENDS
            .iter()
            .map(|(_, feature)| *feature)
            .filter(|feature| self.backends.contains(feature))
            .collect();
        features.extend(self.float.then_some("float"));
        features.extend(self.int16.then_some("int16"));
        features.extend(cfg!(feature = "single-rounding").then_some("single-rounding"));
        features.extend(&self.operators);
        features
    }

    /// The dependency line of the firmware's `Cargo.toml`.
    pub fn dependency(&self) -> String {
        let features: Vec<String> = self
            .features()
            .iter()
            .map(|feature| format!("\"{feature}\""))
            .collect();
        format!(
            "nanoglia = {{ version = \"{}\", default-features = false, features = [{}] }}",
            nanoglia::VERSION,
            features.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn source(path: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(path);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// The `.rs` files of a directory of the workspace, tests left out.
    fn sources(directory: &str) -> Vec<(PathBuf, String)> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(directory);
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&root).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|extension| extension == "rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                files.push((path, text));
            }
        }
        files
    }

    /// The name of the kernel `module::STATIC` of `nanoglia-kernels`: the first `"rust/..."`
    /// literal after its declaration, by `static` or by one of the macros that declare kernels.
    fn rust_kernel_name(path: &str) -> String {
        let (module, name) = path.split_once("::").unwrap();
        let text = source(&format!("nanoglia-kernels/src/{module}.rs"));
        let declared = text
            .find(&format!("static {name}:"))
            .or_else(|| text.find(&format!("\n    {name},\n")))
            .unwrap_or_else(|| panic!("{path} is not declared"));
        let rest = &text[declared..];
        let start = rest.find("\"rust/").unwrap() + 1;
        let end = start + rest[start..].find('"').unwrap();
        rest[start..end].to_owned()
    }

    /// The kernels of the table of `nanoglia-kernels` behind `#[cfg(feature = "<feature>")]`.
    fn gated_rust_kernels(feature: &str) -> BTreeSet<String> {
        let table = source("nanoglia-kernels/src/lib.rs");
        let attribute = format!("#[cfg(feature = \"{feature}\")]");
        let mut lines = table.lines().map(str::trim);
        let mut kernels = BTreeSet::new();
        while let Some(line) = lines.next() {
            if line == attribute {
                let path = lines.next().unwrap().trim_end_matches(',');
                kernels.insert(rust_kernel_name(path));
            }
        }
        kernels
    }

    #[test]
    fn the_float_and_int16_kernels_are_those_of_the_sources() {
        let float: BTreeSet<String> = FLOAT_KERNELS.iter().map(|name| name.to_string()).collect();
        assert_eq!(float, gated_rust_kernels("float"));

        // CMSIS-NN's int16 kernels are its `_s16` ones, which it selects only with `INT16`.
        let mut int16 = gated_rust_kernels("int16");
        for (_, text) in sources("nanoglia-cmsis-nn/src") {
            for literal in text.split('"').filter(|s| s.starts_with("cmsis-nn/")) {
                if literal.ends_with("_s16") {
                    int16.insert(literal.to_owned());
                }
            }
        }
        let listed: BTreeSet<String> = INT16_KERNELS.iter().map(|name| name.to_string()).collect();
        assert_eq!(listed, int16);

        // The features gate kernels nowhere else: in `nanoglia-kernels` the table only (the
        // whole crate `allow`s the code they leave unused), in CMSIS-NN the `INT16` constant;
        // ESP-NN runs int8 only and has neither feature.
        for (directory, feature, allowed) in [
            ("nanoglia-kernels/src", "float", "lib.rs"),
            ("nanoglia-kernels/src", "int16", "lib.rs"),
            ("nanoglia-cmsis-nn/src", "int16", "lib.rs"),
            ("nanoglia-cmsis-nn/src", "float", ""),
            ("nanoglia-esp-nn/src", "int16", ""),
            ("nanoglia-esp-nn/src", "float", ""),
        ] {
            for (path, text) in sources(directory) {
                if text.contains(&format!("feature = \"{feature}\"")) {
                    assert!(
                        path.ends_with(allowed) && !allowed.is_empty(),
                        "{}",
                        path.display()
                    );
                }
            }
        }
        let cmsis = source("nanoglia-cmsis-nn/src/lib.rs");
        assert_eq!(cmsis.matches("feature = \"int16\"").count(), 1);
        for (_, text) in sources("nanoglia-esp-nn/src") {
            assert!(!text.contains("_s16\""));
        }
    }
}
