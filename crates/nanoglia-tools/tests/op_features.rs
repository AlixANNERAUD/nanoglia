//! Checks the `op-*` features across the workspace, so that an operator cannot get a kernel
//! without its feature, nor a feature be forgotten in one crate:
//!
//! - `BuiltinOp::feature` (nanoglia-core) names the features, and each backend declares in its
//!   manifest exactly the features of the operators its dispatch table has;
//! - a backend claims an operator exactly when the feature is on, whatever the features the test
//!   is built with (`mise run op-features` builds it with each feature alone);
//! - `nanoglia`, `nanoglia-c` and this crate declare every feature and forward it to every
//!   backend that has the operator, and `nanoglia-esp-nn` to its CMSIS-NN fallback; `all-ops`
//!   lists them all and is a default feature everywhere.
//!
//! This crate hosts the test because it is the one that always builds the three backends and
//! declares every feature. To add an operator: add its arm, behind its feature, to the backend's
//! table, name the feature in `BuiltinOp::feature`, and declare it in the manifests this test
//! names when it fails.

use nanoglia::BuiltinOp;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Every `op-*` feature of this crate, and whether this build has it.
const FEATURES: &[(&str, bool)] = &[
    ("op-abs", cfg!(feature = "op-abs")),
    ("op-add", cfg!(feature = "op-add")),
    ("op-arg-max", cfg!(feature = "op-arg-max")),
    ("op-arg-min", cfg!(feature = "op-arg-min")),
    ("op-average-pool-2d", cfg!(feature = "op-average-pool-2d")),
    (
        "op-batch-to-space-nd",
        cfg!(feature = "op-batch-to-space-nd"),
    ),
    ("op-cast", cfg!(feature = "op-cast")),
    ("op-concatenation", cfg!(feature = "op-concatenation")),
    ("op-conv-2d", cfg!(feature = "op-conv-2d")),
    (
        "op-depthwise-conv-2d",
        cfg!(feature = "op-depthwise-conv-2d"),
    ),
    ("op-depth-to-space", cfg!(feature = "op-depth-to-space")),
    ("op-dequantize", cfg!(feature = "op-dequantize")),
    ("op-div", cfg!(feature = "op-div")),
    ("op-equal", cfg!(feature = "op-equal")),
    ("op-exp", cfg!(feature = "op-exp")),
    ("op-expand-dims", cfg!(feature = "op-expand-dims")),
    ("op-fully-connected", cfg!(feature = "op-fully-connected")),
    ("op-gather", cfg!(feature = "op-gather")),
    ("op-greater", cfg!(feature = "op-greater")),
    ("op-greater-equal", cfg!(feature = "op-greater-equal")),
    ("op-hard-swish", cfg!(feature = "op-hard-swish")),
    ("op-leaky-relu", cfg!(feature = "op-leaky-relu")),
    ("op-less", cfg!(feature = "op-less")),
    ("op-less-equal", cfg!(feature = "op-less-equal")),
    ("op-log", cfg!(feature = "op-log")),
    ("op-logistic", cfg!(feature = "op-logistic")),
    ("op-maximum", cfg!(feature = "op-maximum")),
    ("op-max-pool-2d", cfg!(feature = "op-max-pool-2d")),
    ("op-mean", cfg!(feature = "op-mean")),
    ("op-minimum", cfg!(feature = "op-minimum")),
    ("op-mul", cfg!(feature = "op-mul")),
    ("op-neg", cfg!(feature = "op-neg")),
    ("op-not-equal", cfg!(feature = "op-not-equal")),
    ("op-pack", cfg!(feature = "op-pack")),
    ("op-pad", cfg!(feature = "op-pad")),
    ("op-padv2", cfg!(feature = "op-padv2")),
    ("op-prelu", cfg!(feature = "op-prelu")),
    ("op-quantize", cfg!(feature = "op-quantize")),
    ("op-reduce-max", cfg!(feature = "op-reduce-max")),
    ("op-reduce-min", cfg!(feature = "op-reduce-min")),
    ("op-relu", cfg!(feature = "op-relu")),
    ("op-relu6", cfg!(feature = "op-relu6")),
    ("op-relu-n1-to-1", cfg!(feature = "op-relu-n1-to-1")),
    ("op-reshape", cfg!(feature = "op-reshape")),
    ("op-resize-bilinear", cfg!(feature = "op-resize-bilinear")),
    (
        "op-resize-nearest-neighbor",
        cfg!(feature = "op-resize-nearest-neighbor"),
    ),
    ("op-rsqrt", cfg!(feature = "op-rsqrt")),
    ("op-select-v2", cfg!(feature = "op-select-v2")),
    ("op-slice", cfg!(feature = "op-slice")),
    ("op-softmax", cfg!(feature = "op-softmax")),
    (
        "op-space-to-batch-nd",
        cfg!(feature = "op-space-to-batch-nd"),
    ),
    ("op-space-to-depth", cfg!(feature = "op-space-to-depth")),
    ("op-split", cfg!(feature = "op-split")),
    ("op-split-v", cfg!(feature = "op-split-v")),
    ("op-sqrt", cfg!(feature = "op-sqrt")),
    (
        "op-squared-difference",
        cfg!(feature = "op-squared-difference"),
    ),
    ("op-squeeze", cfg!(feature = "op-squeeze")),
    ("op-strided-slice", cfg!(feature = "op-strided-slice")),
    ("op-sub", cfg!(feature = "op-sub")),
    ("op-sum", cfg!(feature = "op-sum")),
    ("op-tanh", cfg!(feature = "op-tanh")),
    ("op-transpose", cfg!(feature = "op-transpose")),
    ("op-transpose-conv", cfg!(feature = "op-transpose-conv")),
    ("op-unpack", cfg!(feature = "op-unpack")),
];

/// The builtin operator codes of the schema (see `BuiltinOp::name`).
const CODES: std::ops::Range<i32> = 0..210;

type Features = BTreeMap<String, BTreeSet<String>>;

/// The `[features]` table of the crate `name`: each feature with what it enables.
fn manifest(name: &str) -> Features {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(name)
        .join("Cargo.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    // The lines of the table, without comments, as one string: `name = ["a", "b"]` entries, some
    // of them on several lines.
    let table: String = text
        .lines()
        .skip_while(|line| line.trim() != "[features]")
        .skip(1)
        .take_while(|line| !line.starts_with('['))
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect();
    let mut features = Features::new();
    for entry in table.split(']') {
        if entry.trim().is_empty() {
            continue;
        }
        let (feature, values) = entry
            .split_once('=')
            .unwrap_or_else(|| panic!("{name}: cannot read the feature entry `{entry}`"));
        let values = values.trim().strip_prefix('[').unwrap();
        let values = values
            .split(',')
            .map(|value| value.trim().trim_matches('"').to_string())
            .filter(|value| !value.is_empty())
            .collect();
        features.insert(feature.trim().to_string(), values);
    }
    features
}

/// The `op-*` features `features` declares.
fn operators(features: &Features) -> BTreeSet<String> {
    features
        .keys()
        .filter(|feature| feature.starts_with("op-"))
        .cloned()
        .collect()
}

fn set(values: &[String]) -> BTreeSet<String> {
    values.iter().cloned().collect()
}

/// The features `BuiltinOp::feature` names.
fn core_features() -> BTreeSet<String> {
    CODES
        .filter_map(|code| BuiltinOp(code).feature())
        .map(String::from)
        .collect()
}

/// Whether the dispatch table of a backend has an operator in this build.
type HasOperator = fn(BuiltinOp) -> bool;

/// The backends: crate name and dispatch table.
const BACKENDS: &[(&str, HasOperator)] = &[
    (
        "nanoglia-cmsis-nn",
        nanoglia_cmsis_nn::CmsisNn::has_operator,
    ),
    ("nanoglia-esp-nn", nanoglia_esp_nn::EspNn::has_operator),
    (
        "nanoglia-kernels",
        nanoglia_kernels::RustKernels::has_operator,
    ),
];

#[test]
fn this_crate_declares_the_features_core_names() {
    let listed: BTreeSet<String> = FEATURES.iter().map(|(name, _)| name.to_string()).collect();
    assert_eq!(listed.len(), FEATURES.len(), "duplicate in FEATURES");
    assert_eq!(
        listed,
        core_features(),
        "FEATURES of this test and BuiltinOp::feature differ"
    );
    assert_eq!(
        operators(&manifest("nanoglia-tools")),
        listed,
        "features of nanoglia-tools/Cargo.toml and FEATURES of this test differ"
    );
}

#[test]
fn a_backend_has_an_operator_exactly_with_its_feature() {
    let enabled: BTreeMap<&str, bool> = FEATURES.iter().copied().collect();
    let mut declared_somewhere = BTreeSet::new();
    for (name, has_operator) in BACKENDS {
        let features = manifest(name);
        let declared = operators(&features);
        for code in CODES {
            let op = BuiltinOp(code);
            let label = op.name().unwrap();
            let expected = match op.feature() {
                Some(feature) => declared.contains(feature) && enabled[feature],
                None => false,
            };
            assert_eq!(
                has_operator(op),
                expected,
                "{name}: {label}: the dispatch table, the `op-*` feature of {name}/Cargo.toml \
                 and BuiltinOp::feature ({:?}) disagree",
                op.feature()
            );
        }
        assert!(
            declared.is_subset(&core_features()),
            "{name}: a feature BuiltinOp::feature does not name"
        );
        assert_eq!(features["all-ops"], declared, "{name}: all-ops");
        assert!(features["default"].contains("all-ops"), "{name}: default");
        declared_somewhere.extend(declared);
    }
    assert_eq!(
        declared_somewhere,
        core_features(),
        "BuiltinOp::feature names a feature no backend declares"
    );
    assert!(!BuiltinOp(-1).feature().is_some_and(|f| enabled[f]));
}

#[test]
fn the_esp_nn_backend_turns_on_its_cmsis_nn_fallback() {
    let esp_nn = manifest("nanoglia-esp-nn");
    let cmsis_nn = operators(&manifest("nanoglia-cmsis-nn"));
    for feature in operators(&esp_nn) {
        assert!(cmsis_nn.contains(&feature), "{feature}: no CMSIS-NN kernel");
        assert_eq!(
            esp_nn[&feature],
            set(&[format!("nanoglia-cmsis-nn/{feature}")]),
            "nanoglia-esp-nn: {feature}"
        );
    }
}

#[test]
fn every_feature_is_forwarded_to_the_backends_that_have_the_operator() {
    let backends: Vec<(&str, BTreeSet<String>)> = BACKENDS
        .iter()
        .map(|(name, _)| (*name, operators(&manifest(name))))
        .collect();
    let all = core_features();
    // `nanoglia` has optional backends (`backend?/feature`), this crate always builds them.
    for (name, optional) in [("nanoglia", "?"), ("nanoglia-tools", "")] {
        let features = manifest(name);
        assert_eq!(operators(&features), all, "{name}: the `op-*` features");
        assert_eq!(features["all-ops"], all, "{name}: all-ops");
        assert!(features["default"].contains("all-ops"), "{name}: default");
        for feature in &all {
            let expected: BTreeSet<String> = backends
                .iter()
                .filter(|(_, declared)| declared.contains(feature))
                .map(|(backend, _)| format!("{backend}{optional}/{feature}"))
                .collect();
            assert_eq!(features[feature], expected, "{name}: {feature}");
        }
    }
    // The facade refuses to build with a backend and no operator: its guard names every feature.
    let guard = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../nanoglia/src/lib.rs"),
    )
    .unwrap();
    for feature in &all {
        assert!(
            guard.contains(&format!("feature = \"{feature}\"")),
            "nanoglia/src/lib.rs: {feature} is missing from the no-operator guard"
        );
    }
    // The C API forwards to `nanoglia`.
    let features = manifest("nanoglia-c");
    assert_eq!(operators(&features), all, "nanoglia-c: the `op-*` features");
    assert_eq!(
        features["all-ops"],
        set(&["nanoglia/all-ops".to_string()]),
        "nanoglia-c: all-ops"
    );
    assert!(
        features["default"].contains("all-ops"),
        "nanoglia-c: default"
    );
    for feature in &all {
        assert_eq!(
            features[feature],
            set(&[format!("nanoglia/{feature}")]),
            "nanoglia-c: {feature}"
        );
    }
}
