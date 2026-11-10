//! Runs the `nanoglia` binary on models of `testdata/`: exit status, table and JSON.

// The tests start the binary as a process, which a WebAssembly runtime cannot do.
#![cfg(not(target_family = "wasm"))]

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const KWS: &str = "testdata/mlperf-tiny/kws_ref_model.tflite";

fn repo(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

/// Runs `nanoglia` with `args` and returns its exit status, standard output and standard error.
fn nanoglia(args: &[&str]) -> (i32, String, String) {
    let Output {
        status,
        stdout,
        stderr,
    } = Command::new(env!("CARGO_BIN_EXE_nanoglia"))
        .args(args)
        .output()
        .unwrap();
    (
        status.code().unwrap(),
        String::from_utf8(stdout).unwrap(),
        String::from_utf8(stderr).unwrap(),
    )
}

/// Runs `nanoglia check --json` on a file of the repository with `options`.
fn check_json(model: &str, options: &[&str]) -> (i32, Value) {
    let model = repo(model);
    let mut args = vec!["check", model.to_str().unwrap(), "--json"];
    args.extend_from_slice(options);
    let (status, stdout, stderr) = nanoglia(&args);
    assert_eq!(stderr, "");
    let json = serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("{e}:\n{stdout}"));
    (status, json)
}

#[test]
fn a_supported_model_passes() {
    let model = repo(KWS);
    let (status, stdout, stderr) = nanoglia(&["check", model.to_str().unwrap()]);
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert!(stdout.contains("checked with: backends cmsis-nn, kernels; features "));
    assert!(stdout.contains(" 0  CONV_2D v3 "));
    assert!(stdout.contains("int8[1,49,10,1] +2 const  int8[1,25,5,64]  cmsis-nn/conv2d_s8\n"));
    assert!(stdout.contains("\n12  SOFTMAX v2 "));
    assert!(stdout.contains("\nRUNS: 13 of 13 operators have a kernel.\n"));
    assert!(stdout.contains("  graph block "));
    assert!(!stdout.contains("RAM budget"));

    let (status, json) = check_json(KWS, &[]);
    assert_eq!(status, 0);
    assert_eq!(json["model"]["bytes"], 53936);
    assert_eq!(json["backends"], serde_json::json!(["cmsis-nn", "kernels"]));
    assert_eq!(json["features"]["float"], cfg!(feature = "float"));
    assert_eq!(json["features"]["int16"], cfg!(feature = "int16"));
    assert_eq!(json["features"]["all-ops"], cfg!(feature = "all-ops"));
    assert_eq!(
        json["features"]["single-rounding"],
        cfg!(feature = "single-rounding")
    );
    assert_eq!(json["pointer_bits"], usize::BITS);
    assert_eq!(json["passes"], true);
    assert_eq!(json["runs"], true);
    assert_eq!(json["unsupported"], 0);
    assert_eq!(json["error"], Value::Null);
    assert_eq!(json["ram_budget_bytes"], Value::Null);
    assert_eq!(json["fits"], Value::Null);

    let operators = json["operators"].as_array().unwrap();
    assert_eq!(operators.len(), 13);
    assert_eq!(
        operators[0],
        serde_json::json!({
            "index": 0,
            "opcode": 3,
            "name": "CONV_2D",
            "custom": false,
            "version": 3,
            "inputs": [
                {"tensor": operators[0]["inputs"][0]["tensor"], "type": "int8", "shape": [1, 49, 10, 1], "constant": false},
                {"tensor": operators[0]["inputs"][1]["tensor"], "type": "int8", "shape": [64, 10, 4, 1], "constant": true},
                {"tensor": operators[0]["inputs"][2]["tensor"], "type": "int32", "shape": [64], "constant": true},
            ],
            "outputs": [
                {"tensor": operators[0]["outputs"][0]["tensor"], "type": "int8", "shape": [1, 25, 5, 64], "constant": false},
            ],
            "kernel": "cmsis-nn/conv2d_s8",
            "backend": "cmsis-nn",
            "persistent_bytes": operators[0]["persistent_bytes"],
            "scratch_bytes": operators[0]["scratch_bytes"],
            "error": null,
        })
    );
    assert_eq!(operators[10]["kernel"], "rust/reshape");
    assert_eq!(operators[10]["backend"], "rust");

    let memory = &json["memory"];
    let bytes = |name: &str| memory[name].as_u64().unwrap();
    assert_eq!(
        bytes("context_bytes"),
        bytes("activation_bytes") + bytes("scratch_bytes")
    );
    assert_eq!(
        bytes("arena_bytes"),
        15 + bytes("graph_bytes") + bytes("context_bytes")
    );
    let largest_scratch = operators
        .iter()
        .map(|op| op["scratch_bytes"].as_u64().unwrap())
        .max()
        .unwrap();
    assert_eq!(bytes("scratch_bytes"), largest_scratch.next_multiple_of(16));
    // The sizes `documentation/reference/memory.md` gives for a 64-bit host.
    if usize::BITS == 64 {
        assert_eq!(
            (bytes("graph_bytes"), bytes("context_bytes")),
            (7712, 16160)
        );
    }
}

#[test]
fn every_operator_without_a_kernel_is_listed() {
    // The Rust kernels alone leave the int8 convolutions, pooling, fully-connected and softmax
    // of keyword spotting without a kernel: 12 of its 13 operators.
    let model = repo(KWS);
    let (status, stdout, stderr) =
        nanoglia(&["check", "--backends", "kernels", model.to_str().unwrap()]);
    assert_eq!((status, stderr.as_str()), (1, ""));
    assert!(stdout.contains("checked with: backends kernels; "));
    assert_eq!(
        stdout
            .matches("  NONE: no kernel for these types, shapes and options\n")
            .count(),
        12
    );
    assert!(stdout.contains("rust/reshape\n"));
    assert!(stdout.contains(
        "\nDOES NOT RUN: 12 of 13 operators have no kernel (CONV_2D v3 x5, \
         DEPTHWISE_CONV_2D v3 x4, AVERAGE_POOL_2D v2 x1, FULLY_CONNECTED v4 x1, SOFTMAX v2 x1).\n"
    ));
    assert!(stdout.contains("/documentation/reference/supported-operators.md\n"));
    assert!(!stdout.contains("memory (bytes)"));

    // CMSIS-NN alone lacks RESHAPE.
    let (status, stdout, _) =
        nanoglia(&["check", "--backends", "cmsis-nn", model.to_str().unwrap()]);
    assert_eq!(status, 1);
    assert!(stdout.contains("\nDOES NOT RUN: 1 of 13 operators has no kernel (RESHAPE v1 x1).\n"));

    let (status, json) = check_json(KWS, &["--backends=kernels"]);
    assert_eq!(status, 1);
    assert_eq!(json["passes"], false);
    assert_eq!(json["runs"], false);
    assert_eq!(json["unsupported"], 12);
    assert_eq!(json["memory"], Value::Null);
    // The error of the model is the one of its operators.
    assert_eq!(json["error"], Value::Null);
    let operators = json["operators"].as_array().unwrap();
    let blocking: Vec<u64> = operators
        .iter()
        .filter(|op| !op["error"].is_null())
        .map(|op| op["index"].as_u64().unwrap())
        .collect();
    assert_eq!(blocking, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]);
    let softmax = &operators[12];
    assert_eq!(softmax["name"], "SOFTMAX");
    assert_eq!(softmax["kernel"], Value::Null);
    assert_eq!(softmax["backend"], Value::Null);
    assert_eq!(softmax["persistent_bytes"], Value::Null);
    assert_eq!(softmax["error"]["code"], "unsupported-operation");
    assert!(
        softmax["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no selected backend")
    );
    assert_eq!(softmax["inputs"][0]["shape"], serde_json::json!([1, 12]));

    let (status, json) = check_json(KWS, &["--backends", "cmsis-nn"]);
    assert_eq!(status, 1);
    assert_eq!(json["unsupported"], 1);
    assert_eq!(json["operators"][10]["name"], "RESHAPE");
    assert!(!json["operators"][10]["error"].is_null());
}

#[test]
fn the_backends_and_features_change_the_kernels() {
    // ESP-NN first, as feature `esp-nn` of nanoglia registers it; it requantizes with double
    // rounding only.
    let (status, json) = check_json(KWS, &["--backends", "kernels,cmsis-nn,esp-nn"]);
    assert_eq!(status, 0);
    assert_eq!(
        json["backends"],
        serde_json::json!(["esp-nn", "cmsis-nn", "kernels"])
    );
    let backend = if cfg!(feature = "single-rounding") {
        "cmsis-nn"
    } else {
        "esp-nn"
    };
    assert_eq!(json["operators"][0]["backend"], backend);
    assert_eq!(json["operators"][10]["backend"], "rust");

    // The float32 and int16 kernels of convolutions are built with the features of the same
    // names (other operators, like ADD, run these types without them).
    for (model, feature) in [
        (
            "testdata/ops/conv2d_3x3_same_f32/model.tflite",
            cfg!(feature = "float"),
        ),
        (
            "testdata/ops/conv2d_3x3_same_int16/model.tflite",
            cfg!(feature = "int16"),
        ),
        ("testdata/ops/conv2d_3x3_same_int8/model.tflite", true),
        ("testdata/ops/add_f32/model.tflite", true),
    ] {
        let (status, json) = check_json(model, &[]);
        assert_eq!(json["runs"], feature, "{model}");
        assert_eq!(status, i32::from(!feature), "{model}");
    }
}

#[test]
fn a_ram_budget_must_hold_the_graph_and_a_context() {
    let (status, json) = check_json(KWS, &["--ram", "256k"]);
    assert_eq!(status, 0);
    assert_eq!(json["ram_budget_bytes"], 262144);
    assert_eq!(json["fits"], true);
    assert_eq!(json["passes"], true);

    let (status, json) = check_json(KWS, &["--ram=16k"]);
    assert_eq!(status, 1);
    assert_eq!(json["runs"], true);
    assert_eq!(json["fits"], false);
    assert_eq!(json["passes"], false);
    assert!(json["memory"]["arena_bytes"].as_u64().unwrap() > 16384);

    let model = repo(KWS);
    let (status, stdout, _) = nanoglia(&["check", model.to_str().unwrap(), "--ram", "16k"]);
    assert_eq!(status, 1);
    assert!(stdout.contains("\nDOES NOT FIT: every operator has a kernel, but "));
    assert!(stdout.contains("  RAM budget             16384  "));
    assert!(stdout.contains(" bytes short\n"));
    let (status, stdout, _) = nanoglia(&["check", model.to_str().unwrap(), "--ram", "1M"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("\nRUNS: "));
    assert!(stdout.contains("  RAM budget           1048576  fits, "));

    // The budget of a model that does not run is not compared.
    let (status, json) = check_json(KWS, &["--ram", "1M", "--backends", "kernels"]);
    assert_eq!(status, 1);
    assert_eq!(json["fits"], Value::Null);
}

#[test]
fn what_is_not_a_model_does_not_run() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let not_a_model = dir.join("not-a-model.tflite");
    std::fs::write(&not_a_model, b"\"quoted\" text, not a model").unwrap();
    // A model cut short: the identifier is there, the tables are not.
    let truncated = dir.join("truncated.tflite");
    let bytes = std::fs::read(repo(KWS)).unwrap();
    std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();

    for (path, message) in [
        (&not_a_model, "not a TensorFlow Lite model"),
        (&truncated, "malformed model"),
    ] {
        let path = path.to_str().unwrap();
        let (status, stdout, stderr) = nanoglia(&["check", path]);
        assert_eq!((status, stderr.as_str()), (1, ""));
        assert!(stdout.contains("\nDOES NOT RUN: "), "{stdout}");
        assert!(!stdout.contains("operators"), "{stdout}");
        assert!(stdout.contains(message), "{stdout}");

        let (status, stdout, _) = nanoglia(&["check", path, "--json"]);
        assert_eq!(status, 1);
        let json: Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(json["model"]["path"], path);
        assert_eq!(json["runs"], false);
        assert_eq!(json["unsupported"], Value::Null);
        assert_eq!(json["operators"], serde_json::json!([]));
        assert_eq!(json["error"]["code"], "invalid-encoding");
        assert!(json["error"]["message"].as_str().unwrap().contains(message));
    }
}

#[test]
fn wrong_arguments_and_unreadable_files_exit_with_2() {
    let model = repo(KWS);
    let model = model.to_str().unwrap();
    for args in [
        &[][..],
        &["check"],
        &["check", "no-such-file.tflite"],
        &["verify", model],
        &["check", model, "--ram", "lots"],
        &["check", model, "--backends", "tflm"],
        &["check", model, "--unknown"],
        &["check", model, model],
    ] {
        let (status, stdout, stderr) = nanoglia(args);
        assert_eq!(status, 2, "{args:?}");
        assert_eq!(stdout, "", "{args:?}");
        assert!(stderr.starts_with("nanoglia: "), "{args:?}: {stderr}");
    }

    let (status, stdout, stderr) = nanoglia(&["--help"]);
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert!(stdout.contains("Usage: nanoglia check [OPTIONS] <MODEL.tflite>"));
    let (status, stdout, _) = nanoglia(&["--version"]);
    assert_eq!(status, 0);
    assert_eq!(stdout, format!("nanoglia {}\n", env!("CARGO_PKG_VERSION")));
}
