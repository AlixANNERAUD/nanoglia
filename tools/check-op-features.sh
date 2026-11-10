#!/bin/sh
# Builds and tests the `op-*` features one at a time (`mise run op-features`).
#
# Every operator has a Cargo feature, all on by default; a firmware that turns the defaults off
# picks those of its models. Each feature must then work alone, in every crate: this script
# builds the three backends with no operator, then with each feature alone, and checks each time
# that the backends claim exactly the operators of the features that are on
# (crates/nanoglia-tools/tests/op_features.rs; nanoglia-tools is the crate that always builds the
# three backends and forwards every feature to them). Warnings are errors (RUSTFLAGS).
#
# One feature at a time, not their combinations: features only add match arms and modules, so two
# of them cannot conflict. One Cargo process at a time, in one target directory.
set -eu

cd "$(dirname "$0")/.."

# The features, from the manifest of the facade (the test checks that list against the backends).
features=$(sed -n 's/^\(op-[a-z0-9-]*\) = .*/\1/p' crates/nanoglia/Cargo.toml)
count=$(echo "$features" | wc -l)
echo "op-features: $count features"

# int8 only (neither `float` nor `int16`): the build with the least code, the most likely to leave
# an import or a helper unused. Then with both, type-checked only.
for feature in "" $features; do
  echo "op-features: ${feature:-no operator}"
  cargo test --quiet -p nanoglia-tools --no-default-features --features "$feature" --test op_features
  cargo check --quiet -p nanoglia-tools --no-default-features --features "float,int16,$feature"
done

# The facade refuses to build with a backend and no operator, with a message naming `all-ops`.
echo "op-features: nanoglia without any operator must not build"
if output=$(cargo check --quiet -p nanoglia --no-default-features --features cmsis-nn,kernels 2>&1); then
  echo "op-features: nanoglia built with a backend and no operator" >&2
  exit 1
fi
echo "$output" | grep -q "all-ops" || { echo "$output" >&2; exit 1; }

# The facade and the C API forward the features: with those of the keyword-spotting model, which
# must then run and give LiteRT's bytes while another model is refused
# (crates/nanoglia/tests/operator_subset.rs).
# In the order `nanoglia check` prints them.
kws=op-average-pool-2d,op-conv-2d,op-depthwise-conv-2d,op-fully-connected,op-reshape,op-softmax
for set in "$kws"; do
  echo "op-features: nanoglia with $set"
  cargo test --quiet -p nanoglia --no-default-features --features "cmsis-nn,kernels,$set" --lib --test operator_subset
  cargo test --quiet -p nanoglia --no-default-features --features "cmsis-nn,esp-nn,kernels,$set" --test operator_subset
  cargo check --quiet -p nanoglia-c --no-default-features --features "cmsis-nn,kernels,$set"
  # Bare metal (Cortex-M4F), without any global allocator.
  cargo build --quiet -p nanoglia --lib --target thumbv7em-none-eabihf --no-default-features --features "cmsis-nn,kernels,$set"
done
# The tests of `nanoglia check` adapt to the features of the tool: run them int8 only (the f32
# and 16x8 models then do not run), as the default build runs them in the test job.
echo "op-features: nanoglia-tools int8 only"
cargo test --quiet -p nanoglia-tools --no-default-features --features all-ops --test check

# `nanoglia check` prints the features of a firmware that runs a model. The tool built with
# exactly those (the same registry as such a firmware) must select the same kernels as the full
# build, operator by operator; the features printed for the keyword-spotting model must be the
# set above, with which nanoglia loads it and gives LiteRT's bytes; and the tool built with that
# set must say which feature the ResNet model lacks. The outputs go to target/op-features.
work=target/op-features
mkdir -p "$work"
models="testdata/mlperf-tiny/kws_ref_model.tflite testdata/ops/conv2d_3x3_same_f32/model.tflite
  testdata/ops/conv2d_3x3_same_int16/model.tflite"
# The kernel of each operator, one per line, from the JSON document on the standard input (a kernel
# name has no comma; `null` without a kernel).
kernels_of() {
  grep -o '"kernel": [^,]*'
}
# The default build first, then one build per model: Cargo relinks the tool when the features
# change, so they are not interleaved.
i=0
for model in $models; do
  i=$((i + 1))
  cargo run --quiet -p nanoglia-tools -- check "$model" \
    | sed -n 's/^nanoglia = .*, features = \[\(.*\)\] }$/\1/p' | tr -d '" ' > "$work/$i.features"
  cargo run --quiet -p nanoglia-tools -- check "$model" --json | kernels_of > "$work/$i.kernels"
done
i=0
for model in $models; do
  i=$((i + 1))
  printed=$(cat "$work/$i.features")
  echo "op-features: $model needs $printed"
  [ -n "$printed" ] || { echo "op-features: no features printed for $model" >&2; exit 1; }
  backends=$(echo "$printed" | tr , '\n' | grep -E '^(esp-nn|cmsis-nn|kernels)$' | paste -sd , -)
  features=$(echo "$printed" | tr , '\n' | grep -vE '^(esp-nn|cmsis-nn|kernels)$' | paste -sd , -)
  cargo run --quiet -p nanoglia-tools --no-default-features --features "$features" -- \
    check "$model" --backends "$backends" --json | kernels_of > "$work/$i.minimal-kernels"
  if ! cmp -s "$work/$i.kernels" "$work/$i.minimal-kernels"; then
    echo "op-features: with $printed, $model gets other kernels:" >&2
    diff "$work/$i.kernels" "$work/$i.minimal-kernels" >&2
    exit 1
  fi
done
printed=$(cat "$work/1.features")
if [ "$printed" != "cmsis-nn,kernels,$kws" ]; then
  echo "op-features: nanoglia check printed $printed for keyword spotting, not cmsis-nn,kernels,$kws" >&2
  exit 1
fi
if output=$(cargo run --quiet -p nanoglia-tools --no-default-features --features "$kws" -- \
    check testdata/mlperf-tiny/pretrainedResnet_quant.tflite); then
  echo "op-features: the ResNet model ran without op-add" >&2
  exit 1
fi
echo "$output" | grep -q 'NONE: built without `op-add`' || { echo "$output" >&2; exit 1; }
echo "op-features: ok"
