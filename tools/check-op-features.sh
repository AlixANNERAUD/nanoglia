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
kws=op-conv-2d,op-depthwise-conv-2d,op-average-pool-2d,op-reshape,op-fully-connected,op-softmax
for set in "$kws"; do
  echo "op-features: nanoglia with $set"
  cargo test --quiet -p nanoglia --no-default-features --features "cmsis-nn,kernels,$set" --lib --test operator_subset
  cargo test --quiet -p nanoglia --no-default-features --features "cmsis-nn,esp-nn,kernels,$set" --test operator_subset
  cargo check --quiet -p nanoglia-c --no-default-features --features "cmsis-nn,kernels,$set"
  # Bare metal (Cortex-M4F), without any global allocator.
  cargo build --quiet -p nanoglia --lib --target thumbv7em-none-eabihf --no-default-features --features "cmsis-nn,kernels,$set"
done
echo "op-features: ok"
