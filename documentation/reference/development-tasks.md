# Development tasks

The [mise](https://mise.jdx.dev) tasks of `mise.toml` and the GitHub
workflows that run them. `mise install` installs the tools; `mise run
<task>` runs a task; `mise tasks` lists them. CMSIS-NN and ESP-NN are git
submodules: run `mise run submodules` after cloning. Cargo goes through mbx
([mr-boxington](https://github.com/jdx/mr-boxington)), a shared build
cache that restores the compiled C objects of CMSIS-NN and the build
script outputs. Do not run several builds in the same target directory
at once.

## Tools and environment

| Tool | Version | Used by |
| --- | --- | --- |
| Rust | stable, with `rustfmt`, `clippy`, `llvm-tools` and the targets `thumbv6m-none-eabi`, `thumbv7em-none-eabihf`, `thumbv8m.main-none-eabihf`, `riscv32imc-unknown-none-elf`, `wasm32-unknown-unknown`, `wasm32-wasip1` | every task |
| mr-boxington | latest | every Cargo command |
| uv | latest | `oracle`, `bench-compare` |
| cargo-llvm-cov | latest | `coverage` |
| cargo-deny | latest (task tool) | `deny` |
| wasi-sdk | `wasi-sdk-33` (task tool) | `cross-wasm32`, `test-wasip1`, `bench-wasm` |
| wasmtime | 50.0.0 (task tool) | `test-wasip1`, `bench-wasm` |
| Rust nightly, cargo-fuzz | task tools | `fuzz` |
| Rust 1.89 | task tool | `msrv` |
| espup | 0.19.0 (task tool) | `install-esp` |
| cbindgen | 0.29.4 (task tool; Linux x86_64 and aarch64 binaries only) | `header`, `check-header` |

The C cross compilers come from Debian/Ubuntu packages declared in
`[bootstrap.packages]`: `gcc-arm-none-eabi`, `libnewlib-arm-none-eabi`,
`gcc-riscv64-unknown-elf`, `picolibc-riscv64-unknown-elf`. Install them
with `mise bootstrap packages apply`. `mise.toml` sets `CC_<target>`,
`AR_<target>` and `CFLAGS_<target>` for `thumbv6m-none-eabi`,
`thumbv7em-none-eabihf` and `riscv32imc-unknown-none-elf`; only the C
headers of newlib and picolibc are used, nothing links a C library.
`thumbv8m.main-none-eabihf` has a dot in its name, which the `CFLAGS_`
variable would not survive in mbx's build-script wrapper, so
`cross-arm` passes its compiler and flags through `TARGET_CC`,
`TARGET_AR` and `TARGET_CFLAGS` (from `ARM_NEWLIB_FLAGS`).

The C tasks (`test-c`, `cross-c`) also need CMake 3.21 or later and a
host C compiler, not managed by mise (the GitHub Ubuntu runners have
them): CMake runs Cargo with the mise environment it inherits.

## Tasks

| Task | Runs | CI |
| --- | --- | --- |
| `submodules` | `git submodule update --init --recursive`: checks out CMSIS-NN (`crates/nanoglia-cmsis-nn-sys/cmsis-nn`) and ESP-NN (`crates/nanoglia-esp-nn-sys/esp-nn`) at their pinned tags. Run it after cloning and after a change of a pinned commit; without it the build scripts of the `-sys` crates stop with this command in their message. | every workflow that builds checks out with `submodules: recursive` |
| `check` | `fmt`, `clippy`, `test`, `cross-arm`, one after the other. Run it before pushing. | — |
| `fmt` | `cargo fmt --all --check`. | `ci.yml` Format |
| `clippy` | `cargo clippy --workspace --all-targets`, then the same with `--all-features` (which builds the `single-rounding` mode), with `RUSTFLAGS=-D warnings`. | `ci.yml` Clippy |
| `test` | `cargo test --workspace` (unit tests, golden cases, memory tests, the C API called from Rust in `crates/nanoglia-c/tests/api.rs`), then `cargo test -p nanoglia --features esp-nn --test golden` (the golden cases with the ESP-NN backend first, its ANSI C kernels on the host), `RUSTFLAGS=-D warnings`. | `ci.yml` Test (default, ESP-NN); `release.yml` validate |
| `cross-arm` | `cargo build --workspace --lib` for `thumbv7em-none-eabihf` (Cortex-M4F, the Nano 33 BLE), `thumbv6m-none-eabi` (Cortex-M0+) and `thumbv8m.main-none-eabihf` (Cortex-M33), without a global allocator. | `ci.yml` Cross (Cortex-M0+, M4F and M33) |
| `cross-riscv32` | `cargo build --workspace --lib --target riscv32imc-unknown-none-elf` (ESP32-C3 class). | `ci.yml` Cross (riscv32imc) |
| `install-esp` | `espup install --targets esp32s3`: the Xtensa Rust toolchain (`esp`) and Espressif's GCC, with the environment script `$HOME/export-esp.sh`. | `ci.yml` Cross (ESP32-S3, ESP-NN optimized) |
| `cross-esp32s3` | Sources `$HOME/export-esp.sh` (or `ESP_EXPORT`), then, with the `esp` toolchain and `xtensa-esp32s3-elf-gcc`, `mbx build --workspace --lib` and `mbx build -p nanoglia --lib --features esp-nn` for `xtensa-esp32s3-none-elf` (`-Zbuild-std=core`, in `target/esp`), then `tools/check-no-malloc.sh target/esp`. | `ci.yml` Cross (ESP32-S3, ESP-NN optimized) |
| `cross-wasm32` | `cargo build --workspace --lib --target wasm32-unknown-unknown`, scalar and with SIMD128 (`-C target-feature=+simd128`), CMSIS-NN compiled by wasi-sdk's `clang` against the wasi-sdk sysroot headers. | `ci.yml` Cross (wasm32-unknown-unknown) |
| `test-wasip1` | `cargo test --workspace --target wasm32-wasip1 --release` under wasmtime, with the repository directory mapped so the golden tests find `testdata/`; then the golden and memory tests again with SIMD128, which must give the same bytes. | `ci.yml` Cross (wasm32-wasip1 tests under wasmtime) |
| `no-malloc` | `cross-arm`, `cross-riscv32`, `cross-c`, then `tools/check-no-malloc.sh`: no C library built for these targets, nor the C API archive `libnanoglia_c.a` (which bundles CMSIS-NN), may reference `malloc`, `calloc`, `realloc`, `free` or their variants, nor the Rust allocator (`__rust_alloc`...). | `ci.yml` Cross (no C heap) |
| `header` | `cbindgen --config cbindgen.toml --output include/nanoglia.h` in `crates/nanoglia-c`: regenerates the C header from `src/lib.rs`. Run it after changing the C API, and commit the header. | — |
| `check-header` | Generates the header into `target/c/nanoglia.h` and fails if it differs from the committed `crates/nanoglia-c/include/nanoglia.h`. | `ci.yml` C API (through `test-c`) |
| `test-c` | `check-header`, then `cmake -S crates/nanoglia-c -B target/c -DCMAKE_BUILD_TYPE=Release`, `cmake --build target/c` (Cargo builds `libnanoglia_c.a` for the host with `std`, in `target/c/cargo`), `ctest --test-dir target/c` (the C tests of `tests/test.c` and the keyword-spotting example on golden sample 0), then `cmake --install` into `target/c/install` and a build and run of the example against the installed package (`tests/package`, `find_package(nanoglia)`). The C compiler is CMake's (`CC`). | `ci.yml` C API (gcc, clang) |
| `cross-c` | Builds `libnanoglia_c.a` for `thumbv7em-none-eabihf` through the CMake project (`NANOGLIA_CARGO_TARGET`, no `std`, feature `panic-handler`) into `target/thumbv7em-none-eabihf/release`, links it with `arm-none-eabi-gcc -nostdlib` into a program calling every function (`tests/firmware.c`, never run) and prints its size, then builds the int8-only variant (`--no-default-features --features panic-handler,cmsis-nn,kernels`, in `target/c-thumbv7em-int8`). `RUSTFLAGS=-D warnings`. | `ci.yml` Cross (C library for Cortex-M4F) |
| `example-nano33ble` | `cargo build --release` in `examples/nano33ble`, then `arm-none-eabi-size` of the firmware. | `ci.yml` Cross (Nano 33 BLE firmware) |
| `bench` | `cargo run --release -p nanoglia --example bench`: median time of one inference of each MLPerf Tiny model on the host. The time budget per model is 2000 ms, or `NANOGLIA_BENCH_MS`. | — |
| `bench-wasm` | The same benchmark for `wasm32-wasip1` under wasmtime, once scalar and once with SIMD128 (`-C target-feature=+simd128`; the build scripts add `-msimd128` to the C code), each in its own target directory. | — |
| `bench-compare` | `uv run --project tools python -I tools/compare.py` in `benchmarks/compare` (its own Cargo workspace and `Cargo.lock`, and its own uv project in `tools/`): builds one host benchmark per engine (nanoglia, tract, MicroFlow, embedded-nn, ember-rs) and runs each 3 times in turn (`ROUNDS`, time budget per model `BENCH_MS`, default 2000 ms), checking every output against the golden samples; then builds one keyword-spotting firmware per engine for `thumbv7em-none-eabihf` (profile `firmware`), reads its size with `arm-none-eabi-size` and runs it under a Cortex-M4 emulator (Unicorn) to check its output and measure its peak stack. Prints Markdown tables; raw results in `benchmarks/compare/target/compare/`. Downloads the other engines from crates.io and GitHub. See [performance](performance.md#compared-with-other-rust-engines). | — |
| `oracle` | `uv run python -I generate.py` in `tools/oracle`: regenerates `testdata/ops/` and `testdata/mlperf-tiny/golden/` with LiteRT's reference kernels. Deterministic. | — |
| `coverage` | `cargo llvm-cov --workspace --lcov --output-path coverage.lcov`, with mbx's build-script wrapper turned off. | `coverage.yml` |
| `deny` | `cargo deny check` (licenses, advisories, bans, sources; `deny.toml`). | `ci.yml` Dependencies (cargo-deny) |
| `msrv` | With Rust 1.89: `cargo check --workspace --all-targets` and `cargo check --workspace --lib --target thumbv7em-none-eabihf`, in `target/msrv`. | `ci.yml` MSRV |
| `doc` | `cargo doc --no-deps --workspace` with `DOCS_RS=1` and `RUSTDOCFLAGS=-D warnings`. | `ci.yml` Docs (docs.rs mode) |
| `package` | Removes `target/package` and the nanoglia sources Cargo extracted into `$CARGO_HOME/registry/src` on an earlier run, then `cargo package --workspace` in `target/package` (mbx's build-script cache off): packages every crate as crates.io would and builds each one from its package, against the current sources. | `release.yml` validate |
| `fuzz` | `cargo fuzz run parse` for 60 seconds (inputs up to 400000 bytes) in `fuzz/`, seeded with the MLPerf Tiny models. | `ci.yml` Fuzz the model reader (1 min) |

## CI jobs without a task

`ci.yml` Test runs three entries with `mise exec -- cargo test`:
"default" (`--workspace`), "int8 only"
(`-p nanoglia --no-default-features --features cmsis-nn,kernels --lib --test memory`,
the configuration of the Nano 33 BLE firmware) and "ESP-NN"
(`-p nanoglia --features esp-nn --test golden`).

## Workflows

| Workflow | Trigger | Jobs |
| --- | --- | --- |
| `ci.yml` | push to `main`, pull requests, manual | Format, Clippy, Test (default, int8 only, ESP-NN), C API (`test-c` with `CC=gcc` and `CC=clang`), Cross (seven entries above), Cross (ESP32-S3, ESP-NN optimized: `install-esp` then `cross-esp32s3`), Dependencies, MSRV, Docs, Fuzz. `RUSTFLAGS=-D warnings` for all. |
| `coverage.yml` | push to `main`, pull requests | `mise run coverage`, uploaded to Codecov. |
| `pr-title.yml` | pull requests | Checks that the title follows Conventional Commits (`feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `ci`, `build`, `conf`, `chore`, `style`, `revert`, optional scope, optional `!`). release-plz builds the changelogs from these titles. |
| `release.yml` | push to `main` | validate (`mise run test`, `mise run package`), then release-plz: opens or updates a release pull request with versions and changelogs (`release-plz.toml`), and publishes to crates.io when it is merged. |

Every job that compiles the CMSIS-NN or ESP-NN sources, and the release
jobs (which package and publish the `-sys` crates), checks out with
`submodules: recursive`; Format, Dependencies and Fuzz do not need them.
Dependabot (`.github/dependabot.yml`) opens weekly pull requests for the
Cargo, uv and GitHub Actions dependencies and for the two submodules. It
proposes the head of the upstream default branch: pin the new release tag
instead, following the bump procedure of the crate's README
([`nanoglia-cmsis-nn-sys`](../../crates/nanoglia-cmsis-nn-sys/README.md),
[`nanoglia-esp-nn-sys`](../../crates/nanoglia-esp-nn-sys/README.md)).
