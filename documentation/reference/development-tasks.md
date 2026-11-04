# Development tasks

The [mise](https://mise.jdx.dev) tasks of `mise.toml` and the GitHub
workflows that run them. `mise install` installs the tools; `mise run
<task>` runs a task; `mise tasks` lists them. Cargo goes through mbx
([mr-boxington](https://github.com/jdx/mr-boxington)), a shared build
cache that restores the compiled C objects of CMSIS-NN and the build
script outputs. Do not run several builds in the same target directory
at once.

## Tools and environment

| Tool | Version | Used by |
| --- | --- | --- |
| Rust | stable, with `rustfmt`, `clippy`, `llvm-tools` and the targets `thumbv6m-none-eabi`, `thumbv7em-none-eabihf`, `thumbv8m.main-none-eabihf`, `riscv32imc-unknown-none-elf`, `wasm32-unknown-unknown`, `wasm32-wasip1` | every task |
| mr-boxington | latest | every Cargo command |
| uv | latest | `oracle` |
| cargo-llvm-cov | latest | `coverage` |
| cargo-deny | latest (task tool) | `deny` |
| wasi-sdk | `wasi-sdk-33` (task tool) | `cross-wasm32`, `test-wasip1`, `bench-wasm` |
| wasmtime | 50.0.0 (task tool) | `test-wasip1`, `bench-wasm` |
| Rust nightly, cargo-fuzz | task tools | `fuzz` |
| Rust 1.89 | task tool | `msrv` |
| espup | 0.19.0 (task tool) | `install-esp` |

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

## Tasks

| Task | Runs | CI |
| --- | --- | --- |
| `check` | `fmt`, `clippy`, `test`, `cross-arm`, one after the other. Run it before pushing. | — |
| `fmt` | `cargo fmt --all --check`. | `ci.yml` Format |
| `clippy` | `cargo clippy --workspace --all-targets`, then the same with `--all-features` (which builds the `single-rounding` mode), with `RUSTFLAGS=-D warnings`. | `ci.yml` Clippy |
| `test` | `cargo test --workspace` (unit tests, golden cases, memory tests), then `cargo test -p nanoglia --features esp-nn --test golden` (the golden cases with the ESP-NN backend first, its ANSI C kernels on the host), `RUSTFLAGS=-D warnings`. | `ci.yml` Test (default, ESP-NN); `release.yml` validate |
| `cross-arm` | `cargo build --workspace --lib` for `thumbv7em-none-eabihf` (Cortex-M4F, the Nano 33 BLE), `thumbv6m-none-eabi` (Cortex-M0+) and `thumbv8m.main-none-eabihf` (Cortex-M33), without a global allocator. | `ci.yml` Cross (Cortex-M0+, M4F and M33) |
| `cross-riscv32` | `cargo build --workspace --lib --target riscv32imc-unknown-none-elf` (ESP32-C3 class). | `ci.yml` Cross (riscv32imc) |
| `install-esp` | `espup install --targets esp32s3`: the Xtensa Rust toolchain (`esp`) and Espressif's GCC, with the environment script `$HOME/export-esp.sh`. | `ci.yml` Cross (ESP32-S3, ESP-NN optimized) |
| `cross-esp32s3` | Sources `$HOME/export-esp.sh` (or `ESP_EXPORT`), then, with the `esp` toolchain and `xtensa-esp32s3-elf-gcc`, `mbx build --workspace --lib` and `mbx build -p nanoglia --lib --features esp-nn` for `xtensa-esp32s3-none-elf` (`-Zbuild-std=core`, in `target/esp`), then `tools/check-no-malloc.sh target/esp`. | `ci.yml` Cross (ESP32-S3, ESP-NN optimized) |
| `cross-wasm32` | `cargo build --workspace --lib --target wasm32-unknown-unknown`, CMSIS-NN compiled by wasi-sdk's `clang` against the wasi-sdk sysroot headers. | `ci.yml` Cross (wasm32-unknown-unknown) |
| `test-wasip1` | `cargo test --workspace --target wasm32-wasip1 --release` under wasmtime, with the repository directory mapped so the golden tests find `testdata/`. | `ci.yml` Cross (wasm32-wasip1 tests under wasmtime) |
| `no-malloc` | `cross-arm`, `cross-riscv32`, then `tools/check-no-malloc.sh`: no C library built for these targets may reference `malloc`, `calloc`, `realloc`, `free` or their variants. | `ci.yml` Cross (no C heap) |
| `example-nano33ble` | `cargo build --release` in `examples/nano33ble`, then `arm-none-eabi-size` of the firmware. | `ci.yml` Cross (Nano 33 BLE firmware) |
| `bench` | `cargo run --release -p nanoglia --example bench`: median time of one inference of each MLPerf Tiny model on the host. The time budget per model is 2000 ms, or `NANOGLIA_BENCH_MS`. | — |
| `bench-wasm` | The same benchmark for `wasm32-wasip1` under wasmtime, once scalar and once with SIMD128 (`-C target-feature=+simd128`, `CFLAGS_wasm32_wasip1=-msimd128`), each in its own target directory. | — |
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
| `ci.yml` | push to `main`, pull requests, manual | Format, Clippy, Test (default, int8 only, ESP-NN), Cross (six entries above), Cross (ESP32-S3, ESP-NN optimized: `install-esp` then `cross-esp32s3`), Dependencies, MSRV, Docs, Fuzz. `RUSTFLAGS=-D warnings` for all. |
| `coverage.yml` | push to `main`, pull requests | `mise run coverage`, uploaded to Codecov. |
| `pr-title.yml` | pull requests | Checks that the title follows Conventional Commits (`feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `ci`, `build`, `conf`, `chore`, `style`, `revert`, optional scope, optional `!`). release-plz builds the changelogs from these titles. |
| `release.yml` | push to `main` | validate (`mise run test`, `mise run package`), then release-plz: opens or updates a release pull request with versions and changelogs (`release-plz.toml`), and publishes to crates.io when it is merged. |
