#!/bin/sh
# Checks that the C libraries compiled for the embedded targets never reference the C heap
# (malloc, calloc, realloc, free and their reentrant variants): nanoglia takes every byte from the
# allocator its caller passes. Also checks the static library of the C API (libnanoglia_c.a, from
# `mise run cross-c`), which must not need the C heap nor a Rust global allocator.
#
# Usage: tools/check-no-malloc.sh [target-dir]   (run after `mise run cross-arm`/`cross-riscv32`)
set -eu

target_dir=${1:-${CARGO_TARGET_DIR:-target}}
heap='^_*(malloc|calloc|realloc|free|aligned_alloc|memalign|posix_memalign|valloc|pvalloc)(_r)?$|^__rust_(alloc|dealloc|realloc|alloc_zeroed)$'

# The binutils of the target if installed, else the host `nm` (which reads any ELF file when
# built with all targets).
nm_for() {
    case "$1" in
    thumb* | arm*) candidates="arm-none-eabi-nm" ;;
    riscv*) candidates="riscv64-unknown-elf-nm riscv32-esp-elf-nm" ;;
    xtensa*) candidates="xtensa-esp-elf-nm xtensa-esp32s3-elf-nm xtensa-esp32-elf-nm" ;;
    *) candidates="" ;;
    esac
    for tool in $candidates llvm-nm nm; do
        if command -v "$tool" >/dev/null 2>&1; then
            echo "$tool"
            return
        fi
    done
    echo "error: no nm found for $1" >&2
    exit 2
}

checked=0
failed=0
# Static libraries built by the build scripts of the cross targets, e.g.
# target/thumbv7em-none-eabihf/debug/build/nanoglia-cmsis-nn-sys-<hash>/out/libcmsis_nn.a, and the
# C API, e.g. target/thumbv7em-none-eabihf/release/libnanoglia_c.a.
for lib in "$target_dir"/thumb*/*/build/*/out/*.a \
    "$target_dir"/riscv*/*/build/*/out/*.a \
    "$target_dir"/xtensa*/*/build/*/out/*.a \
    "$target_dir"/thumb*/*/libnanoglia_c.a \
    "$target_dir"/riscv*/*/libnanoglia_c.a \
    "$target_dir"/xtensa*/*/libnanoglia_c.a; do
    [ -f "$lib" ] || continue
    triple=$(echo "${lib#"$target_dir"/}" | cut -d/ -f1)
    nm=$(nm_for "$triple")
    undefined=$("$nm" -u "$lib" | awk '{ print $NF }' | grep -E "$heap" | sort -u || true)
    checked=$((checked + 1))
    if [ -n "$undefined" ]; then
        echo "FAIL $lib references the C heap:" >&2
        echo "$undefined" | sed 's/^/  /' >&2
        failed=$((failed + 1))
    else
        echo "ok   $lib"
    fi
done

if [ "$checked" -eq 0 ]; then
    echo "error: no C library found under $target_dir; run \`mise run cross-arm\` first" >&2
    exit 2
fi
if [ "$failed" -ne 0 ]; then
    echo "$failed of $checked libraries reference malloc/calloc/realloc/free or the Rust allocator" >&2
    exit 1
fi
echo "$checked libraries checked, none references the C heap"
