#!/usr/bin/env bash
# Builds arbor-ffi for Hyperion and drops it into the ArborToolkit plugin, which links
# it rather than building Rust itself:
#
#   tools/hyperion-drop.sh ~/Dev/HyperionEngine
#
# Writes Source/Plugins/ArborToolkit/External/arbor/ there:
#   lib/arbor_ffi.lib     static library, MSVC ABI, dynamic CRT (/MD) as Hyperion builds
#   include/arbor.h
#   native-libs.txt       system libraries the static library needs linked beside it
#   textures/             the texture sets the built-in species name
#   VERSION               the arbor commit it was built from
#
# Needs the MSVC toolchain beside the default GNU one:
#   rustup toolchain install stable-x86_64-pc-windows-msvc
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <HyperionEngine dir>" >&2
    exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hyperion="$(cd "$1" && pwd)"
plugin="$hyperion/Source/Plugins/ArborToolkit"
out="$plugin/External/arbor"
target=x86_64-pc-windows-msvc
cd "$root"

if [ ! -d "$plugin" ]; then
    echo "no ArborToolkit plugin in $hyperion" >&2
    exit 1
fi

# `--print native-static-libs` is how rustc says what the library leans on. It goes
# to stderr, alongside cargo's own progress.
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
cargo +stable-$target rustc --release -p arbor-ffi --lib --crate-type staticlib --target $target \
    -- --print native-static-libs 2>&1 | tee "$log"
libs="$(sed -nE 's/.*native-static-libs: (.*)/\1/p' "$log" | tail -n 1)"
if [ -z "$libs" ]; then
    echo "rustc did not say which native libraries arbor_ffi needs" >&2
    exit 1
fi

rm -rf "$out"
mkdir -p "$out/lib" "$out/include" "$out/textures"
cp "target/$target/release/arbor_ffi.lib" "$out/lib/"
cp crates/arbor-ffi/include/arbor.h "$out/include/"
# One per line, without the CRT the engine picks for itself.
echo "$libs" | tr ' ' '\n' | grep -E '\.lib$' > "$out/native-libs.txt"
git rev-parse HEAD > "$out/VERSION"
if ! git diff --quiet HEAD -- crates; then
    echo "dirty" >> "$out/VERSION"
fi

# Only the sets the built-in species name: the rest of the folder is source art.
names=$(grep -ohE '(bark_)?texture: "[^"]+"' assets/species/*.ron | sed -E 's/.*"(.*)"/\1/' | sort -u)
for name in $names; do
    for map in albedo normal roughness translucency; do
        file="assets/textures/${name}_${map}.png"
        if [ -f "$file" ]; then
            cp "$file" "$out/textures/"
        fi
    done
done

echo "dropped into $out ($(du -sh "$out" | cut -f1)): links $(tr '\n' ' ' < "$out/native-libs.txt")"
