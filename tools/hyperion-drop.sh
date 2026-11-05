#!/usr/bin/env bash
# Builds arbor-ffi for Hyperion and drops it into the ArborToolkit plugin, which links
# it rather than building Rust itself:
#
#   tools/hyperion-drop.sh ~/Dev/HyperionEngine
#   tools/hyperion-drop.sh --out dist/arbor      (what the release workflow packs)
#
# Builds for the machine it runs on: Windows (MSVC ABI, dynamic CRT as Hyperion builds), macOS
# (for macOS 14, as Hyperion targets) or Linux. ARBOR_TARGET picks another Rust target.
#
# Writes Source/Plugins/ArborToolkit/External/arbor/ there, or the folder --out names:
#   lib/arbor_ffi.lib     the static library (lib/libarbor_ffi.a on macOS and Linux)
#   include/arbor.h
#   native-libs.txt       system libraries the static library needs linked beside it, one per
#                         line as CMake's target_link_libraries takes them
#   textures/             the texture sets the built-in species name, which the plugin's
#                         buildarbortextures commandlet turns into BC texture assets; the
#                         release package leaves them out
#   VERSION               the arbor commit it was built from
#
# On Windows it needs the MSVC toolchain, which a machine with the GNU one as default lacks:
#   rustup toolchain install stable-x86_64-pc-windows-msvc
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ $# -eq 2 ] && [ "$1" = "--out" ]; then
    mkdir -p "$2"
    out="$(cd "$2" && pwd)"
elif [ $# -eq 1 ]; then
    hyperion="$(cd "$1" && pwd)"
    plugin="$hyperion/Source/Plugins/ArborToolkit"
    if [ ! -d "$plugin" ]; then
        echo "no ArborToolkit plugin in $hyperion" >&2
        exit 1
    fi
    out="$plugin/External/arbor"
else
    echo "usage: $0 <HyperionEngine dir> | --out <dir>" >&2
    exit 2
fi
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN* | Windows_NT) host=windows ;;
    Darwin) host=macos ;;
    Linux) host=linux ;;
    *)
        echo "no drop for $(uname -s)" >&2
        exit 1
        ;;
esac
case "$(uname -m)" in
    arm64 | aarch64) arch=aarch64 ;;
    *) arch=x86_64 ;;
esac
case "$host" in
    windows) target="${ARBOR_TARGET:-x86_64-pc-windows-msvc}" ;;
    macos) target="${ARBOR_TARGET:-$arch-apple-darwin}" ;;
    linux) target="${ARBOR_TARGET:-$arch-unknown-linux-gnu}" ;;
esac
# the toolchain named for the target on Windows, where the default may be the GNU one
toolchain=()
if [ "$host" = windows ]; then
    toolchain=("+stable-$target")
fi
# the oldest macOS Hyperion supports, or the library's objects are newer than what links them
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
cd "$root"

# `--print native-static-libs` is how rustc says what the library leans on. It goes
# to stderr, alongside cargo's own progress.
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
cargo ${toolchain[@]+"${toolchain[@]}"} rustc --release -p arbor-ffi --lib --crate-type staticlib --target $target \
    -- --print native-static-libs 2>&1 | tee "$log"
libs="$(sed -nE 's/.*native-static-libs: (.*)/\1/p' "$log" | tail -n 1)"
if [ -z "$libs" ]; then
    echo "rustc did not say which native libraries arbor_ffi needs" >&2
    exit 1
fi

rm -rf "$out"
mkdir -p "$out/lib" "$out/include" "$out/textures"
if [ "$host" = windows ]; then
    cp "target/$target/release/arbor_ffi.lib" "$out/lib/"
    # without the CRT the engine picks for itself
    echo "$libs" | tr ' ' '\n' | grep -E '\.lib$' > "$out/native-libs.txt"
else
    cp "target/$target/release/libarbor_ffi.a" "$out/lib/"
    # -lfoo as foo, and -framework Foo as one item, which is how CMake takes them
    echo "$libs" | tr ' ' '\n' | awk '
        /^-l/ { print substr($0, 3); next }
        $0 == "-framework" { framework = 1; next }
        framework { print "-framework " $0; framework = 0; next }
    ' > "$out/native-libs.txt"
fi
cp crates/arbor-ffi/include/arbor.h "$out/include/"
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
