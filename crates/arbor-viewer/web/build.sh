#!/usr/bin/env bash
# Builds the viewer for the web into docs/: the page, the wasm, and the texture maps
# the built-in species draw with. GitHub Pages serves that folder straight from the
# branch; any other static host takes it as it is. To try it locally:
#
#   bash crates/arbor-viewer/web/build.sh
#   python -m http.server 8080 -d docs      # then open http://localhost:8080
#
# Needs the wasm32-unknown-unknown target, and wasm-bindgen-cli at the same version as
# the wasm-bindgen crate in Cargo.lock (`cargo tree -i wasm-bindgen --target
# wasm32-unknown-unknown -p arbor-viewer` says which):
#
#   rustup target add wasm32-unknown-unknown
#   cargo install --locked wasm-bindgen-cli --version <that version>
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
out="$root/docs"
cd "$root"

if ! command -v wasm-bindgen >/dev/null; then
    echo "wasm-bindgen is not installed; see the top of $0" >&2
    exit 1
fi

cargo build --release -p arbor-viewer --target wasm32-unknown-unknown
mkdir -p "$out"
wasm-bindgen --target web --no-typescript --out-dir "$out" --out-name arbor \
    target/wasm32-unknown-unknown/release/arbor-viewer.wasm
cp crates/arbor-viewer/web/index.html "$out/"
# Served as it is, not run through Jekyll first.
touch "$out/.nojekyll"

# The page fetches a species' maps from beside it as the species is picked. Only the
# sets the built-in species name are taken: the rest of the folder is source art.
mkdir -p "$out/assets/textures"
names=$(grep -ohE '(bark_)?texture: "[^"]+"' assets/species/*.ron | sed -E 's/.*"(.*)"/\1/' | sort -u)
for name in $names; do
    for map in albedo normal roughness; do
        file="assets/textures/${name}_${map}.png"
        if [ -f "$file" ]; then
            cp -u "$file" "$out/assets/textures/"
        fi
    done
done

echo "built $out ($(du -sh "$out" | cut -f1))"
