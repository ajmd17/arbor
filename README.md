# Arbor

Procedural tree model generator in Rust. Aim is to generate ~7k-20k tri models.

![screenshot](docs/screenshot.png)

Note, this application is almost entirely vibe-coded.

I didn't want to pay for SpeedTree.

## Layout

| Crate | What it is |
| --- | --- |
| `arbor-core` | The model: growth, meshing, foliage, cluster baking. No GL, no windowing. |
| `arbor-viewer` | An OpenGL viewer (eframe/egui + glow) with live parameter sliders. |
| `arbor-cli` | Grows a tree from the terminal, prints stats, optionally writes an OBJ or glTF. |

Species are plain RON files in `assets/species`, compiled in as the built-in
presets. Presets saved from the viewer's panel (**Save as**) go to
`assets/species/custom/<name>.ron`, are listed under *Saved* in the preset menu, and
are read from disk each time they are picked, so a hand edit shows up without a
rebuild. Both tools find them by name: `--species <name>` in the viewer,
`arbor-cli <name>` headless. Textures live in `assets/textures`, named `<key>_albedo.png`,
`_normal.png` and `_roughness.png`, with the key coming from the species
(`bark_texture: "bark_oak"`, `leaves.texture: "leaf_oak"`).

### Ranges

Any number in a species can be a range instead of a value, so a species describes a
kind of tree rather than one tree:

```ron
trunk: (
    length: (12.0, 18.0),
    ...
),
envelope_scale: (0.85, 1.15),
```

Each tree lands somewhere in each range, and its seed decides where: the same seed
always gives the same tree, and the next seed gives a different one. Each range is
drawn on its own, so adding a range to one number never moves where the others land.
This is not the same as `length_variance` and the other variances, which spread the
stems *within* one tree; a range is drawn once for the whole tree. A trunk-length
range also takes the crown envelope with it, which the trunk's `length_variance`
does not.

In the viewer, the **±** beside a slider turns it into a range with two handles, and
the second one reports where the tree on screen landed. Pressing **±** again folds the
range back to that value. Export **Variations** lands every range afresh for each
tree. The CLI prints each landing under the seed.

## Running it

The viewer reads textures from `assets/textures` relative to the working
directory, so run it from the repository root.

```bash
cargo run --release -p arbor-viewer
```

```bash
cargo run --release -p arbor-viewer
```

### Options:
`--species`, `--seed`, `--no-leaves`, `--wireframe`, `--no-shadows`,
`--translucency`, `--time`, `--sun-elevation`, `--sun-azimuth`,
`--sun-intensity`, `--yaw`, `--pitch`, `--distance`, `--target-y`,
`--coverage-lod`, `--wind <strength>`, `--gustiness`, `--wind-dir <degrees>`,
`--wind-time <seconds>`, and `--screenshot <path> [--settle N]`.

A screenshot is taken in still air unless `--wind` asks otherwise, and then on a
stopped clock (`--wind-time`, default 0), so the same command always gives the same
frame.

Headless, from the CLI:

```bash
cargo run --release -p arbor-cli -- oak --seed 7 --obj oak.obj
```

The OBJ comes out as triangles in two groups, `bark` and `leaves`, with leaf UVs
baked into their atlas cell (the viewer's shader picks a cell per card, an
exported mesh has no such shader).

### In a browser

The viewer also builds for the web, on WebGL2:

```bash
bash crates/arbor-viewer/web/build.sh
python -m http.server 8080 -d docs
```

Then open http://localhost:8080. `docs/` is the whole site: the page, the wasm, and
the texture maps the built-in species use (about 55 MB, most of it the spruce bark).
GitHub Pages serves it from the branch (**Settings → Pages → Deploy from a branch**,
folder `/docs`), and it works on any other static host.

The build needs the `wasm32-unknown-unknown` target and `wasm-bindgen-cli`. The CLI's
version has to match the `wasm-bindgen` crate in `Cargo.lock`, which
`cargo tree -p arbor-viewer --target wasm32-unknown-unknown -i wasm-bindgen` prints.

```bash
rustup target add wasm32-unknown-unknown
cargo install --locked wasm-bindgen-cli --version 0.2.126
```

On the web:

- Textures are fetched when a species is picked. Switching species holds the page
  for a moment while its leaf atlas is baked.
- **Download GLB** exports the tree on screen as a download. Batches, glTF and
  saving presets need the desktop viewer; **Copy as RON** works in both.
- There's no wireframe, because WebGL can't draw one.
- A seed grows the same tree as on the desktop. Only float noise differs, well under
  a millimetre.

## Exporting glTF

In the viewer, under **Save as**:

- **Export to** is the folder exports go to (`exports` by default, created if it's
  missing). Type a path or pick one with **Browse…**.
- **Variations** is how many trees to write: the one on screen, then the same species
  grown from each seed after its own.
- **Export GLB** / **Export glTF** write them, named as the preset would be saved:
  `<name>.glb` for one tree, `<name>_seed<N>.glb` for each of a batch, so any of them
  can be grown again from its seed. A batch shows its progress and can be stopped
  between trees.

From the CLI:

```bash
cargo run --release -p arbor-cli -- oak --seed 7 --glb oak.glb
```

```bash
cargo run --release -p arbor-cli -- oak --seed 7 --gltf out/oak.gltf --variations 10 --wind-data
```

The second writes `out/oak_seed7.gltf` through `out/oak_seed16.gltf`.

A `.glb` is one self-contained file. A `.gltf` is JSON, with its `.bin` and the
textures as `<name>_*.png` written beside it. A batch of `.gltf` files shares one set
of textures, since they're the species' rather than the tree's. Textures are
prepared once per export, however many trees it writes. Textures are read from
`assets/textures`; `--textures <dir>` reads them from somewhere else, and
`--no-textures` leaves them out.

The scene is a node named for the species with two children, `bark` and `leaves`,
in metres with Y up. The materials are standard metallic-roughness, set up to match
the viewer as closely as that allows:

- **Bark:** the species' bark maps, with its tint as the base colour factor and the
  darkening toward the foot of the trunk as a vertex colour (`COLOR_0`).
- **Dead wood:** its own material, with the viewer's bleaching baked into a copy of
  the bark albedo. Only a tree with dead wood the species bleaches has one.
- **Leaves:** the leaf art, or the cluster atlas baked from it, alpha-masked at the
  viewer's cutoff (0.35). Each card's tint and crown-depth shade go in `COLOR_0`.
  glTF can't choose a texture by which side of a card is seen, so a species whose
  leaves show a different cell from behind (the oak, the birch) gets a second,
  back-facing copy of every card. Species with one cell get single, double-sided
  cards.

Moss, light through the leaves, and the viewer's coverage-preserving leaf mipmaps
don't carry over. Expect distant canopies to thin a little in engines that build
ordinary mips.

`--wind-data` (the **Wind data** box in the viewer) adds what an engine needs to sway
the tree as the viewer does. The tree bends as a set of stems. A stem is a limb, a
branch, or all the twigs off one point of a branch, and each one bends about where it's
attached. Each stem is written once, in a table. Each vertex names the finest stem it
bends with in `TEXCOORD_1`:

| `TEXCOORD_1` | Holds |
| --- | --- |
| x | The stem's row in the table, as a float. Row 0 is the trunk, which bends by height alone. |
| y | How far out along that stem the vertex sits, 0 at its pivot to 1 at its reach. Every corner of a leaf card carries its origin's value. |

The rest goes in the `ARBOR_tree_wind` extension on every primitive:

| Key | Holds |
| --- | --- |
| `branches` | Accessor, `VEC4`, two per row: the pivot (xyz) and reach in metres (w), then the carrying stem's row, how far out along it this stem leaves (0–1), and the order (0 limb, 1 branch, 2 twigs and finer). Row 0 is all zeros. |
| `leafOrigins` | Leaves only. Accessor, `VEC3` per vertex: the twig point a card hangs from, which it flutters about. |
| `flexibility` | How far the trunk, limbs, branches and twigs bend in a full gale. |
| `frequency` | How fast the trunk sways, in hertz. |
| `flutter` | How far a leaf card flutters, in radians. |
| `height` | The tree's height in metres. The trunk's bend is shaped over it, from the foot at the origin. |

To sway a vertex, start at its row with its `t`. The stem there swings the vertex about
its pivot by `reach × cantilever(t) × flexibility[order + 1]` metres, where
`cantilever(x) = x²(6 − 4x + x²)/3`. Then move to the carrying stem's row and its
`t`, and repeat until row 0.

## Tools

```bash
cargo run --release -p arbor-core   --example morphology -- oak   # stem lengths, turn, children per level
cargo run --release -p arbor-core   --example budget     -- oak   # where the triangles go
cargo run --release -p arbor-core   --example levers     -- oak   # what each saving is actually worth
cargo run --release -p arbor-viewer --example preview    -- oak out.png
cargo run --release -p arbor-viewer --example bake_cluster -- leaf_oak
cargo run --release -p arbor-viewer --example import_textures -- leaf_oak --albedo src.png --alpha mask.png
cargo run --release -p arbor-viewer --example alpha_coverage_report -- assets/textures/leaf_oak_albedo.png
```

```bash
cargo test
```
