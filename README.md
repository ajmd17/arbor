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
| `arbor-cli` | Grows a tree from the terminal, prints stats, optionally writes an OBJ. |

Species are plain RON files in `assets/species`, compiled in as the built-in
presets. Presets saved from the viewer's panel (**Save as**) go to
`assets/species/custom/<name>.ron`, are listed under *Saved* in the preset menu, and
are read from disk each time they are picked, so a hand edit shows up without a
rebuild. Both tools find them by name: `--species <name>` in the viewer,
`arbor-cli <name>` headless. Textures live in `assets/textures`, named `<key>_albedo.png`,
`_normal.png` and `_roughness.png`, with the key coming from the species
(`bark_texture: "bark_oak"`, `leaves.texture: "leaf_oak"`).

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
