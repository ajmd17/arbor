//! The presets the panel offers: the species compiled into the binary, and the ones
//! saved from the panel itself.
//!
//! A saved preset is a species file like any other, written in full — every value and
//! every range, the seed included — so picking it again grows exactly the tree that was
//! saved, and the species it came from at any other seed. It
//! is read from disk every time it is picked rather than once at startup, which means
//! one edited by hand takes effect on the next pick, with no rebuild. That is unlike
//! the built-ins, which are `include_str!`-ed and so only change when the viewer is
//! rebuilt.

use std::path::{Path, PathBuf};

use arbor_core::species::{builtin_presets, parse_template};
use arbor_core::SpeciesTemplate;

pub enum Source {
    Builtin(&'static str),
    Saved(PathBuf),
}

pub struct Preset {
    /// What the list shows and `--species` answers to: the built-in's key, or the
    /// saved file's name without its extension.
    pub name: String,
    pub source: Source,
}

impl Preset {
    pub fn is_saved(&self) -> bool {
        matches!(self.source, Source::Saved(_))
    }

    pub fn load(&self) -> Result<SpeciesTemplate, String> {
        match &self.source {
            Source::Builtin(src) => parse_template(src),
            Source::Saved(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
                parse_template(&text)
            }
        }
    }
}

/// Every preset: the built-ins in their own order, then the saved ones by name.
pub fn list(dir: &Path) -> Vec<Preset> {
    let mut all: Vec<Preset> = builtin_presets()
        .into_iter()
        .map(|(name, src)| Preset {
            name: name.to_string(),
            source: Source::Builtin(src),
        })
        .collect();
    all.extend(saved(dir));
    all
}

/// The presets saved in `dir`. A directory that does not exist yet simply has none.
fn saved(dir: &Path) -> Vec<Preset> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<Preset> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("ron")))
        .filter_map(|path| {
            let name = path.file_stem()?.to_str()?.to_string();
            Some(Preset {
                name,
                source: Source::Saved(path),
            })
        })
        .collect();
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

/// The file name a preset called `name` is saved under, or why it cannot be.
///
/// Names become file names and `--species` arguments, so they are kept to what is safe
/// as both: lower case letters, digits, `-` and `_`, with spaces taken as `_`. Lower
/// case because the file system on Windows would otherwise treat `Oak` and `oak` as one
/// preset while the list showed two. A built-in's name is refused, since a saved preset
/// under it could never be picked by name.
pub fn key_for(name: &str) -> Result<String, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("give the preset a name".to_string());
    }
    let mut key = String::with_capacity(trimmed.len());
    for c in trimmed.chars() {
        match c {
            'a'..='z' | '0'..='9' | '-' | '_' => key.push(c),
            'A'..='Z' => key.push(c.to_ascii_lowercase()),
            c if c.is_whitespace() => key.push('_'),
            other => {
                return Err(format!(
                    "'{other}' cannot go in a preset name: use letters, digits, - and _"
                ));
            }
        }
    }
    if key.len() > 64 {
        return Err("that name is too long".to_string());
    }
    if builtin_presets().iter().any(|(b, _)| *b == key) {
        return Err(format!("{key} is a built-in preset: pick another name"));
    }
    Ok(key)
}

/// Where a preset called `name` is, or would be, saved.
pub fn path_for(dir: &Path, name: &str) -> Result<PathBuf, String> {
    Ok(dir.join(format!("{}.ron", key_for(name)?)))
}

/// Writes `params` as a preset called `name` into `dir`, replacing one of that name if
/// there is one, and returns where it went. The species takes the name as it was typed,
/// so the file says what it is when read on its own.
pub fn save(dir: &Path, name: &str, params: &SpeciesTemplate) -> Result<PathBuf, String> {
    let path = path_for(dir, name)?;
    let mut species = params.clone();
    species.name = name.trim().to_string();
    let body = ron::ser::to_string_pretty(&species, ron::ser::PrettyConfig::default())
        .map_err(|e| format!("cannot write the species as RON: {e}"))?;
    let text = format!(
        "// Saved from the arbor viewer. Every value is written out, the seed included, so\n\
         // this grows exactly the tree that was saved. A pair is a range each tree lands\n\
         // in, decided by its seed. It is read afresh each time it is picked, so edits\n\
         // here show up on the next pick without a rebuild.\n{body}\n"
    );
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

/// Removes a saved preset's file. Built-ins cannot be deleted.
pub fn delete(preset: &Preset) -> Result<(), String> {
    match &preset.source {
        Source::Builtin(_) => Err(format!("{} is built in and cannot be deleted", preset.name)),
        Source::Saved(path) => std::fs::remove_file(path)
            .map_err(|e| format!("cannot delete {}: {e}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arbor_core::Ranged;

    /// A directory of its own under the system temp, emptied when the test is done.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "arbor-presets-{tag}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_saved_preset_comes_back_as_the_tree_that_was_saved() {
        let scratch = Scratch::new("roundtrip");
        let mut params = parse_template(builtin_presets()[1].1).unwrap();
        params.seed = 4242;
        params.trunk.length = Ranged::Between(14.0, 19.5);
        params.wind.flutter = Ranged::Fixed(0.9);

        let path = save(&scratch.0, "My Oak", &params).unwrap();
        assert_eq!(path.file_name().unwrap(), "my_oak.ron");

        let all = list(&scratch.0);
        let saved: Vec<&Preset> = all.iter().filter(|p| p.is_saved()).collect();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].name, "my_oak");
        let back = saved[0].load().unwrap();
        // Everything that was saved, the seed and the range included; only the name is
        // the new one.
        assert_eq!(back.name, "My Oak");
        assert_eq!(back.trunk.length, Ranged::Between(14.0, 19.5));
        params.name = "My Oak".to_string();
        assert_eq!(back, params);
    }

    #[test]
    fn saving_under_a_name_already_taken_replaces_it() {
        let scratch = Scratch::new("overwrite");
        let mut params = parse_template(builtin_presets()[0].1).unwrap();
        save(&scratch.0, "tall", &params).unwrap();
        params.seed = 99;
        save(&scratch.0, "Tall", &params).unwrap();
        let all = list(&scratch.0);
        let saved: Vec<&Preset> = all.iter().filter(|p| p.is_saved()).collect();
        assert_eq!(saved.len(), 1, "one name, one preset");
        assert_eq!(saved[0].load().unwrap().seed, 99);
    }

    #[test]
    fn deleting_a_saved_preset_takes_it_off_the_list() {
        let scratch = Scratch::new("delete");
        let params = parse_template(builtin_presets()[0].1).unwrap();
        save(&scratch.0, "gone", &params).unwrap();
        let all = list(&scratch.0);
        let gone = all.iter().find(|p| p.name == "gone").unwrap();
        delete(gone).unwrap();
        assert!(list(&scratch.0).iter().all(|p| p.name != "gone"));
        // And the built-ins are not the panel's to delete.
        let builtin = &list(&scratch.0)[0];
        assert!(delete(builtin).is_err());
    }

    #[test]
    fn names_have_to_be_safe_as_file_names_and_not_built_in() {
        assert_eq!(key_for("  Windy Birch 2 ").unwrap(), "windy_birch_2");
        assert_eq!(key_for("fir-open_b").unwrap(), "fir-open_b");
        assert!(key_for("").is_err());
        assert!(key_for("   ").is_err());
        assert!(key_for("../escape").is_err());
        assert!(key_for("a/b").is_err());
        assert!(key_for("oak").is_err(), "a saved oak could never be picked by name");
        assert!(key_for("Pine").is_err());
    }

    #[test]
    fn a_missing_directory_just_has_no_saved_presets() {
        let scratch = Scratch::new("missing");
        let all = list(&scratch.0);
        assert_eq!(all.len(), builtin_presets().len());
        assert!(all.iter().all(|p| !p.is_saved()));
    }
}
