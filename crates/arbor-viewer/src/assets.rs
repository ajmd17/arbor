//! The texture maps and environments, as the viewer gets hold of them.
//!
//! On the desktop a file is read off disk the moment it is wanted. On the web it has to
//! be fetched, and arrives some frames later, so a species' maps are asked for first
//! with `ready` and read once they are all in.

#[cfg(not(target_arch = "wasm32"))]
use arbor_core::textures::{MapSource, TEXTURE_DIR};

/// The files in one folder on disk.
#[cfg(not(target_arch = "wasm32"))]
pub struct Maps {
    dir: &'static std::path::Path,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for Maps {
    /// The repository's texture folder.
    fn default() -> Self {
        Self::new(TEXTURE_DIR)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Maps {
    pub fn new(dir: &'static str) -> Self {
        Self {
            dir: std::path::Path::new(dir),
        }
    }

    /// Whether every one of `files` can be read now. On disk they always can.
    pub fn ready(&self, _files: &[String]) -> bool {
        true
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl MapSource for Maps {
    fn read(&self, file: &str) -> Option<Vec<u8>> {
        self.dir.read(file)
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::Maps;

#[cfg(target_arch = "wasm32")]
mod web {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use arbor_core::textures::{MapSource, TEXTURE_DIR};

    enum Fetch {
        Waiting,
        /// The file, or `None` for one the server does not have.
        Done(Option<Vec<u8>>),
    }

    /// The files fetched so far, from one folder beside the page.
    pub struct Maps {
        dir: &'static str,
        files: Arc<Mutex<HashMap<String, Fetch>>>,
    }

    impl Default for Maps {
        /// The texture folder.
        fn default() -> Self {
            Self::new(TEXTURE_DIR)
        }
    }

    impl Maps {
        pub fn new(dir: &'static str) -> Self {
            Self {
                dir,
                files: Arc::default(),
            }
        }

        /// Whether every one of `files` is in, or known not to exist. Any not asked for
        /// yet is fetched now.
        pub fn ready(&self, files: &[String]) -> bool {
            let mut wanted = Vec::new();
            let mut ready = true;
            {
                let mut known = self.files.lock().unwrap();
                for file in files {
                    match known.get(file) {
                        Some(Fetch::Done(_)) => {}
                        Some(Fetch::Waiting) => ready = false,
                        None => {
                            known.insert(file.clone(), Fetch::Waiting);
                            wanted.push(file.clone());
                            ready = false;
                        }
                    }
                }
            }
            // Asked for with the lock let go, so an answer can never find it held.
            for file in wanted {
                let files = Arc::clone(&self.files);
                let request = ehttp::Request::get(format!("{}/{file}", self.dir));
                ehttp::fetch(request, move |response| {
                    let bytes = response.ok().filter(|r| r.ok).map(|r| r.bytes);
                    files.lock().unwrap().insert(file, Fetch::Done(bytes));
                });
            }
            ready
        }
    }

    impl MapSource for Maps {
        fn read(&self, file: &str) -> Option<Vec<u8>> {
            match self.files.lock().unwrap().get(file) {
                Some(Fetch::Done(bytes)) => bytes.clone(),
                _ => None,
            }
        }
    }
}
