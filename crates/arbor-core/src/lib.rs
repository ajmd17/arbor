pub mod envelope;
pub mod growth;
pub(crate) mod math;
pub mod mesh;
pub mod seed;
pub mod skeleton;
pub mod species;

pub use envelope::EnvelopeParams;
pub use growth::grow;
pub use mesh::{build_mesh, Mesh};
pub use skeleton::{Skeleton, SkeletonNode, SkeletonStats};
pub use species::SpeciesParams;
