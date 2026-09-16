use rand::rngs::SmallRng;
use rand::SeedableRng;

use rand::Rng;

const PATH_PRIME: u64 = 0x9E3779B97F4A7C15;

pub struct TreeRng {
    seed: u64,
}

impl TreeRng {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn stream(&self, path: u64) -> SmallRng {
        let mut z = self.seed ^ mix(path);
        z = mix(z);
        z = mix(z);
        SmallRng::seed_from_u64(z)
    }
}

pub fn child_path(parent: u64, slot: u32) -> u64 {
    parent
        .wrapping_mul(PATH_PRIME)
        .wrapping_add(slot as u64 + 1)
        .max(1)
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub fn range_f32(rng: &mut SmallRng, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return lo;
    }
    rng.random_range(lo..hi)
}
