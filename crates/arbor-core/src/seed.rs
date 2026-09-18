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

/// Where the tree grown from `seed` lands in the range declared at `key`, from 0 to 1.
///
/// Drawn from the key itself rather than from one stream shared by every range, so
/// each range is drawn on its own: giving one number a range, or taking it away, leaves
/// every other number landing exactly where it did, and dragging one end of a range
/// slides the tree through it instead of reshuffling the whole tree. Salted apart from
/// the growth streams, which a range must never disturb — a species with no ranges in
/// it grows the same tree it always did.
pub fn land(seed: u64, key: &str) -> f32 {
    const LAND_SALT: u64 = 0x5EED_1A2D_0F7A_11E5;
    // FNV-1a, which is all a short ASCII path needs.
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in key.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01B3);
    }
    let z = mix(mix(seed ^ LAND_SALT) ^ h);
    (z >> 40) as f32 / (1u64 << 24) as f32
}

pub fn range_f32(rng: &mut SmallRng, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return lo;
    }
    rng.random_range(lo..hi)
}
