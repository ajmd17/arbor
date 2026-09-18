use rand::rand_core::{impls, le};
use rand::{Rng, RngCore, SeedableRng};

const PATH_PRIME: u64 = 0x9E3779B97F4A7C15;
/// SplitMix64's step: the golden ratio as a fraction of 2^64.
const PHI: u64 = 0x9E3779B97F4A7C15;

/// The generator every draw in a tree comes from: xoshiro256++, seeded through
/// SplitMix64, draw for draw what `rand`'s `SmallRng` is on a 64-bit machine. It is
/// named here rather than taken as `SmallRng` because on a 32-bit target, the web
/// among them, `SmallRng` is another generator, and a seed would grow another tree.
#[derive(Clone, Debug)]
pub struct PortableRng {
    s: [u64; 4],
}

impl SeedableRng for PortableRng {
    type Seed = [u8; 32];

    fn from_seed(seed: [u8; 32]) -> Self {
        let mut s = [0; 4];
        le::read_u64_into(&seed, &mut s);
        // All zeros is the one state xoshiro never leaves.
        if s == [0; 4] {
            return Self::seed_from_u64(0);
        }
        Self { s }
    }

    fn seed_from_u64(mut state: u64) -> Self {
        let mut s = [0; 4];
        for word in &mut s {
            state = state.wrapping_add(PHI);
            *word = splitmix(state);
        }
        Self { s }
    }
}

impl RngCore for PortableRng {
    fn next_u32(&mut self) -> u32 {
        // The low bits are the weak ones.
        (self.next_u64() >> 32) as u32
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let out = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        impls::fill_bytes_via_next(self, dst)
    }
}

/// SplitMix64's output function.
fn splitmix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub struct TreeRng {
    seed: u64,
}

impl TreeRng {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn stream(&self, path: u64) -> PortableRng {
        let mut z = self.seed ^ mix(path);
        z = mix(z);
        z = mix(z);
        PortableRng::seed_from_u64(z)
    }
}

pub fn child_path(parent: u64, slot: u32) -> u64 {
    parent
        .wrapping_mul(PATH_PRIME)
        .wrapping_add(slot as u64 + 1)
        .max(1)
}

fn mix(z: u64) -> u64 {
    splitmix(z.wrapping_add(PHI))
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

pub fn range_f32(rng: &mut PortableRng, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return lo;
    }
    rng.random_range(lo..hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every tree grown before the generator was pinned was grown with `SmallRng` on a
    /// 64-bit machine, and has to come out of this one exactly the same.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn the_portable_generator_is_small_rng_on_a_64_bit_machine() {
        use rand::rngs::SmallRng;
        for seed in [0, 1, 7, 42, u64::MAX, 0x5EED_1A2D_0F7A_11E5] {
            let (mut ours, mut theirs) = (PortableRng::seed_from_u64(seed), SmallRng::seed_from_u64(seed));
            for _ in 0..64 {
                assert_eq!(ours.next_u64(), theirs.next_u64(), "seed {seed}");
                assert_eq!(ours.next_u32(), theirs.next_u32(), "seed {seed}");
                assert_eq!(ours.random_range(-1.5f32..2.5), theirs.random_range(-1.5f32..2.5));
                assert_eq!(ours.random_range(0..13usize), theirs.random_range(0..13usize));
            }
        }
    }
}
