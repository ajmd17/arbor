//! Numbers a species gives as a span for each tree to land in, rather than as a value.
//!
//! A species file describes a kind of tree, not one tree, and a great deal of what
//! tells two trees of one kind apart is how tall each grew, how far it spreads, how hard
//! its limbs hang. Any number in a species may be written as a pair — `length: (12.0,
//! 18.0)` — and every tree grown from it lands somewhere in that span, where being
//! decided by its seed. A plain number is the same for every tree, as it always was.
//!
//! This is a different thing from the variances a level already carries.
//! `length_variance` spreads the stems *within* one tree, drawn afresh for every stem; a
//! range is drawn once for the whole tree and holds for everything in it, so one tree
//! is tall and the next is short, rather than every tree being a mix of the two.
//!
//! A species is parsed into a `SpeciesParams<Ranged>`, which says what the kind allows,
//! and `instance` pins it down to the `SpeciesParams` one tree is grown from.

use std::fmt;

use serde::de::{self, IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::seed::land;

/// One number in a species: the same for every tree, or a span each tree lands in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ranged {
    Fixed(f32),
    /// From the first to the second, the point in between drawn once per tree.
    Between(f32, f32),
}

impl Ranged {
    pub fn is_range(self) -> bool {
        matches!(self, Ranged::Between(..))
    }

    /// The low end, or the value itself.
    pub fn lo(self) -> f32 {
        match self {
            Ranged::Fixed(v) | Ranged::Between(v, _) => v,
        }
    }

    /// The high end, or the value itself.
    pub fn hi(self) -> f32 {
        match self {
            Ranged::Fixed(v) | Ranged::Between(_, v) => v,
        }
    }

    /// The value `u` of the way through the span, from 0 at its first end to 1 at its
    /// second. A fixed number is itself whatever `u` is.
    pub fn at(self, u: f32) -> f32 {
        match self {
            Ranged::Fixed(v) => v,
            Ranged::Between(lo, hi) => lo + (hi - lo) * u,
        }
    }

    /// Where the tree grown from `seed` lands, for the number known as `key`.
    pub fn land(self, seed: u64, key: &str) -> f32 {
        match self {
            Ranged::Fixed(v) => v,
            Ranged::Between(..) => self.at(land(seed, key)),
        }
    }
}

impl From<f32> for Ranged {
    fn from(v: f32) -> Self {
        Ranged::Fixed(v)
    }
}

/// What a species' numbers are made of: `f32` for one tree, `Ranged` for a kind of
/// tree. Every parameter struct is generic over it, so the two forms share one
/// description, one set of defaults and one file format.
pub trait Scalar: Copy + PartialEq + fmt::Debug {
    /// A number that is the same for every tree.
    fn fixed(v: f32) -> Self;
}

impl Scalar for f32 {
    fn fixed(v: f32) -> Self {
        v
    }
}

impl Scalar for Ranged {
    fn fixed(v: f32) -> Self {
        Ranged::Fixed(v)
    }
}

/// The key a field is known by as a species is walked: its path through the species,
/// `trunk.children.scale` or `branch_levels.1.length`. It is what a range is drawn
/// against, so it has to stay put as long as the field does.
pub(crate) fn key(at: &str, field: &str) -> String {
    if at.is_empty() {
        field.to_string()
    } else {
        format!("{at}.{field}")
    }
}

/// Each number of an array through `f`, keyed by its index under `at`: `leaves.tint.1`.
pub(crate) fn map_array<V: Copy, W, const N: usize>(
    at: &str,
    values: [V; N],
    f: &mut impl FnMut(&str, V) -> W,
) -> [W; N] {
    let mut i = 0;
    values.map(|v| {
        let k = format!("{at}.{i}");
        i += 1;
        f(&k, v)
    })
}

// Written as the bare number when fixed, so a species with no ranges in it reads and
// writes exactly as it did before there were any.
impl Serialize for Ranged {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match *self {
            Ranged::Fixed(v) => s.serialize_f32(v),
            Ranged::Between(lo, hi) => (lo, hi).serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for Ranged {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct RangedVisitor;

        impl<'de> Visitor<'de> for RangedVisitor {
            type Value = Ranged;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a number, or a (low, high) pair for each tree to land between")
            }

            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Ranged, E> {
                Ok(Ranged::Fixed(v as f32))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Ranged, E> {
                Ok(Ranged::Fixed(v as f32))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Ranged, E> {
                Ok(Ranged::Fixed(v as f32))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Ranged, A::Error> {
                let lo: f32 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let hi: f32 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                if seq.next_element::<IgnoredAny>()?.is_some() {
                    return Err(de::Error::invalid_length(3, &self));
                }
                Ok(Ranged::Between(lo, hi))
            }
        }

        d.deserialize_any(RangedVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Holder {
        a: Ranged,
        b: Ranged,
        c: Ranged,
        d: [Ranged; 2],
    }

    #[test]
    fn a_number_or_a_pair_reads_and_writes_as_it_was_written() {
        let src = "(a: 15.0, b: (12.0, 18.5), c: -3, d: (0.5, (0.25, 0.75)))";
        let h: Holder = ron::from_str(src).unwrap();
        assert_eq!(h.a, Ranged::Fixed(15.0));
        assert_eq!(h.b, Ranged::Between(12.0, 18.5));
        assert_eq!(h.c, Ranged::Fixed(-3.0));
        assert_eq!(h.d, [Ranged::Fixed(0.5), Ranged::Between(0.25, 0.75)]);
        let back: Holder = ron::from_str(&ron::to_string(&h).unwrap()).unwrap();
        assert_eq!(back, h);
        assert!(ron::to_string(&h).unwrap().contains("a:15.0"), "a fixed number stays bare");
    }

    #[test]
    fn a_pair_needs_exactly_two_ends() {
        assert!(ron::from_str::<Ranged>("(1.0)").is_err());
        assert!(ron::from_str::<Ranged>("(1.0, 2.0, 3.0)").is_err());
        assert!(ron::from_str::<Ranged>("\"tall\"").is_err());
    }

    #[test]
    fn every_seed_lands_inside_the_span_and_they_do_not_all_land_together() {
        let r = Ranged::Between(12.0, 18.0);
        let landed: Vec<f32> = (0..200).map(|seed| r.land(seed, "trunk.length")).collect();
        assert!(landed.iter().all(|v| (12.0..=18.0).contains(v)));
        let low = landed.iter().filter(|&&v| v < 14.0).count();
        let high = landed.iter().filter(|&&v| v > 16.0).count();
        assert!(low > 40 && high > 40, "a third each side, roughly: {low} low, {high} high");
        assert_eq!(Ranged::Fixed(7.0).land(3, "anything"), 7.0);
    }

    #[test]
    fn each_number_is_drawn_on_its_own() {
        // Two fields with the same span on the same tree land in different places,
        // and the same field lands in the same place every time.
        let r = Ranged::Between(0.0, 1.0);
        assert_ne!(r.land(5, "trunk.length"), r.land(5, "trunk.radius"));
        assert_eq!(r.land(5, "trunk.length"), r.land(5, "trunk.length"));
    }
}
