//! Mip generation for foliage cutouts.
//!
//! Two chains, because the two ways of resolving a cutout want opposite things from a
//! minified texel. An alpha *test* compares alpha to a cutoff, so a plain box filter
//! slowly drops the whole leaf under the cutoff and the canopy vanishes as the camera
//! pulls back; [`coverage_preserving_chain`] rescales alpha so the fraction of texels
//! that pass stays put, and the offline preview shades with it. Alpha-to-coverage reads
//! the averaged alpha as a coverage fraction, so [`coverage_chain`] leaves it alone and
//! the canopy thins smoothly and to the right density; the viewer draws with it.

// The viewer, the offline preview, and the tests all include this file but each uses a
// different half of it, so an item one consumer does not touch is not really dead.
#![allow(dead_code)]

/// One level of a mip chain: pixels, width, height.
pub type MipLevel = (Vec<u8>, u32, u32);

/// How far a level's coverage may sit from level 0 before its alpha is rescaled.
/// Small overshoots are left alone: correcting them only erodes the soft edge of the
/// cutout, and the error compounds down the chain.
const COVERAGE_TOLERANCE: f32 = 0.03;

/// Fraction of texels whose alpha passes `cutoff`, after scaling alpha by `scale`.
pub fn alpha_coverage(rgba: &[u8], cutoff: f32, scale: f32) -> f32 {
    let total = rgba.len() / 4;
    if total == 0 {
        return 0.0;
    }
    let passing = rgba
        .chunks_exact(4)
        .filter(|p| (p[3] as f32 / 255.0 * scale).min(1.0) >= cutoff)
        .count();
    passing as f32 / total as f32
}

/// Builds the full chain down to 1x1, keeping alpha coverage equal to level 0.
///
/// This is the chain an alpha *test* wants: it keeps the fraction of texels that pass
/// a hard cutoff constant, which is what stops a binary cutout from eating itself as it
/// minifies. Alpha-to-coverage wants the other chain, [`coverage_chain`].
pub fn coverage_preserving_chain(rgba: &[u8], w: u32, h: u32, cutoff: f32) -> Vec<MipLevel> {
    let target = alpha_coverage(rgba, cutoff, 1.0);
    let mut levels: Vec<MipLevel> = vec![(rgba.to_vec(), w, h)];
    while levels.last().map(|(_, w, h)| *w > 1 || *h > 1) == Some(true) {
        let (src, sw, sh) = levels.last().expect("chain is never empty");
        let (mut next, nw, nh) = halve(src, *sw, *sh);
        if target > 0.0 {
            rescale_alpha(&mut next, cutoff, target);
        }
        levels.push((next, nw, nh));
    }
    levels
}

/// Builds the full chain down to 1x1 leaving alpha as the linear coverage it already
/// is, which is what alpha-to-coverage reads.
///
/// A minified texel's averaged alpha is exactly the fraction of that texel the leaf
/// covers, so handing it straight to coverage keeps the canopy's total density constant
/// instead of clumping into solid texels. Rescaling alpha to a hard cutoff here, as the
/// test-oriented chain does, is what makes distant foliage shimmer.
pub fn coverage_chain(rgba: &[u8], w: u32, h: u32) -> Vec<MipLevel> {
    let mut levels: Vec<MipLevel> = vec![(rgba.to_vec(), w, h)];
    while levels.last().map(|(_, w, h)| *w > 1 || *h > 1) == Some(true) {
        let (src, sw, sh) = levels.last().expect("chain is never empty");
        levels.push(halve(src, *sw, *sh));
    }
    levels
}

/// Decodes one sRGB byte to linear. The atlas is uploaded as an sRGB texture, so
/// averaging its texels has to happen in linear or the mip chain darkens every
/// high-contrast edge (half white and half black would average to 128 instead of 186).
fn srgb_decode(v: u8) -> f32 {
    (v as f32 / 255.0).powf(2.2)
}

/// Encodes one linear value back to the sRGB byte the texture stores.
fn srgb_encode(v: f32) -> u8 {
    (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
}

/// Box filter to half size. RGB is decoded to linear and averaged weighted by alpha,
/// so colour from fully transparent texels never bleeds into the visible part of the
/// leaf and the average matches what the hardware decodes at sample time.
pub fn halve(src: &[u8], w: u32, h: u32) -> MipLevel {
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    let mut out = Vec::with_capacity((nw * nh * 4) as usize);
    for y in 0..nh {
        for x in 0..nw {
            let mut weighted = [0.0f32; 3];
            let mut plain = [0.0f32; 3];
            let mut alpha_sum = 0.0f32;
            let mut count = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let sx = (x * 2 + dx).min(w - 1);
                    let sy = (y * 2 + dy).min(h - 1);
                    let i = ((sy * w + sx) * 4) as usize;
                    let a = src[i + 3] as f32 / 255.0;
                    for k in 0..3 {
                        let c = srgb_decode(src[i + k]);
                        weighted[k] += c * a;
                        plain[k] += c;
                    }
                    alpha_sum += a;
                    count += 1.0;
                }
            }
            for k in 0..3 {
                let c = if alpha_sum > 1e-5 {
                    weighted[k] / alpha_sum
                } else {
                    plain[k] / count
                };
                out.push(srgb_encode(c));
            }
            out.push((alpha_sum / count * 255.0).clamp(0.0, 255.0) as u8);
        }
    }
    (out, nw, nh)
}

/// Scales alpha so the share of texels passing `cutoff` comes back to `target`.
fn rescale_alpha(rgba: &mut [u8], cutoff: f32, target: f32) {
    // Coverage is a step function of the scale, so the bisection below lands on the
    // edge of whatever range of scales satisfies the target rather than on 1.0.
    // Where this level already has the coverage it should, leave its alpha untouched.
    if (alpha_coverage(rgba, cutoff, 1.0) - target).abs() <= COVERAGE_TOLERANCE {
        return;
    }
    // Coverage only ever rises with the scale, so it can be bisected directly.
    let (mut lo, mut hi) = (0.0f32, 64.0f32);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if alpha_coverage(rgba, cutoff, mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut scale = 0.5 * (lo + hi);
    if (scale - 1.0).abs() < 1e-3 {
        return;
    }

    let source: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
    let apply = |rgba: &mut [u8], scale: f32| {
        for (p, &a) in rgba.chunks_exact_mut(4).zip(source.iter()) {
            p[3] = ((a as f32 / 255.0 * scale).min(1.0) * 255.0).round() as u8;
        }
    };
    apply(rgba, scale);

    // The bisection lands on the scale where coverage *just* reaches the target, and
    // quantising back to 8 bits can then drop those texels straight back under the
    // cutoff. On the last couple of levels, where a handful of texels carry the whole
    // leaf, that is the difference between distant foliage and none.
    for _ in 0..8 {
        if alpha_coverage(rgba, cutoff, 1.0) + 1e-6 >= target {
            break;
        }
        scale *= 1.05;
        apply(rgba, scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A narrow opaque blade on a transparent field. Thin shapes are what actually
    /// dissolve: they are almost all edge, so averaging eats them fastest.
    fn blade(size: u32, half_width: f32) -> Vec<u8> {
        let mut px = Vec::with_capacity((size * size * 4) as usize);
        let c = size as f32 / 2.0;
        for y in 0..size {
            for x in 0..size {
                let along = (y as f32 - c) / c;
                let taper = (1.0 - along * along).max(0.0);
                let a = if (x as f32 - c).abs() < half_width * taper {
                    255
                } else {
                    0
                };
                px.extend_from_slice(&[40, 120, 40, a]);
            }
        }
        px
    }

    fn disc(size: u32, radius: f32) -> Vec<u8> {
        let mut px = Vec::with_capacity((size * size * 4) as usize);
        let c = size as f32 / 2.0;
        for y in 0..size {
            for x in 0..size {
                let d = ((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt();
                let a = if d < radius { 255 } else { 0 };
                px.extend_from_slice(&[40, 120, 40, a]);
            }
        }
        px
    }

    fn plain_chain(rgba: &[u8], w: u32, h: u32) -> Vec<MipLevel> {
        let mut levels: Vec<MipLevel> = vec![(rgba.to_vec(), w, h)];
        while levels.last().map(|(_, w, h)| *w > 1 || *h > 1) == Some(true) {
            let (src, sw, sh) = levels.last().unwrap();
            levels.push(halve(src, *sw, *sh));
        }
        levels
    }

    #[test]
    fn plain_mips_drop_a_cutout_entirely_at_the_bottom() {
        // The behaviour being corrected. A leaf occupies a fraction of its atlas cell,
        // so once the chain averages the whole cell into a few texels the result sits
        // under the alpha test and every fragment is discarded. On screen that is the
        // canopy vanishing as the camera pulls back, not fading.
        let src = blade(256, 30.0);
        let cutoff = 0.35;
        let plain = plain_chain(&src, 256, 256);
        let full = alpha_coverage(&plain[0].0, cutoff, 1.0);
        assert!(full < cutoff, "test shape should not fill its cell: {full}");

        let bottom = plain.last().expect("chain is never empty");
        assert_eq!(
            alpha_coverage(&bottom.0, cutoff, 1.0),
            0.0,
            "expected the plain chain to lose the cutout completely"
        );

        let fixed = coverage_preserving_chain(&src, 256, 256, cutoff);
        for (i, (px, w, h)) in fixed.iter().enumerate() {
            assert!(
                alpha_coverage(px, cutoff, 1.0) > 0.0,
                "level {i} ({w}x{h}) of the fixed chain is fully discarded"
            );
        }
    }

    #[test]
    fn coverage_holds_down_the_whole_chain() {
        let src = blade(256, 30.0);
        let chain = coverage_preserving_chain(&src, 256, 256, 0.35);
        let target = alpha_coverage(&chain[0].0, 0.35, 1.0);
        assert!(target > 0.0);
        // The last few levels are a handful of texels, where coverage cannot be
        // matched exactly; everything above that should track closely.
        for (i, (px, w, h)) in chain.iter().enumerate() {
            if *w < 8 || *h < 8 {
                continue;
            }
            let cov = alpha_coverage(px, 0.35, 1.0);
            assert!(
                (cov - target).abs() < 0.06,
                "level {i} ({w}x{h}) coverage {cov}, level 0 was {target}"
            );
        }
    }

    #[test]
    fn chain_reaches_one_by_one_with_halving_sizes() {
        let chain = coverage_preserving_chain(&disc(64, 20.0), 64, 64, 0.35);
        assert_eq!(chain.len(), 7);
        for (i, (px, w, h)) in chain.iter().enumerate() {
            assert_eq!((*w, *h), (64 >> i, 64 >> i), "level {i}");
            assert_eq!(px.len(), (w * h * 4) as usize);
        }
    }

    #[test]
    fn fully_opaque_textures_are_left_alone() {
        let src = vec![200u8; 32 * 32 * 4];
        let chain = coverage_preserving_chain(&src, 32, 32, 0.35);
        for (px, _, _) in &chain {
            assert!(px.chunks_exact(4).all(|p| p[3] == 200));
        }
    }

    #[test]
    fn transparent_texels_do_not_bleed_colour() {
        // Half opaque green, half transparent black: the averaged colour must stay
        // green rather than darkening toward the invisible half.
        let mut px = Vec::new();
        for y in 0..4u32 {
            for _ in 0..4u32 {
                if y < 2 {
                    px.extend_from_slice(&[0, 200, 0, 255]);
                } else {
                    px.extend_from_slice(&[0, 0, 0, 0]);
                }
            }
        }
        let (next, _, _) = halve(&px, 4, 4);
        assert_eq!(next[1], 200, "green channel was diluted by transparent texels");
    }

    #[test]
    fn colour_averages_in_linear_not_encoded() {
        // Half white, half black, fully opaque. Averaging the encoded bytes would
        // store 128, which decodes to ~0.216 linear; the physical mean of the two
        // texels is 0.5 linear, which encodes to 186. The stored mip must match the
        // linear mean or every mip level drifts darker than level 0.
        let mut px = Vec::new();
        for y in 0..4u32 {
            for x in 0..4u32 {
                if (x + y) % 2 == 0 {
                    px.extend_from_slice(&[255, 255, 255, 255]);
                } else {
                    px.extend_from_slice(&[0, 0, 0, 255]);
                }
            }
        }
        let (next, _, _) = halve(&px, 4, 4);
        for p in next.chunks_exact(4) {
            for &c in &p[..3] {
                assert_eq!(c, 186, "expected the linear mean encoded, got {c}");
            }
        }
    }

    #[test]
    fn coverage_chain_leaves_mean_alpha_alone() {
        // Area the leaf covers, and so the density alpha-to-coverage resolves, is the
        // mean alpha. Box filtering conserves it through the chain; the test-oriented
        // chain does not, which is the whole reason the two exist.
        let src = blade(256, 30.0);
        let chain = coverage_chain(&src, 256, 256);
        let mean = |px: &[u8]| {
            px.chunks_exact(4).map(|p| p[3] as f32 / 255.0).sum::<f32>()
                / (px.len() / 4) as f32
        };
        let base = mean(&src);
        for (i, (px, w, h)) in chain.iter().enumerate() {
            if *w < 8 || *h < 8 {
                continue;
            }
            assert!(
                (mean(px) - base).abs() < 0.02,
                "level {i} ({w}x{h}) mean alpha {} drifted from {base}",
                mean(px)
            );
        }
        // The point of the raw chain: the bottom mip sits well under the old cutoff yet
        // still carries real coverage for alpha-to-coverage to resolve.
        let bottom = &chain.last().expect("chain is never empty").0;
        assert!(
            mean(bottom) > 0.0 && mean(bottom) < 0.35,
            "the bottom mip should keep partial coverage, got {}",
            mean(bottom)
        );
    }
}
