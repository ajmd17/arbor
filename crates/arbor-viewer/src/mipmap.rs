//! Mip generation for alpha-tested foliage.
//!
//! A plain box filter halves a leaf texture by averaging alpha, so a leaf that covers
//! 30% of its texels at full size covers less and less of each texel as the chain goes
//! down. Once the averaged alpha drops under the alpha-test cutoff the fragments are
//! discarded, and the canopy visibly thins and then disappears as the camera pulls
//! back. The fix is to rescale alpha in every mip so the fraction of texels that pass
//! the cutoff stays what it was at full resolution.

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

/// Box filter to half size. RGB is averaged weighted by alpha so colour from fully
/// transparent texels never bleeds into the visible part of the leaf.
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
                        let c = src[i + k] as f32;
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
                out.push(c.clamp(0.0, 255.0) as u8);
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
}
