//! Every shader the viewer draws with.
//!
//! The lit shaders are put together from shared pieces, always in the same order:
//! [COMMON_GLSL], [SKY_GLSL], [ENV_GLSL], [BRDF_GLSL], [SHADOW_GLSL], then the body.
//! Kept as pieces in one place because two copies of a BRDF are two chances for the
//! bark and the leaves to disagree about what light does.
//!
//! Lit shaders write linear radiance already multiplied by the exposure (Filament's
//! pre-exposure) into a half-float target, so a sunlit frame and a dusk one both land
//! well inside what a half float holds. Nothing here tonemaps except [COMPOSITE_FS],
//! once, after the bloom.

/// Constants, noise and mappings every shader can use.
pub const COMMON_GLSL: &str = r#"
const float PI = 3.14159265359;
uniform float u_exposure;

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

// Value noise on a world position, for breaking up anything that would otherwise be
// uniform across a whole surface.
float hash31(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
}

float value_noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash31(i), n100 = hash31(i + vec3(1, 0, 0));
    float n010 = hash31(i + vec3(0, 1, 0)), n110 = hash31(i + vec3(1, 1, 0));
    float n001 = hash31(i + vec3(0, 0, 1)), n101 = hash31(i + vec3(1, 0, 1));
    float n011 = hash31(i + vec3(0, 1, 1)), n111 = hash31(i + vec3(1, 1, 1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

// Interleaved gradient noise (Jimenez): a different value per pixel, spread so evenly
// that the error it leaves reads as fine grain rather than as a pattern.
float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// The equirectangular mapping `hdri.rs` uses: the middle of the image looks down -Z,
// the image runs rightward toward +X, and the top row is straight up.
vec2 equirect_uv(vec3 d) {
    return vec2(0.5 + atan(d.x, -d.z) / (2.0 * PI), acos(clamp(d.y, -1.0, 1.0)) / PI);
}

// The direction through a point of a cube map face, `st` running -1 to 1 along the
// face's s and t, t increasing with the texel row. The inverse of GL's face selection.
vec3 cube_dir(int face, vec2 st) {
    if (face == 0) return vec3(1.0, -st.y, -st.x);
    if (face == 1) return vec3(-1.0, -st.y, st.x);
    if (face == 2) return vec3(st.x, 1.0, st.y);
    if (face == 3) return vec3(st.x, -1.0, -st.y);
    if (face == 4) return vec3(st.x, -st.y, 1.0);
    return vec3(-st.x, -st.y, -1.0);
}
"#;

/// The procedural sky: a dome built from the sun's height (see `lighting.rs`).
pub const SKY_GLSL: &str = r#"
// The dome: a pale band at the horizon under a colder zenith, plus the bounce coming
// back up off the ground. The colours arrive already built from the sun's angle, so
// this shape is all that is left to do here.
uniform vec3 u_sky_zenith;
uniform vec3 u_sky_horizon;
uniform vec3 u_ground_bounce;

vec3 sky_color(vec3 dir) {
    float h = clamp(dir.y, -1.0, 1.0);
    vec3 dome = mix(u_sky_horizon, u_sky_zenith, pow(max(h, 0.0), 0.42));
    return mix(u_ground_bounce, dome, smoothstep(-0.25, 0.03, h));
}

// The dome with the wash the sun throws across it, but not the disc: the disc is the
// sun itself, which every surface already takes as a light of its own.
vec3 sky_radiance(vec3 dir, vec3 sun_dir, vec3 sun_color) {
    float d = max(dot(dir, sun_dir), 0.0);
    return sky_color(dir) + sun_color * 0.09 * pow(d, 18.0) + u_sky_horizon * 0.35 * pow(d, 3.0);
}
"#;

/// Where the light comes from: the sun, and everything else. Everything else is the
/// environment, the procedural sky or a photograph, with its sun taken out of it.
pub const ENV_GLSL: &str = r#"
uniform vec3 u_sun_dir;
// Irradiance on a surface square to the sun: a Lambert surface returns albedo / PI of it.
uniform vec3 u_sun_color;
// Angular radius of the sun. A shadow softens with distance from its caster by it.
uniform float u_sun_radius;
uniform vec3 u_cam_pos;
// 0 for the procedural sky, 1 for a photograph.
uniform int u_env_mode;
// World directions into the environment's own frame, which a photograph can be turned in.
uniform mat3 u_env_rot;
uniform float u_env_intensity;
// The environment as nine spherical harmonics, for the light a surface gathers.
uniform vec3 u_sh[9];
// The environment prefiltered for GGX, a step of roughness per mip.
uniform samplerCube u_env_specular;
uniform float u_env_max_lod;
// The photograph itself, sun and all, for the background and the ground it was shot over.
uniform sampler2D u_equirect;
// Split-sum scale and bias for GGX, with multiple scattering (Filament's DFG).
uniform sampler2D u_dfg;
// Screen-space ambient occlusion, one texel per pixel of the target being drawn.
uniform sampler2D u_ao_tex;
uniform vec2 u_ao_texel;
// The photograph laid onto a dome with a flat floor, so the tree stands on the ground
// in the picture and what is far off stands up around it. x: how far above the floor
// the photograph was taken. y: 1 to do this at all. z: the dome's radius.
uniform vec3 u_ground_proj;
uniform float u_background_blur;
// How much sky the crown leaves open, seen from straight below: the leaves' coverage
// looked down on, blurred as wide as the sky it takes. `u_canopy_map` is the map's
// centre on the ground (xy), one over its half-width (z) and how much it counts (w);
// `u_canopy_heights` the bottom and top of the crown.
uniform sampler2D u_canopy;
uniform vec4 u_canopy_map;
uniform vec2 u_canopy_heights;

// Irradiance arriving at a surface facing n, divided by PI so a surface multiplies it
// by albedo and nothing else. Ramamoorthi and Hanrahan's closed form: it integrates
// the whole environment, so a face turned down gets the ground and a face turned up
// the sky, whichever sky that is.
vec3 irradiance(vec3 n) {
    const float c1 = 0.429043 / PI;
    const float c2 = 0.511664 / PI;
    const float c3 = 0.743125 / PI;
    const float c4 = 0.886227 / PI;
    const float c5 = 0.247708 / PI;
    vec3 m = u_env_rot * n;
    float x = m.x, y = m.y, z = m.z;
    vec3 e = u_sh[0] * c4 - u_sh[6] * c5
        + (u_sh[3] * x + u_sh[1] * y + u_sh[2] * z) * (2.0 * c2)
        + u_sh[6] * (c3 * z * z)
        + (u_sh[4] * x * y + u_sh[7] * x * z + u_sh[5] * y * z) * (2.0 * c1)
        + u_sh[8] * (c1 * (x * x - y * y));
    return max(e, vec3(0.0)) * u_env_intensity;
}

// Filament's roughness-to-mip mapping, which the prefilter was built to match.
vec3 prefiltered_radiance(vec3 r, float perceptual) {
    float lod = u_env_max_lod * perceptual * (2.0 - perceptual);
    return textureLod(u_env_specular, u_env_rot * r, lod).rgb * u_env_intensity;
}

vec2 env_dfg(float NoV, float perceptual) {
    return textureLod(u_dfg, vec2(NoV, perceptual), 0.0).xy;
}

float screen_ao() {
    return texture(u_ao_tex, gl_FragCoord.xy * u_ao_texel).r;
}

// The share of the sky the crown leaves open above p. Screen-space occlusion only
// reaches a metre or so; this is what darkens the ground under a tree, and a trunk
// standing in its own crown's shade, the way a crown of leaves does.
float canopy_sky(vec3 p) {
    if (u_canopy_map.w <= 0.0) {
        return 1.0;
    }
    vec2 uv = (p.xz - u_canopy_map.xy) * (u_canopy_map.z * 0.5) + 0.5;
    float open = textureLod(u_canopy, uv, 0.0).r;
    // Only crown overhead shades a point, so it counts less the higher up it is.
    float below = 1.0 - smoothstep(u_canopy_heights.x, u_canopy_heights.y, p.y);
    return mix(1.0, open, u_canopy_map.w * below);
}

// The photograph looking along a world direction. Filtered by its own screen-space
// rate of change with the jump where the longitude wraps taken out, or the seam reads
// the smallest mip and draws a line down the sky.
vec3 photo(vec3 dir) {
    vec2 uv = equirect_uv(u_env_rot * dir);
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);
    dx.x -= floor(dx.x + 0.5);
    dy.x -= floor(dy.x + 0.5);
    return textureGrad(u_equirect, uv, dx, dy).rgb * u_env_intensity;
}

// Where a ray from the camera meets the photograph laid onto its dome, as the
// direction from where the photograph was taken to that point; `floor` says whether
// it met the floor rather than the wall. Without the dome the whole lower half would
// lie flat on an endless floor, and the trees on the horizon with it.
vec3 projected_look(vec3 dir, out float floor_hit) {
    vec3 centre = vec3(0.0, u_ground_proj.x, 0.0);
    vec3 o = u_cam_pos - centre;
    float b = dot(o, dir);
    float c = dot(o, o) - u_ground_proj.z * u_ground_proj.z;
    float t_wall = -b + sqrt(max(b * b - c, 0.0));
    floor_hit = 0.0;
    if (dir.y < 0.0 && u_cam_pos.y > 0.0) {
        float t_floor = -u_cam_pos.y / dir.y;
        if (t_floor < t_wall) {
            floor_hit = 1.0;
            return normalize(o + dir * t_floor);
        }
    }
    return normalize(o + dir * t_wall);
}

// The sun spread as wide as the background blur, for the blurred background: that is
// read from the prefiltered map, which the sun was taken out of.
vec3 sun_glow(vec3 dir, float blur) {
    float width = mix(max(u_sun_radius, 0.005), 0.45, blur * blur);
    float angle = acos(clamp(dot(dir, u_sun_dir), -1.0, 1.0));
    return u_sun_color * exp(-(angle * angle) / (width * width)) / (PI * width * width);
}

// What a ray from the camera sees when it reaches nothing, and whether that is the
// photograph's floor.
vec3 background_at(vec3 dir, out float floor_hit) {
    floor_hit = 0.0;
    if (u_env_mode == 0) {
        vec3 c = sky_radiance(dir, u_sun_dir, u_sun_color);
        // The disc, at the sun's irradiance spread over its own solid angle but held
        // short of what a half float can carry, and the glare around it.
        float d = dot(dir, u_sun_dir);
        float r = max(u_sun_radius, 0.003);
        float disc = smoothstep(cos(r * 1.15), cos(r * 0.85), d);
        c += u_sun_color * disc * min(1.0 / (PI * r * r), 600.0);
        c += u_sun_color * 0.22 * pow(max(d, 0.0), 900.0);
        return c;
    }
    // The photograph is looked up from where it was taken toward the point the ray
    // meets: the ground in the picture then lies flat under the tree instead of
    // hanging at infinity around it.
    vec3 look = u_ground_proj.y > 0.5 ? projected_look(dir, floor_hit) : dir;
    vec3 sharp = photo(look);
    float blur = u_background_blur * (1.0 - floor_hit);
    if (blur <= 0.0) {
        return sharp;
    }
    vec3 soft = textureLod(u_env_specular, u_env_rot * look, blur * u_env_max_lod).rgb * u_env_intensity
        + sun_glow(look, blur);
    return mix(sharp, soft, min(blur * 6.0, 1.0));
}

vec3 background(vec3 dir) {
    float floor_hit;
    return background_at(dir, floor_hit);
}
"#;

/// Filament's standard surface: GGX distribution, height-correlated Smith visibility,
/// Schlick Fresnel and Lambert diffuse, with the energy that single-scattering GGX
/// loses at high roughness put back, and the environment lit by split sum.
pub const BRDF_GLSL: &str = r#"
float D_GGX(float NoH, float a) {
    float a2 = a * a;
    float f = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (PI * f * f + 1e-7);
}

// Height-correlated Smith, with the 1 / (4 NoL NoV) folded in.
float V_SmithGGXCorrelated(float NoV, float NoL, float a) {
    float a2 = a * a;
    float lv = NoL * sqrt((NoV - NoV * a2) * NoV + a2);
    float ll = NoV * sqrt((NoL - NoL * a2) * NoL + a2);
    return 0.5 / max(lv + ll, 1e-5);
}

vec3 F_Schlick(vec3 f0, float VoH) {
    float f = pow(1.0 - VoH, 5.0);
    return f + f0 * (1.0 - f);
}

// Specular anti-aliasing (Kaplanyan and Hoffman, as Filament has it): where the normal
// turns faster than a pixel can follow, the highlight it would draw is narrower than a
// pixel and sparkles, so the roughness is widened by how fast it turns.
float filtered_roughness(float perceptual, vec3 n) {
    vec3 du = dFdx(n);
    vec3 dv = dFdy(n);
    float variance = 0.15 * (dot(du, du) + dot(dv, dv));
    float a = perceptual * perceptual;
    float a2 = clamp(a * a + min(2.0 * variance, 0.2), 0.0, 1.0);
    return sqrt(sqrt(a2));
}

// Ambient occlusion with the light that bounces back out of a crevice (Jimenez): a
// bright surface loses less of its ambient to occlusion than a dark one.
vec3 multi_bounce(float visibility, vec3 albedo) {
    vec3 a = 2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c = 2.7552 * albedo + 0.6903;
    return max(vec3(visibility), ((visibility * a + b) * visibility + c) * visibility);
}

// How much of the reflected environment an occluded surface still sees (Lagarde).
float specular_occlusion(float NoV, float visibility, float a) {
    return saturate(pow(NoV + visibility, exp2(-16.0 * a - 1.0)) - 1.0 + visibility);
}

struct Surface {
    vec3 diffuse;
    vec3 f0;
    float perceptual;
    float alpha;
    vec3 n;
    vec3 v;
    float NoV;
    // Directional albedo of the specular lobe, and the multiscatter compensation.
    vec3 E;
    vec3 energy;
};

Surface surface(vec3 base, float metallic, float perceptual, vec3 n, vec3 v) {
    Surface s;
    s.diffuse = base * (1.0 - metallic);
    s.f0 = mix(vec3(0.04), base, metallic);
    s.perceptual = perceptual;
    s.alpha = perceptual * perceptual;
    s.n = n;
    s.v = v;
    s.NoV = max(dot(n, v), 1e-4);
    vec2 dfg = env_dfg(s.NoV, perceptual);
    s.E = mix(dfg.xxx, dfg.yyy, s.f0);
    s.energy = 1.0 + s.f0 * (1.0 / max(dfg.y, 1e-3) - 1.0);
    return s;
}

vec3 specular_lobe(Surface s, vec3 l, float NoL) {
    vec3 h = normalize(s.v + l);
    float NoH = saturate(dot(s.n, h));
    float VoH = saturate(dot(s.v, h));
    return D_GGX(NoH, s.alpha) * V_SmithGGXCorrelated(s.NoV, NoL, s.alpha)
        * F_Schlick(s.f0, VoH) * s.energy;
}

// Light arriving along l, per unit of irradiance: diffuse and specular, times the
// cosine. The light's colour and its shadow are the caller's.
vec3 direct(Surface s, vec3 l) {
    float NoL = saturate(dot(s.n, l));
    return (s.diffuse / PI + specular_lobe(s, l, NoL)) * NoL;
}

// Everything but the sun: the environment's irradiance for the diffuse part and its
// prefiltered radiance for the specular part, each occluded in its own way. `ng` is
// the surface's own normal, before any normal map.
vec3 ambient(Surface s, vec3 ng, float visibility) {
    vec3 r = reflect(-s.v, s.n);
    // A normal map can aim a reflection below the surface it is drawn on, where the
    // surface itself would hide what it reflects.
    float horizon = min(1.0 + dot(r, ng), 1.0);
    vec3 spec = s.E * prefiltered_radiance(r, s.perceptual) * s.energy
        * (specular_occlusion(s.NoV, visibility, s.alpha) * horizon * horizon);
    vec3 diff = s.diffuse * irradiance(s.n) * (1.0 - s.E) * multi_bounce(visibility, s.diffuse);
    return diff + spec;
}
"#;

/// Sun shadows, softened the way a disc of sun softens them.
pub const SHADOW_GLSL: &str = r#"
// The map's raw depths, for finding what stands between a point and the sun, and the
// same map compared in hardware, for filtering.
uniform sampler2D u_shadow_tex;
uniform sampler2DShadow u_shadow_cmp;
// x: one texel, in uv. y: metres per unit of stored depth. z: metres per texel.
// w: tangent of the sun's angular radius.
uniform vec4 u_shadow_params;

const vec2 POISSON[16] = vec2[16](
    vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845), vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367), vec2(0.14383161, -0.14100790)
);

// Percentage-closer soft shadows. The sun is a disc rather than a point, so a shadow
// is sharp where it meets whatever casts it and loosens with distance from it: crisp
// at the foot of the trunk, soft under the far edge of the crown. The blockers are
// found first, and how far below them the point lies sets how wide to filter.
//
// `bias_m` is in metres, for a face square to the light; a sloped face adds its own
// run across the filter to it. `taps` trades grain for cost, up to 16.
float soft_shadow(vec4 sc, float NoL, float bias_m, int taps) {
    vec3 p = sc.xyz / sc.w * 0.5 + 0.5;
    if (p.x <= 0.0 || p.x >= 1.0 || p.y <= 0.0 || p.y >= 1.0 || p.z >= 1.0) {
        return 1.0;
    }
    float texel = u_shadow_params.x;
    float angle = ign(gl_FragCoord.xy) * 2.0 * PI;
    float cs = cos(angle), sn = sin(angle);
    mat2 spin = mat2(cs, sn, -sn, cs);
    float slope = min(sqrt(max(1.0 - NoL * NoL, 0.0)) / max(NoL, 0.05), 4.0);
    float per_texel = u_shadow_params.z / u_shadow_params.y;
    float bias = bias_m / u_shadow_params.y;

    // As wide as a penumbra could be, cast from as far above as the map reaches.
    float search = clamp(u_shadow_params.w * p.z * u_shadow_params.y / u_shadow_params.z, 2.0, 40.0);
    float found = 0.0;
    float blockers = 0.0;
    float search_bias = bias + slope * search * per_texel * 0.5;
    for (int i = 0; i < taps; ++i) {
        float d = textureLod(u_shadow_tex, p.xy + spin * POISSON[i] * (search * texel), 0.0).r;
        if (d < p.z - search_bias) {
            found += d;
            blockers += 1.0;
        }
    }
    if (blockers == 0.0) {
        return 1.0;
    }
    float gap = max(p.z - found / blockers, 0.0) * u_shadow_params.y;
    float radius = clamp(gap * u_shadow_params.w / u_shadow_params.z, 1.0, 40.0);
    float b = bias + slope * radius * per_texel;
    float lit = 0.0;
    for (int i = 0; i < taps; ++i) {
        lit += texture(u_shadow_cmp, vec3(p.xy + spin * POISSON[i] * (radius * texel), p.z - b));
    }
    return lit / float(taps);
}
"#;

/// Low-discrepancy GGX sampling, for building the environment's lookup tables.
pub const SAMPLING_GLSL: &str = r#"
float radical_inverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 hammersley(int i, int n) {
    return vec2(float(i) / float(n), radical_inverse(uint(i)));
}

// A half vector drawn in proportion to D(h) (n.h), about +Z.
vec3 importance_ggx(vec2 u, float a) {
    float phi = 2.0 * PI * u.x;
    float cos_t = sqrt((1.0 - u.y) / (1.0 + (a * a - 1.0) * u.y));
    float sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
    return vec3(sin_t * cos(phi), sin_t * sin(phi), cos_t);
}
"#;

/// GLSL every vertex shader that draws the tree shares, so the tree, its shadow and
/// its wireframe all sway as one.
///
/// Each vertex arrives knowing, for the limb, the branch and the twigs it belongs to,
/// where that stem is attached and how far a point there swings (`a_wind1..3`: pivot in
/// xyz, weight in metres in w). Each order is bent about its own attachment, finest
/// first, and then the whole tree about its foot, so a twig rides its branch and the
/// branch its limb. A bend keeps only the part of a push square to the arm it acts on,
/// and the arm keeps its length, so wood swings rather than stretches — and a limb
/// pointing straight downwind is not shoved along its own length.
pub const WIND_GLSL: &str = r#"
uniform float u_time;
// xy: the way the wind blows, on the ground (x, z). z: strength, 1 a full gale.
// w: gustiness. A strength of zero is still air, and everything below is skipped.
uniform vec4 u_wind;
// How far the trunk, the limbs, the branches and the twigs bend in a full gale.
uniform vec4 u_wind_flex;
// x: how fast the trunk sways, in hertz. y: how far a leaf flutters, in radians.
uniform vec2 u_wind_motion;
uniform float u_tree_height;

const float WIND_TAU = 6.28318531;

float wind_hash(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
}

vec3 wind_down() { return vec3(u_wind.x, 0.0, u_wind.y); }
vec3 wind_across() { return vec3(-u_wind.y, 0.0, u_wind.x); }

// How hard the wind is blowing at `p` just now. Gusts are slow swells and lulls that
// travel downwind, so the near side of a crown takes one a moment before the far side.
float wind_strength(vec3 p) {
    float t = u_time - dot(p.xz, u_wind.xy) / 8.0;
    float g = 0.5 + 0.26 * sin(t * 0.53) + 0.16 * sin(t * 1.31 + 1.7) + 0.08 * sin(t * 3.1 + 0.4);
    return u_wind.z * max(1.0 + u_wind.w * (2.0 * g - 1.0), 0.0);
}

// A cantilever's deflection under an even load, 0 at the root and 1 at the tip.
float wind_cantilever(float x) {
    x = clamp(x, 0.0, 1.0);
    return x * x * (6.0 - 4.0 * x + x * x) / 3.0;
}

vec3 wind_bend(vec3 p, vec3 pivot, vec3 push) {
    vec3 arm = p - pivot;
    float len = length(arm);
    if (len < 1e-4) {
        return p;
    }
    return pivot + normalize(arm + push) * len;
}

// One order of wood: pushed downwind, bobbing up and down and swinging across, each
// stem on its own phase and at its own pace so no two limbs move in step.
vec3 wind_order(vec3 p, vec4 anchor, float flex, float pace) {
    if (anchor.w <= 0.0 || flex <= 0.0) {
        return p;
    }
    float h = wind_hash(anchor.xyz);
    float phase = h * WIND_TAU;
    float omega = WIND_TAU * u_wind_motion.x * pace * (0.8 + 0.4 * fract(h * 7.31));
    float t = u_time;
    vec3 push = wind_down() * (0.55 + 0.45 * sin(omega * t + phase))
        + vec3(0.0, 0.6 * sin(omega * 1.37 * t + phase * 1.9), 0.0)
        + wind_across() * (0.35 * sin(omega * 0.73 * t + phase * 2.7));
    return wind_bend(p, anchor.xyz, push * (wind_strength(anchor.xyz) * flex * anchor.w));
}

// The whole tree about its foot: a lean that follows the gusts, and a sway about it at
// the trunk's own pace that the turbulence keeps going.
vec3 wind_trunk(vec3 p) {
    float height = max(u_tree_height, 1.0);
    float w = height * wind_cantilever(p.y / height);
    if (w <= 0.0) {
        return p;
    }
    float omega = WIND_TAU * u_wind_motion.x;
    float t = u_time;
    float sway = u_wind.z * (0.2 + 0.4 * u_wind.w);
    vec3 push = wind_down() * (wind_strength(vec3(0.0)) + sway * sin(omega * t))
        + wind_across() * (sway * 0.35 * sin(omega * 0.81 * t + 1.1));
    return wind_bend(p, vec3(0.0), push * (u_wind_flex.x * w));
}

vec3 wind_displace(vec3 p, vec4 w1, vec4 w2, vec4 w3) {
    if (u_wind.z <= 0.0) {
        return p;
    }
    p = wind_order(p, w3, u_wind_flex.w, 4.6);
    p = wind_order(p, w2, u_wind_flex.z, 3.0);
    p = wind_order(p, w1, u_wind_flex.y, 1.9);
    return wind_trunk(p);
}

vec3 wind_rotate(vec3 v, vec3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

// A leaf card rides the point of the twig it hangs from, rigidly, and flutters about
// it. The normal turns with the card, which is what makes a canopy shimmer.
vec3 wind_leaf(vec3 pos, inout vec3 normal, vec3 origin, vec4 w1, vec4 w2, vec4 w3) {
    if (u_wind.z <= 0.0) {
        return pos;
    }
    vec3 local = pos - origin;
    if (u_wind_motion.y > 0.0) {
        float h = wind_hash(origin);
        float omega = WIND_TAU * (3.5 + 3.0 * h);
        float t = u_time;
        float angle = u_wind_motion.y * wind_strength(origin)
            * (0.7 * sin(omega * t + h * WIND_TAU) + 0.3 * sin(omega * 2.3 * t + h * 17.0));
        // Mostly about a level axis, so a leaf flaps and twists rather than spinning
        // flat, each on its own heading.
        float a = fract(h * 13.7) * WIND_TAU;
        vec3 axis = normalize(wind_across() * cos(a) + wind_down() * sin(a)
            + vec3(0.0, 0.35 * (fract(h * 5.3) - 0.5), 0.0));
        local = wind_rotate(local, axis, angle);
        normal = wind_rotate(normal, axis, angle);
    }
    return wind_displace(origin, w1, w2, w3) + local;
}
"#;

const MESH_VS_BODY: &str = r#"in vec3 a_pos;
in vec3 a_normal;
in vec2 a_uv;
in vec4 a_tangent;
in float a_weathering;
in vec4 a_wind1;
in vec4 a_wind2;
in vec4 a_wind3;
uniform mat4 u_view_proj;
uniform mat4 u_light_view_proj;
uniform float u_normal_bias;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
out vec4 v_tangent;
out vec4 v_shadow;
out float v_weathering;
void main() {
    vec3 pos = wind_displace(a_pos, a_wind1, a_wind2, a_wind3);
    v_world = pos;
    v_normal = a_normal;
    v_uv = a_uv;
    v_tangent = a_tangent;
    v_weathering = a_weathering;
    // Looked up a little along the normal rather than at the surface itself. With a
    // low sun the light grazes everything, and that is where plain depth bias either
    // stripes the bark with acne or lifts the shadow off its caster.
    v_shadow = u_light_view_proj * vec4(pos + a_normal * u_normal_bias, 1.0);
    gl_Position = u_view_proj * vec4(pos, 1.0);
}"#;

/// Every vertex shader that sways puts the wind ahead of its own body.
fn with_wind(body: &str) -> String {
    format!("#version 150
{WIND_GLSL}{body}")
}

/// A lit fragment shader: every shared piece, then its own body.
fn lit(body: &str) -> String {
    format!("#version 150
{COMMON_GLSL}{SKY_GLSL}{ENV_GLSL}{BRDF_GLSL}{SHADOW_GLSL}{body}")
}

pub fn mesh_vs() -> String {
    with_wind(MESH_VS_BODY)
}

const MESH_FS_BODY: &str = r#"in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_tangent;
in vec4 v_shadow;
in float v_weathering;
uniform vec3 u_albedo_color;
uniform vec3 u_dead_color;
uniform float u_dead_weathering;
uniform vec3 u_moss_color;
uniform float u_moss_height;
uniform float u_moss_amount;
uniform float u_bark_darken_low;
uniform float u_roughness;
uniform float u_metallic;
uniform int u_mode;
uniform int u_use_normal_map;
uniform sampler2D u_albedo_tex;
uniform sampler2D u_normal_tex;
uniform sampler2D u_rough_tex;
out vec4 out_color;

void main() {
    if (u_mode == 1) {
        float c = mod(floor(v_uv.x * 4.0) + floor(v_uv.y * 4.0), 2.0);
        out_color = vec4(mix(vec3(0.85, 0.85, 0.9), vec3(0.3, 0.55, 0.85), c), 1.0);
        return;
    }
    vec3 Ng = normalize(v_normal);
    vec3 N = Ng;
    if (u_use_normal_map == 1) {
        vec3 T = normalize(v_tangent.xyz - N * dot(N, v_tangent.xyz));
        vec3 B = cross(N, T) * v_tangent.w;
        vec3 nm = texture(u_normal_tex, v_uv).xyz * 2.0 - 1.0;
        N = normalize(T * nm.x + B * nm.y + N * nm.z);
    }
    vec3 V = normalize(u_cam_pos - v_world);
    if (u_mode == 2) {
        out_color = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }
    if (u_mode == 3) {
        out_color = vec4(vec3(screen_ao() * canopy_sky(v_world)), 1.0);
        return;
    }
    vec3 albedo = texture(u_albedo_tex, v_uv).rgb * u_albedo_color;

    // Bark is not one material. It weathers darker at the foot of a tree than high in
    // the crown, and it carries moss where damp sits: low down, and on surfaces that
    // face up rather than ones rain runs off. Both are read straight off world
    // position and the surface normal, so no extra mesh data is needed for either.
    albedo *= 1.0 - u_bark_darken_low * exp(-max(v_world.y, 0.0) * 0.35);
    // Dead wood bleaches toward one silver-grey whatever the living bark was. The
    // texture's own light and dark are kept as a modulation around that colour, so
    // the grain survives and only the hue and the tone move across.
    float dead = clamp(v_weathering * u_dead_weathering, 0.0, 1.0);
    if (dead > 0.0) {
        float grain = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
        vec3 weathered = u_dead_color * clamp(0.55 + 1.8 * grain, 0.35, 1.6);
        albedo = mix(albedo, weathered, dead);
    }
    if (u_moss_amount > 0.0) {
        float low = 1.0 - smoothstep(0.0, max(u_moss_height, 0.01), v_world.y);
        float facing = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
        float mottle = value_noise(v_world * 1.7) * 0.65 + value_noise(v_world * 6.0) * 0.35;
        float moss = u_moss_amount * low * facing * facing * smoothstep(0.35, 0.75, mottle);
        albedo = mix(albedo, u_moss_color, clamp(moss, 0.0, 1.0));
    }
    float rough = clamp(texture(u_rough_tex, v_uv).r * u_roughness, 0.045, 1.0);
    rough = filtered_roughness(rough, Ng);
    Surface s = surface(albedo, clamp(u_metallic, 0.0, 1.0), rough, N, V);

    float NoL = dot(Ng, u_sun_dir);
    float shadow = NoL > -0.05 ? soft_shadow(v_shadow, max(NoL, 0.0), 0.03, 16) : 0.0;
    vec3 color = direct(s, u_sun_dir) * u_sun_color * shadow
        + ambient(s, Ng, screen_ao() * canopy_sky(v_world));
    out_color = vec4(color * u_exposure, 1.0);
}"#;

pub fn mesh_fs() -> String {
    lit(MESH_FS_BODY)
}

/// Bark into a depth map, swayed exactly as the colour pass sways it, or the tree
/// would move through a shadow that stands still. The same program lays down the
/// camera's depth before the colour pass, for the ambient occlusion to read.
const DEPTH_VS_BODY: &str = r#"in vec3 a_pos;
in vec4 a_wind1;
in vec4 a_wind2;
in vec4 a_wind3;
uniform mat4 u_light_view_proj;
void main() {
    gl_Position = u_light_view_proj * vec4(wind_displace(a_pos, a_wind1, a_wind2, a_wind3), 1.0);
}"#;

pub fn depth_vs() -> String {
    with_wind(DEPTH_VS_BODY)
}

pub const DEPTH_FS: &str = r#"#version 150
out vec4 out_color;
void main() {
    out_color = vec4(1.0);
}"#;

/// The wireframe, swayed with what it outlines. Bark and leaf cards both come
/// through here, told apart by `u_leaf`, because a card rides its origin rather than
/// bending where its corners happen to be.
const COLOR_VS_BODY: &str = r#"in vec3 a_pos;
in vec4 a_wind1;
in vec4 a_wind2;
in vec4 a_wind3;
in vec3 a_leaf_origin;
uniform mat4 u_view_proj;
uniform int u_leaf;
void main() {
    vec3 pos;
    if (u_leaf == 1) {
        vec3 normal = vec3(0.0, 1.0, 0.0);
        pos = wind_leaf(a_pos, normal, a_leaf_origin, a_wind1, a_wind2, a_wind3);
    } else {
        pos = wind_displace(a_pos, a_wind1, a_wind2, a_wind3);
    }
    gl_Position = u_view_proj * vec4(pos, 1.0);
}"#;

pub fn color_vs() -> String {
    with_wind(COLOR_VS_BODY)
}

pub const COLOR_FS: &str = r#"#version 150
uniform vec4 u_color;
out vec4 out_color;
void main() {
    out_color = u_color;
}"#;

pub const LINES_VS: &str = r#"#version 150
in vec3 a_pos;
in vec3 a_col;
uniform mat4 u_mvp;
out vec3 v_col;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_col = a_col;
}"#;

pub const LINES_FS: &str = r#"#version 150
in vec3 v_col;
out vec4 out_color;
void main() {
    out_color = vec4(v_col, 1.0);
}"#;

const LEAF_VS_BODY: &str = r#"in vec3 a_pos;
in vec3 a_normal;
in vec2 a_uv;
in vec4 a_tint;
// Offset down the atlas to the cluster arrangement this card draws.
in float a_atlas_v;
in vec4 a_wind1;
in vec4 a_wind2;
in vec4 a_wind3;
// The point of its twig the card hangs from, which it rides and flutters about.
in vec3 a_leaf_origin;
uniform mat4 u_view_proj;
uniform mat4 u_light_view_proj;
uniform float u_normal_bias;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_card_uv;
out vec4 v_tint;
out vec4 v_shadow;
out float v_atlas_v;
void main() {
    vec3 normal = a_normal;
    vec3 pos = wind_leaf(a_pos, normal, a_leaf_origin, a_wind1, a_wind2, a_wind3);
    v_world = pos;
    v_normal = normal;
    v_card_uv = a_uv;
    v_tint = a_tint;
    v_atlas_v = a_atlas_v;
    v_shadow = u_light_view_proj * vec4(pos + normal * u_normal_bias, 1.0);
    gl_Position = u_view_proj * vec4(pos, 1.0);
}"#;

pub fn leaf_vs() -> String {
    with_wind(LEAF_VS_BODY)
}

const LEAF_FS_BODY: &str = r#"in vec3 v_world;
in vec3 v_normal;
in vec2 v_card_uv;
in vec4 v_tint;
in vec4 v_shadow;
in float v_atlas_v;
uniform vec2 u_atlas_scale;
uniform vec2 u_atlas_front;
uniform vec2 u_atlas_back;
uniform float u_alpha_cutoff;
// Mip level at which coverage starts giving way to a hard cutoff.
uniform float u_coverage_lod;
// How far the soft cutout edge is sharpened toward a one-pixel one.
uniform float u_edge_sharpness;
// How far the stored normal leans toward the outside of the crown, and how much of
// that lean a card seen from behind keeps.
uniform float u_normal_blend;
uniform float u_backface_volume;
// How much of the sun a leaf in shadow loses: 1 is all of it.
uniform float u_self_shadow;
uniform float u_translucency;
uniform int u_mode;
uniform sampler2D u_albedo_tex;
uniform sampler2D u_rough_tex;
out vec4 out_color;

void main() {
    // A leaf is one quad seen from both sides: the lit face and the underside are
    // different cells of the same atlas, picked per fragment. Which arrangement of
    // the cluster it reads is the card's own business and comes down the pipe with it,
    // so one canopy is not one motif repeated everywhere.
    vec2 cell = gl_FrontFacing ? u_atlas_front : u_atlas_back;
    vec2 uv = cell + vec2(0.0, v_atlas_v) + v_card_uv * u_atlas_scale;
    vec4 tex = texture(u_albedo_tex, uv);

    // While a card is bigger than a pixel its filtered alpha *is* its coverage, and
    // handing that straight to alpha-to-coverage gives a properly soft cutout edge.
    //
    // Under minification that stops being true, for two reasons at once. A leaf fills
    // only part of its atlas cell, so the average falls toward that fraction however
    // dense the canopy really is. And alpha-to-coverage derives its sample mask from
    // the coverage value alone, so cards stacked over one pixel pick the same samples
    // and never accumulate: the nearest wins and the rest add nothing. Together they
    // wash a distant canopy out to the sky behind it. Past that point the cutoff is
    // what holds the canopy together, and the mip chain keeps the share of texels
    // passing it constant so the density stays put.
    vec2 texels = vec2(textureSize(u_albedo_tex, 0)) * u_atlas_scale;
    vec2 du = dFdx(v_card_uv * texels);
    vec2 dv = dFdy(v_card_uv * texels);
    float lod = 0.5 * log2(max(dot(du, du), dot(dv, dv)) + 1e-8);
    float snap = clamp((lod - u_coverage_lod) * 0.5, 0.0, 1.0);
    float coverage = mix(tex.a, step(u_alpha_cutoff, tex.a), snap);
    // Fine art — needles a texel or two wide — averages into the air around it a few
    // mips down, and handing that straight to coverage draws the strands as smudges.
    // Rescaling alpha about the cutoff by its own screen-space rate of change puts the
    // edge back to about one pixel wide at any mip, still anti-aliased by the samples,
    // and the coverage-preserving mip chain is what keeps the density honest.
    if (u_edge_sharpness > 0.0) {
        float sharp = clamp((tex.a - u_alpha_cutoff) / max(fwidth(tex.a), 1e-4) + 0.5, 0.0, 1.0);
        coverage = mix(coverage, sharp, u_edge_sharpness * (1.0 - snap));
    }
    if (coverage < 1.0 / 255.0) {
        discard;
    }
    if (u_mode == 1) {
        float c = mod(floor(v_card_uv.x * 4.0) + floor(v_card_uv.y * 4.0), 2.0);
        out_color = vec4(mix(vec3(0.2, 0.5, 0.2), vec3(0.85, 0.9, 0.5), c), 1.0);
        return;
    }
    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing) {
        // Turning the whole normal round also turns the crown's outward lean inward,
        // which shades a card seen from behind as though it were buried. A volume of
        // needles has no back, so it turns only the card's own flat share of the
        // normal; the face comes from the derivatives, and its sign does not matter
        // because it is used twice.
        vec3 face = normalize(cross(dFdx(v_world), dFdy(v_world)));
        vec3 kept = normalize(N - 2.0 * (1.0 - u_normal_blend) * dot(N, face) * face);
        N = normalize(mix(-N, kept, u_backface_volume));
    }
    if (u_mode == 2) {
        out_color = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }
    if (u_mode == 3) {
        out_color = vec4(vec3(screen_ao() * v_tint.a), coverage);
        return;
    }

    vec3 albedo = tex.rgb * v_tint.rgb;
    vec3 V = normalize(u_cam_pos - v_world);
    vec3 L = u_sun_dir;
    // The source map calls a leaf glossy, around 0.29, and at that roughness every
    // blade mirrors the sky. The occlusion below takes most of that away from the
    // leaves inside the crown, but not all, so foliage keeps a matte floor.
    float rough = filtered_roughness(clamp(texture(u_rough_tex, uv).r, 0.5, 1.0), N);
    Surface s = surface(albedo, 0.0, rough, N, V);
    // A card stands for a tuft of blades, not one flat mirror, and its normal is bent
    // toward the outside of the crown. The grazing angle it reports is not one any
    // real blade meets the eye at, and Fresnel would turn every card seen edge-on into
    // a sheet of reflected sky. Held away from grazing, the sky is a sheen instead.
    s.NoV = max(s.NoV, 0.4);
    vec2 tuft = env_dfg(s.NoV, rough);
    s.E = mix(tuft.xxx, tuft.yyy, s.f0);

    // A needle crown lets light through in a thousand gaps, so shadow on foliage is
    // not the yes-or-no the shadow map says.
    float NoL = dot(N, L);
    float shadow = mix(1.0, soft_shadow(v_shadow, abs(NoL), 0.08, 8), u_self_shadow);

    // A leaf is a thin sheet that scatters light out of both faces: reflected back
    // off the lit face and transmitted through to the other, each close to Lambert.
    // The card also stands for a tuft of blades turned every which way, which is what
    // the wrap is for: the terminator across a tuft is soft, not a line. Energy-
    // conserving, so the wrap does not add light that was never there.
    float wrap = 0.35;
    float reflected = max((NoL + wrap) / ((1.0 + wrap) * (1.0 + wrap)), 0.0);
    // Light through the blade from behind. Most of it keeps going the way it came, so
    // the view looking into the sun sees the brightest of it.
    float through = max(-NoL, 0.0);
    float forward = pow(saturate(dot(V, -L)), 4.0);
    float transmitted = u_translucency * through * (0.5 + 2.0 * forward);
    vec3 sun = u_sun_color * shadow;
    vec3 color = (albedo / PI * (reflected + transmitted) + specular_lobe(s, L, saturate(NoL)) * saturate(NoL)) * sun;

    // Everything but the sun. The lit face gathers the environment it faces; the far
    // face gathers the one behind and passes on the translucent share of it. Leaves
    // buried in the crown see less of either, by the crown depth the card was built
    // with and by what the screen can see crowding round them.
    float visibility = screen_ao() * v_tint.a;
    vec3 gathered = irradiance(N) * (1.0 - s.E) + irradiance(-N) * (0.5 * u_translucency);
    // The reflection is of one direction of sky, which the rest of the crown is far
    // likelier to block than the whole dome the diffuse gathers, so it is occluded
    // harder.
    vec3 spec = s.E * prefiltered_radiance(reflect(-V, N), rough) * s.energy
        * specular_occlusion(s.NoV, visibility * visibility, s.alpha);
    color += albedo * gathered * multi_bounce(visibility, albedo) + spec;
    out_color = vec4(color * u_exposure, coverage);
}"#;

pub fn leaf_fs() -> String {
    lit(LEAF_FS_BODY)
}

const LEAF_DEPTH_VS_BODY: &str = r#"in vec3 a_pos;
in vec3 a_normal;
in vec2 a_uv;
in float a_atlas_v;
in vec4 a_wind1;
in vec4 a_wind2;
in vec4 a_wind3;
in vec3 a_leaf_origin;
uniform mat4 u_light_view_proj;
out vec2 v_card_uv;
out float v_atlas_v;
void main() {
    v_card_uv = a_uv;
    v_atlas_v = a_atlas_v;
    vec3 normal = a_normal;
    vec3 pos = wind_leaf(a_pos, normal, a_leaf_origin, a_wind1, a_wind2, a_wind3);
    gl_Position = u_light_view_proj * vec4(pos, 1.0);
}"#;

pub fn leaf_depth_vs() -> String {
    with_wind(LEAF_DEPTH_VS_BODY)
}

pub const LEAF_DEPTH_FS: &str = r#"#version 150
in vec2 v_card_uv;
in float v_atlas_v;
uniform vec2 u_atlas_scale;
uniform vec2 u_atlas_front;
uniform vec2 u_atlas_back;
uniform float u_alpha_cutoff;
uniform sampler2D u_albedo_tex;
out vec4 out_color;
void main() {
    // Without the same alpha test the colour pass uses, every leaf would cast the
    // shadow of its bounding quad. It has to read the card's own arrangement too, and
    // the face the colour pass would draw from this side, or a card casts the shadow
    // of a cluster it is not drawing.
    vec2 cell = gl_FrontFacing ? u_atlas_front : u_atlas_back;
    vec2 uv = cell + vec2(0.0, v_atlas_v) + v_card_uv * u_atlas_scale;
    if (texture(u_albedo_tex, uv).a < u_alpha_cutoff) {
        discard;
    }
    out_color = vec4(1.0);
}"#;

/// Leaf cards seen from straight above, each taking its share of what light is left:
/// drawn with a multiplying blend over white, what remains is the sky the crown lets
/// through to the ground.
pub const CANOPY_FS: &str = r#"#version 150
in vec2 v_card_uv;
in float v_atlas_v;
uniform vec2 u_atlas_scale;
uniform vec2 u_atlas_front;
uniform vec2 u_atlas_back;
uniform sampler2D u_albedo_tex;
out vec4 out_color;
void main() {
    vec2 cell = gl_FrontFacing ? u_atlas_front : u_atlas_back;
    float a = texture(u_albedo_tex, cell + vec2(0.0, v_atlas_v) + v_card_uv * u_atlas_scale).a;
    out_color = vec4(0.0, 0.0, 0.0, a);
}"#;

/// One axis of a Gaussian blur, read from a given mip of the source.
pub const CANOPY_BLUR_FS: &str = r#"#version 150
in vec2 v_uv;
uniform sampler2D u_src;
uniform float u_lod;
// One texel along the blur's axis, in uv, and the blur's width in texels.
uniform vec2 u_step;
uniform float u_sigma;
out vec4 out_color;
void main() {
    int r = int(min(ceil(u_sigma * 2.5), 24.0));
    float sum = 0.0;
    float weight = 0.0;
    for (int i = -r; i <= r; ++i) {
        float g = exp(-0.5 * float(i * i) / (u_sigma * u_sigma));
        sum += textureLod(u_src, v_uv + u_step * float(i), u_lod).r * g;
        weight += g;
    }
    float v = sum / weight;
    out_color = vec4(v, v, v, 1.0);
}"#;

/// One triangle over the whole target, from nothing but `gl_VertexID`, for every
/// full-screen pass. `v_uv` runs 0 to 1 across the target, `v_ndc` -1 to 1.
pub const FULLSCREEN_VS: &str = r#"#version 150
out vec2 v_uv;
out vec2 v_ndc;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = p;
    v_ndc = p * 2.0 - 1.0;
    gl_Position = vec4(v_ndc, 1.0, 1.0);
}"#;

const BACKDROP_FS_BODY: &str = r#"in vec2 v_ndc;
uniform mat4 u_inv_view_proj;
out vec4 out_color;

void main() {
    // The view ray is rebuilt from the inverse view-projection, so the sky sits in the
    // world rather than on the screen.
    vec4 far = u_inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(far.xyz / far.w - u_cam_pos);
    out_color = vec4(min(background(dir) * u_exposure, vec3(60000.0)), 1.0);
}"#;

/// Whatever is behind the tree: the procedural sky or a photograph.
pub fn backdrop_fs() -> String {
    lit(BACKDROP_FS_BODY)
}

/// A ground plane, so the tree casts onto something.
pub const GROUND_VS: &str = r#"#version 150
in vec3 a_pos;
uniform mat4 u_view_proj;
uniform mat4 u_light_view_proj;
uniform float u_normal_bias;
out vec3 v_world;
out vec4 v_shadow;
void main() {
    v_world = a_pos;
    v_shadow = u_light_view_proj * vec4(a_pos + vec3(0.0, u_normal_bias, 0.0), 1.0);
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
}"#;

const GROUND_FS_BODY: &str = r#"in vec3 v_world;
in vec4 v_shadow;
uniform vec3 u_albedo_color;
uniform float u_fade_start;
uniform float u_fade_end;
// 0: a plain ground of its own, lit like the tree. 1: the ground in the photograph,
// laid under the tree and darkened where the tree takes light off it.
uniform int u_ground_mode;
// 3 shows the ambient occlusion alone, as the tree does in that view.
uniform int u_mode;
out vec4 out_color;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

void main() {
    vec3 N = vec3(0.0, 1.0, 0.0);
    vec3 V = normalize(u_cam_pos - v_world);
    vec3 view = -V;
    float NoL = max(u_sun_dir.y, 0.0);
    float shadow = NoL > 0.0 ? soft_shadow(v_shadow, NoL, 0.02, 16) : 0.0;
    float ao = screen_ao() * canopy_sky(v_world);
    if (u_mode == 3) {
        out_color = vec4(vec3(ao), 1.0);
        return;
    }
    vec3 color;
    if (u_ground_mode == 1) {
        // A shadow catcher. The photograph already shows this ground lit by its own
        // sky and sun, so rather than lighting it again, it keeps the share of that
        // light the tree leaves it: the sky's less what the tree occludes, the sun's
        // only where the tree does not shade it. Past the edge of the dome's floor
        // the backdrop shows its wall, and so does this.
        float floor_hit;
        vec3 photographed = background_at(view, floor_hit);
        vec3 sky = irradiance(N) * PI;
        vec3 sun = u_sun_color * NoL;
        vec3 kept = (sky * ao + sun * shadow) / max(sky + sun, vec3(1e-4));
        color = photographed * mix(vec3(1.0), kept, floor_hit);
    } else {
        float grain = vnoise(v_world.xz * 0.7) * 0.35 + vnoise(v_world.xz * 0.11) * 0.65;
        vec3 albedo = u_albedo_color * (0.84 + 0.32 * grain);
        Surface s = surface(albedo, 0.0, 0.9, N, V);
        color = direct(s, u_sun_dir) * u_sun_color * shadow + ambient(s, N, ao);
        // Dissolve into what lies behind at range, so the plane has no visible rim.
        float dist = length(v_world.xz - u_cam_pos.xz);
        color = mix(color, background(view), smoothstep(u_fade_start, u_fade_end, dist));
    }
    out_color = vec4(min(color * u_exposure, vec3(60000.0)), 1.0);
}"#;

pub fn ground_fs() -> String {
    lit(GROUND_FS_BODY)
}

/// Renders an environment into one face of a cube map: the procedural sky, or a
/// photograph from its equirectangular image.
const CAPTURE_FS_BODY: &str = r#"in vec2 v_uv;
uniform int u_face;
// 0 the procedural sky, 1 the photograph.
uniform int u_source;
uniform sampler2D u_source_equirect;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
out vec4 out_color;

void main() {
    vec3 d = normalize(cube_dir(u_face, v_uv * 2.0 - 1.0));
    vec3 c = u_source == 0
        ? sky_radiance(d, u_sun_dir, u_sun_color)
        : textureLod(u_source_equirect, equirect_uv(d), 0.0).rgb;
    out_color = vec4(c, 1.0);
}"#;

pub fn capture_fs() -> String {
    format!("#version 150
{COMMON_GLSL}{SKY_GLSL}{CAPTURE_FS_BODY}")
}

/// One mip of the prefiltered environment: the source convolved with the GGX lobe of
/// that mip's roughness, taking N = V as the split sum does.
///
/// Filtered importance sampling (Křivánek and Colbert): each sample reads the source
/// at the mip whose texels are the size of the solid angle the sample stands for, so a
/// couple of hundred samples come out as smooth as thousands.
const PREFILTER_FS_BODY: &str = r#"in vec2 v_uv;
uniform samplerCube u_source;
uniform float u_source_size;
uniform int u_face;
uniform float u_perceptual;
uniform int u_samples;
uniform float u_target_size;
out vec4 out_color;

void main() {
    vec3 N = normalize(cube_dir(u_face, v_uv * 2.0 - 1.0));
    if (u_perceptual <= 0.0) {
        float lod = max(log2(u_source_size / u_target_size), 0.0);
        out_color = vec4(textureLod(u_source, N, lod).rgb, 1.0);
        return;
    }
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);
    float a = u_perceptual * u_perceptual;
    float a2 = a * a;
    float texel_omega = 4.0 * PI / (6.0 * u_source_size * u_source_size);
    vec3 sum = vec3(0.0);
    float weight = 0.0;
    for (int i = 0; i < u_samples; ++i) {
        vec3 h = importance_ggx(hammersley(i, u_samples), a);
        vec3 H = T * h.x + B * h.y + N * h.z;
        vec3 L = 2.0 * dot(N, H) * H - N;
        float NoL = dot(N, L);
        if (NoL > 0.0) {
            // With V = N the pdf of L is D / 4.
            float f = (h.z * a2 - h.z) * h.z + 1.0;
            float pdf = a2 / (PI * f * f) * 0.25;
            float sample_omega = 1.0 / (float(u_samples) * pdf + 1e-6);
            float lod = max(0.5 * log2(sample_omega / texel_omega) + 1.0, 0.0);
            sum += textureLod(u_source, L, lod).rgb * NoL;
            weight += NoL;
        }
    }
    out_color = vec4(sum / max(weight, 1e-4), 1.0);
}"#;

pub fn prefilter_fs() -> String {
    format!("#version 150
{COMMON_GLSL}{SAMPLING_GLSL}{PREFILTER_FS_BODY}")
}

/// Filament's DFG table: the split sum's scale (x, the Fresnel-weighted part) and
/// total (y) for GGX with height-correlated Smith, by NoV across and perceptual
/// roughness up. Surfaces read `mix(x, y, f0)` for their directional albedo.
const DFG_FS_BODY: &str = r#"in vec2 v_uv;
out vec4 out_color;

float visibility(float NoV, float NoL, float a) {
    float a2 = a * a;
    float lv = NoL * sqrt((NoV - NoV * a2) * NoV + a2);
    float ll = NoV * sqrt((NoL - NoL * a2) * NoL + a2);
    return 0.5 / max(lv + ll, 1e-6);
}

void main() {
    float NoV = v_uv.x;
    float a = v_uv.y * v_uv.y;
    vec3 V = vec3(sqrt(1.0 - NoV * NoV), 0.0, NoV);
    const int N = 512;
    vec2 r = vec2(0.0);
    for (int i = 0; i < N; ++i) {
        vec3 H = importance_ggx(hammersley(i, N), a);
        vec3 L = 2.0 * dot(V, H) * H - V;
        float VoH = saturate(dot(V, H));
        float NoL = saturate(L.z);
        float NoH = saturate(H.z);
        if (NoL > 0.0) {
            float v = visibility(NoV, NoL, a) * NoL * (VoH / max(NoH, 1e-5));
            float Fc = pow(1.0 - VoH, 5.0);
            r.x += v * Fc;
            r.y += v;
        }
    }
    out_color = vec4(r * (4.0 / float(N)), 0.0, 1.0);
}"#;

pub fn dfg_fs() -> String {
    format!("#version 150
{COMMON_GLSL}{SAMPLING_GLSL}{DFG_FS_BODY}")
}

/// Ground-truth-based ambient occlusion (Jimenez et al. 2016), from the camera's depth
/// alone. Each pixel looks along a few directions across the screen for the highest
/// horizon on either side, and integrates how much of the cosine-weighted hemisphere
/// over its normal those horizons leave open.
const GTAO_FS_BODY: &str = r#"in vec2 v_uv;
uniform sampler2D u_depth;
uniform mat4 u_inv_proj;
uniform vec2 u_texel;
// World radius the occlusion looks out to, and pixels per metre at a metre away.
uniform float u_radius;
uniform float u_proj_scale;
uniform float u_power;
out vec4 out_color;

vec3 view_pos(vec2 uv) {
    float d = textureLod(u_depth, uv, 0.0).r;
    vec4 p = u_inv_proj * vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    return p.xyz / p.w;
}

// A 4x4 ordered pattern, so the blur after this can take it out exactly.
float bayer4(vec2 p) {
    ivec2 i = ivec2(mod(p, 4.0));
    int lo = 2 * ((i.x ^ i.y) & 1) + (i.y & 1);
    int hi = 2 * (((i.x ^ i.y) >> 1) & 1) + ((i.y >> 1) & 1);
    return float(4 * lo + hi) / 16.0;
}

void main() {
    float depth = textureLod(u_depth, v_uv, 0.0).r;
    if (depth >= 1.0) {
        out_color = vec4(1.0);
        return;
    }
    vec3 P = view_pos(v_uv);
    vec3 V = normalize(-P);
    // The normal from whichever neighbour on each axis is nearer in depth, so the
    // edge of a leaf in front of the sky does not bend it.
    vec3 pr = view_pos(v_uv + vec2(u_texel.x, 0.0));
    vec3 pl = view_pos(v_uv - vec2(u_texel.x, 0.0));
    vec3 pu = view_pos(v_uv + vec2(0.0, u_texel.y));
    vec3 pd = view_pos(v_uv - vec2(0.0, u_texel.y));
    vec3 dx = abs(pr.z - P.z) < abs(P.z - pl.z) ? pr - P : P - pl;
    vec3 dy = abs(pu.z - P.z) < abs(P.z - pd.z) ? pu - P : P - pd;
    vec3 N = normalize(cross(dx, dy));
    if (dot(N, V) < 0.0) {
        N = -N;
    }

    float radius_px = min(u_radius * u_proj_scale / max(-P.z, 1e-3), 160.0);
    if (radius_px < 1.0) {
        out_color = vec4(1.0);
        return;
    }
    float spin = bayer4(gl_FragCoord.xy);
    float jitter = bayer4(gl_FragCoord.yx + vec2(1.0, 2.0));
    // Past this distance a sample stops counting, falling off from 40% of the way out.
    float falloff_from = u_radius * 0.4;
    float falloff_range = u_radius - falloff_from;

    const int SLICES = 3;
    const int STEPS = 6;
    float visible = 0.0;
    for (int slice = 0; slice < SLICES; ++slice) {
        float phi = (float(slice) + spin) * PI / float(SLICES);
        vec2 omega = vec2(cos(phi), sin(phi));
        vec3 dir = vec3(omega, 0.0);
        vec3 ortho = dir - dot(dir, V) * V;
        vec3 axis = normalize(cross(dir, V));
        vec3 pn = N - axis * dot(N, axis);
        float pn_len = max(length(pn), 1e-4);
        float cos_n = saturate(dot(pn, V) / pn_len);
        float n = sign(dot(ortho, pn)) * acos(cos_n);

        float hc_pos = -1.0;
        float hc_neg = -1.0;
        for (int k = 0; k < STEPS; ++k) {
            float t = (float(k) + jitter) / float(STEPS);
            vec2 offset = omega * (t * t * radius_px + 1.0) * u_texel;
            vec3 s0 = view_pos(v_uv + offset) - P;
            vec3 s1 = view_pos(v_uv - offset) - P;
            float l0 = length(s0);
            float l1 = length(s1);
            float w0 = saturate((u_radius - l0) / falloff_range);
            float w1 = saturate((u_radius - l1) / falloff_range);
            hc_pos = max(hc_pos, mix(-1.0, dot(s0, V) / max(l0, 1e-5), w0));
            hc_neg = max(hc_neg, mix(-1.0, dot(s1, V) / max(l1, 1e-5), w1));
        }
        float h0 = n + clamp(-acos(hc_neg) - n, -PI * 0.5, PI * 0.5);
        float h1 = n + clamp(acos(hc_pos) - n, -PI * 0.5, PI * 0.5);
        float arc0 = (cos_n + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) * 0.25;
        float arc1 = (cos_n + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) * 0.25;
        visible += pn_len * (arc0 + arc1);
    }
    float v = pow(saturate(visible / float(SLICES)), u_power);
    out_color = vec4(v, v, v, 1.0);
}"#;

pub fn gtao_fs() -> String {
    format!("#version 150
{COMMON_GLSL}{GTAO_FS_BODY}")
}

/// The occlusion averaged over the 4x4 block its pattern repeats in, skipping anything
/// at a different depth, so the pattern goes and edges stay.
const AO_BLUR_FS_BODY: &str = r#"in vec2 v_uv;
uniform sampler2D u_ao;
uniform sampler2D u_depth;
uniform mat4 u_inv_proj;
uniform vec2 u_texel;
out vec4 out_color;

float view_z(vec2 uv) {
    float d = textureLod(u_depth, uv, 0.0).r;
    vec4 p = u_inv_proj * vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    return -p.z / p.w;
}

void main() {
    float z = view_z(v_uv);
    float sum = 0.0;
    float weight = 0.0;
    for (int y = -2; y < 2; ++y) {
        for (int x = -2; x < 2; ++x) {
            vec2 uv = v_uv + vec2(float(x), float(y)) * u_texel;
            float w = max(1.0 - abs(view_z(uv) - z) / (0.04 * z + 0.05), 0.0) + 1e-4;
            sum += textureLod(u_ao, uv, 0.0).r * w;
            weight += w;
        }
    }
    float v = sum / weight;
    out_color = vec4(v, v, v, 1.0);
}"#;

pub fn ao_blur_fs() -> String {
    format!("#version 150
{COMMON_GLSL}{AO_BLUR_FS_BODY}")
}

/// One step down the bloom chain: Jimenez's 13-tap filter, which does not flicker as
/// a bright pixel crosses the texel grid. The first step weighs each group of taps by
/// its brightness (Karis), so one blazing pixel cannot fill the whole chain.
pub const BLOOM_DOWN_FS: &str = r#"#version 150
in vec2 v_uv;
uniform sampler2D u_src;
uniform vec2 u_src_texel;
uniform int u_first;
out vec4 out_color;

vec3 at(float x, float y) {
    return textureLod(u_src, v_uv + vec2(x, y) * u_src_texel, 0.0).rgb;
}

float karis(vec3 c) {
    return 1.0 / (1.0 + dot(c, vec3(0.2126, 0.7152, 0.0722)));
}

void main() {
    vec3 a = at(-2.0, 2.0), b = at(0.0, 2.0), c = at(2.0, 2.0);
    vec3 d = at(-2.0, 0.0), e = at(0.0, 0.0), f = at(2.0, 0.0);
    vec3 g = at(-2.0, -2.0), h = at(0.0, -2.0), i = at(2.0, -2.0);
    vec3 j = at(-1.0, 1.0), k = at(1.0, 1.0), l = at(-1.0, -1.0), m = at(1.0, -1.0);
    vec3 result;
    if (u_first == 1) {
        vec3 g0 = (a + b + d + e) * 0.25;
        vec3 g1 = (b + c + e + f) * 0.25;
        vec3 g2 = (d + e + g + h) * 0.25;
        vec3 g3 = (e + f + h + i) * 0.25;
        vec3 g4 = (j + k + l + m) * 0.25;
        float w0 = karis(g0) * 0.125, w1 = karis(g1) * 0.125, w2 = karis(g2) * 0.125;
        float w3 = karis(g3) * 0.125, w4 = karis(g4) * 0.5;
        result = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    } else {
        result = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625
            + (j + k + l + m) * 0.125;
    }
    out_color = vec4(max(result, vec3(0.0)), 1.0);
}"#;

/// One step back up: a 3x3 tent over the smaller mip, added onto the larger one.
pub const BLOOM_UP_FS: &str = r#"#version 150
in vec2 v_uv;
uniform sampler2D u_src;
uniform vec2 u_src_texel;
out vec4 out_color;

vec3 at(float x, float y) {
    return textureLod(u_src, v_uv + vec2(x, y) * u_src_texel, 0.0).rgb;
}

void main() {
    vec3 c = at(0.0, 0.0) * 4.0
        + (at(-1.0, 0.0) + at(1.0, 0.0) + at(0.0, -1.0) + at(0.0, 1.0)) * 2.0
        + at(-1.0, -1.0) + at(1.0, -1.0) + at(-1.0, 1.0) + at(1.0, 1.0);
    out_color = vec4(c / 16.0, 1.0);
}"#;

/// The last pass: bloom mixed in, the scene's radiance fitted to the display, and
/// dithered on the way into eight bits.
const COMPOSITE_FS_BODY: &str = r#"in vec2 v_uv;
uniform sampler2D u_hdr;
uniform sampler2D u_bloom;
uniform float u_bloom_strength;
// 0 AgX, 1 AgX punchy, 2 Khronos PBR neutral, 3 ACES (fitted).
uniform int u_tonemap;
// The debug views write display colours straight out, and are passed through.
uniform int u_raw;
out vec4 out_color;

// AgX (Troy Sobotka), as Blender uses it by default, in Benjamin Wrensch's fit. Works in
// log space over a wide-gamut inset, so a bright saturated colour runs toward white
// the way film does rather than clipping to a flat primary.
vec3 agx_curve(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x
        + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 agx(vec3 c, bool punchy) {
    const mat3 inset = mat3(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const mat3 outset = mat3(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    const float min_ev = -12.47393;
    const float max_ev = 4.026069;
    c = inset * max(c, vec3(1e-10));
    c = clamp(log2(c), min_ev, max_ev);
    c = agx_curve((c - min_ev) / (max_ev - min_ev));
    if (punchy) {
        c = pow(max(c, vec3(0.0)), vec3(1.35));
        float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c = luma + 1.4 * (c - luma);
    }
    // Out of the inset again. The curve's output is already display-encoded.
    return clamp(outset * c, 0.0, 1.0);
}

vec3 pbr_neutral(vec3 color) {
    const float start = 0.8 - 0.04;
    const float desaturation = 0.15;
    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < start) {
        return color;
    }
    const float d = 1.0 - start;
    float new_peak = 1.0 - d * d / (peak + d - start);
    color *= new_peak / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(color, vec3(new_peak), g);
}

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec3 srgb_encode(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

void main() {
    vec3 c = textureLod(u_hdr, v_uv, 0.0).rgb;
    if (u_raw == 1) {
        out_color = vec4(c, 1.0);
        return;
    }
    c = mix(c, textureLod(u_bloom, v_uv, 0.0).rgb, u_bloom_strength);
    if (u_tonemap == 0 || u_tonemap == 1) {
        c = agx(c, u_tonemap == 1);
    } else if (u_tonemap == 2) {
        c = srgb_encode(pbr_neutral(c));
    } else {
        c = srgb_encode(aces(c));
    }
    // Half a step of noise either way, so a sky gradient does not band into steps.
    c += (ign(gl_FragCoord.xy) - 0.5) / 255.0;
    out_color = vec4(c, 1.0);
}"#;

pub fn composite_fs() -> String {
    format!("#version 150
{COMMON_GLSL}{COMPOSITE_FS_BODY}")
}
