/// GLSL every shader that answers to the sky shares: the dome, the ambient it throws,
/// and the tonemapping and exposure that fit a whole day into a display. Kept in one
/// place because four copies of a spherical-harmonic evaluator would be four chances
/// for the renderer to disagree with itself about what the light is doing.
pub const SKY_GLSL: &str = r#"
const float PI = 3.14159265359;

// The dome: a pale band at the horizon under a colder zenith, plus the bounce coming
// back up off the ground. The colours arrive already built from the sun's angle, so
// this shape is all that is left to do here.
uniform vec3 u_sky_zenith;
uniform vec3 u_sky_horizon;
uniform vec3 u_ground_bounce;
// That same dome projected into spherical harmonics, and the exposure that goes with
// the time of day it describes.
uniform vec3 u_sh[9];
uniform float u_exposure;

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

vec3 sky_color(vec3 dir) {
    float h = clamp(dir.y, -1.0, 1.0);
    vec3 dome = mix(u_sky_horizon, u_sky_zenith, pow(max(h, 0.0), 0.42));
    return mix(u_ground_bounce, dome, smoothstep(-0.25, 0.03, h));
}

// Ambient arriving at a surface facing n, divided by PI so a surface multiplies it by
// albedo and nothing else.
//
// Ramamoorthi and Hanrahan's closed form for a cosine-convolved dome. Unlike taking a
// fixed fraction of whatever the surface happens to face, this integrates the whole
// sky: a face turned down gets the warm ground bounce, a face turned up gets the cold
// zenith, and both follow the time of day without a second set of numbers to tune.
vec3 sky_ambient(vec3 n) {
    const float c1 = 0.429043 / PI;
    const float c2 = 0.511664 / PI;
    const float c3 = 0.743125 / PI;
    const float c4 = 0.886227 / PI;
    const float c5 = 0.247708 / PI;
    float x = n.x, y = n.y, z = n.z;
    vec3 e = u_sh[0] * c4 - u_sh[6] * c5
        + (u_sh[3] * x + u_sh[1] * y + u_sh[2] * z) * (2.0 * c2)
        + u_sh[6] * (c3 * z * z)
        + (u_sh[4] * x * y + u_sh[7] * x * z + u_sh[5] * y * z) * (2.0 * c1)
        + u_sh[8] * (c1 * (x * x - y * y));
    return max(e, vec3(0.0));
}

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// Everything on its way to the framebuffer goes through here, so a scene lit by a
// midday sun and one lit by a sunrise both land somewhere a display can show.
vec3 present(vec3 color) {
    return pow(aces(color * u_exposure), vec3(1.0 / 2.2));
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

pub fn mesh_vs() -> String {
    with_wind(MESH_VS_BODY)
}

const MESH_FS_BODY: &str = r#"in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_tangent;
in vec4 v_shadow;
in float v_weathering;
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
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
uniform sampler2D u_shadow_tex;
out vec4 out_color;

// Trowbridge-Reitz normal distribution. `a2` is the square of the perceptual-to-linear
// roughness, so rough^4.
float d_ggx(float ndh, float a2) {
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-7);
}

// Height-correlated Smith visibility, with the 1/(4 NdotL NdotV) folded in.
float v_smith(float ndv, float ndl, float a2) {
    float lambda_v = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
    float lambda_l = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
    return 0.5 / max(lambda_v + lambda_l, 1e-5);
}

vec3 f_schlick(float u, vec3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - u, 5.0);
}

// Analytic fit of the split-sum environment BRDF, so the sky reflection costs no
// precomputed lookup. A scales Fresnel and B is the grazing lobe.
vec2 env_brdf(float ndv, float rough) {
    vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Sky ambient for a surface: split-sum diffuse and specular, with the multiple
// scattering the single-scatter term loses at high roughness added back, which is
// what keeps a rough metal from going dull.
vec3 sky_ibl(vec3 albedo, vec3 f0, float metallic, vec3 n, vec3 v, float rough) {
    float ndv = max(dot(n, v), 0.0) + 1e-4;
    vec2 ab = env_brdf(ndv, rough);
    vec3 f_ss = f0 * ab.x + ab.y;
    float e_ss = ab.x + ab.y;
    float e_ms = 1.0 - e_ss;
    vec3 f_avg = f0 + (vec3(1.0) - f0) / 21.0;
    vec3 f_ms = f_ss * f_avg / max(1.0 - e_ms * f_avg, vec3(1e-4));
    vec3 irradiance = sky_ambient(n);
    vec3 radiance = mix(sky_color(reflect(-v, n)), irradiance, rough);
    return albedo * irradiance * (1.0 - f_ss) * (1.0 - metallic)
        + radiance * (f_ss + f_ms * e_ms);
}

float sample_shadow(vec4 sc, float ndl) {
    vec3 proj = sc.xyz / sc.w * 0.5 + 0.5;
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z > 1.0) {
        return 1.0;
    }
    float bias = max(0.0015 * (1.0 - ndl), 0.0005);
    vec2 texel = 1.0 / vec2(textureSize(u_shadow_tex, 0));
    float sum = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float d = texture(u_shadow_tex, proj.xy + vec2(float(x), float(y)) * texel).r;
            sum += (proj.z - bias > d) ? 0.0 : 1.0;
        }
    }
    return sum / 9.0;
}

void main() {
    if (u_mode == 1) {
        float c = mod(floor(v_uv.x * 4.0) + floor(v_uv.y * 4.0), 2.0);
        out_color = vec4(mix(vec3(0.85, 0.85, 0.9), vec3(0.3, 0.55, 0.85), c), 1.0);
        return;
    }
    vec3 N = normalize(v_normal);
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
    float metallic = clamp(u_metallic, 0.0, 1.0);

    // Direct sun, one Cook-Torrance lobe. The diffuse is scaled by whatever the
    // interface did not reflect, so a glossy or metallic surface does not also glow
    // as a diffuse one.
    float ndv = max(dot(N, V), 0.0) + 1e-4;
    float ndl_raw = dot(N, u_sun_dir);
    float ndl = max(ndl_raw, 0.0);
    float shadow = ndl_raw > -0.05 ? sample_shadow(v_shadow, ndl) : 1.0;
    float a2 = rough * rough * rough * rough;
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 H = normalize(V + u_sun_dir);
    vec3 F = f_schlick(max(dot(V, H), 0.0), f0);
    vec3 spec = d_ggx(max(dot(N, H), 0.0), a2) * v_smith(ndv, ndl, a2) * F;
    vec3 direct = (albedo * (1.0 - F) * (1.0 - metallic) / PI + spec)
        * u_sun_color * ndl * shadow;

    out_color = vec4(present(direct + sky_ibl(albedo, f0, metallic, N, V, rough)), 1.0);
}"#;

pub fn mesh_fs() -> String {
    format!("#version 150
{SKY_GLSL}{MESH_FS_BODY}")
}

/// Bark into the shadow map, swayed exactly as the colour pass sways it, or the tree
/// would move through a shadow that stands still.
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
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
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
uniform sampler2D u_shadow_tex;
out vec4 out_color;

// Trowbridge-Reitz normal distribution. `a2` is the square of the perceptual-to-linear
// roughness, so rough^4.
float d_ggx(float ndh, float a2) {
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-7);
}

// Height-correlated Smith visibility, with the 1/(4 NdotL NdotV) folded in.
float v_smith(float ndv, float ndl, float a2) {
    float lambda_v = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
    float lambda_l = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
    return 0.5 / max(lambda_v + lambda_l, 1e-5);
}

vec3 f_schlick(float u, vec3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - u, 5.0);
}

// Analytic fit of the split-sum environment BRDF, so the sky reflection costs no
// precomputed lookup. A scales Fresnel and B is the grazing lobe.
vec2 env_brdf(float ndv, float rough) {
    vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Sky ambient for a surface: split-sum diffuse and specular, with the multiple
// scattering the single-scatter term loses at high roughness added back, which is
// what keeps a rough metal from going dull.
vec3 sky_ibl(vec3 albedo, vec3 f0, float metallic, vec3 n, vec3 v, float rough) {
    float ndv = max(dot(n, v), 0.0) + 1e-4;
    vec2 ab = env_brdf(ndv, rough);
    vec3 f_ss = f0 * ab.x + ab.y;
    float e_ss = ab.x + ab.y;
    float e_ms = 1.0 - e_ss;
    vec3 f_avg = f0 + (vec3(1.0) - f0) / 21.0;
    vec3 f_ms = f_ss * f_avg / max(1.0 - e_ms * f_avg, vec3(1e-4));
    vec3 irradiance = sky_ambient(n);
    vec3 radiance = mix(sky_color(reflect(-v, n)), irradiance, rough);
    return albedo * irradiance * (1.0 - f_ss) * (1.0 - metallic)
        + radiance * (f_ss + f_ms * e_ms);
}

float sample_shadow(vec4 sc, float ndl) {
    vec3 proj = sc.xyz / sc.w * 0.5 + 0.5;
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z > 1.0) {
        return 1.0;
    }
    float bias = max(0.004 * (1.0 - ndl), 0.0015);
    vec2 texel = 1.0 / vec2(textureSize(u_shadow_tex, 0));
    float sum = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float d = texture(u_shadow_tex, proj.xy + vec2(float(x), float(y)) * texel).r;
            sum += (proj.z - bias > d) ? 0.0 : 1.0;
        }
    }
    return sum / 9.0;
}

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
    // Foliage picks up the dome from both faces, so the ambient keeps the geometric
    // side while the direct terms use the flipped one and never go flat black.
    vec3 geometric = N;
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

    vec3 albedo = tex.rgb * v_tint.rgb;
    vec3 V = normalize(u_cam_pos - v_world);
    // The source map calls a leaf glossy, around 0.29, and at that roughness the
    // environment lobe hands each blade a mirror of the sky dome. Nothing here
    // occludes that dome, so a needle buried in the canopy reflects as much sky as
    // one on the outside and the whole crown washes out to the colour behind it.
    // Until the ambient is occluded, foliage is held to a matte floor.
    float rough = clamp(texture(u_rough_tex, uv).r, 0.55, 1.0);
    float ndl = max(dot(N, u_sun_dir), 0.0);
    // A needle crown lets light through in a thousand gaps, so shadow on foliage is
    // not the yes-or-no the shadow map says.
    float shadow = mix(1.0, sample_shadow(v_shadow, ndl), u_self_shadow);

    // Wrapped diffuse: a thin blade scatters enough that it never goes fully black
    // at grazing angles, and hard terminators across a canopy read as faceted. The
    // reflection term still takes its cut so the blade does not also glow.
    float wrapped = max((dot(N, u_sun_dir) + 0.5) / 1.5, 0.0);
    vec3 f0 = vec3(0.04);
    vec3 H = normalize(V + u_sun_dir);
    vec3 F = f_schlick(max(dot(V, H), 0.0), f0);
    vec3 diffuse = albedo * (1.0 - F) / PI * wrapped * shadow;

    // Light coming through the blade from behind. The view lobe peaks when the
    // camera looks into the sun, but it keeps a floor so a leaf turned away from
    // the sun is lit from behind instead of going black.
    float through = max(dot(-N, u_sun_dir), 0.0);
    float lobe = 0.35 + 0.65 * pow(max(dot(V, -u_sun_dir), 0.0), 3.0);
    vec3 transmitted = albedo * u_translucency * through * lobe * shadow;

    // The same Cook-Torrance lobe the bark uses, so a waxy blade catches a highlight
    // that tracks the roughness map instead of a fixed Blinn-Phong exponent.
    float a2 = rough * rough * rough * rough;
    vec3 spec = d_ggx(max(dot(N, H), 0.0), a2)
        * v_smith(max(dot(N, V), 0.0) + 1e-4, ndl, a2) * F * ndl * shadow;

    // Foliage picks up the dome from both faces, so the ambient uses the geometric
    // side rather than the flipped one and never goes flat black underneath.
    vec3 color = (diffuse + transmitted + spec) * u_sun_color
        + sky_ibl(albedo, f0, 0.0, geometric, V, rough);
    // Leaves buried in the crown get less sky than the ones on the outside.
    color *= v_tint.a;
    out_color = vec4(present(color), coverage);
}"#;

pub fn leaf_fs() -> String {
    format!("#version 150
{SKY_GLSL}{LEAF_FS_BODY}")
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
uniform float u_alpha_cutoff;
uniform sampler2D u_albedo_tex;
out vec4 out_color;
void main() {
    // Without the same alpha test the depth pass uses, every leaf would cast the
    // shadow of its bounding quad. It has to read the card's own arrangement too, or
    // a card casts the shadow of a cluster it is not drawing.
    vec2 uv = u_atlas_front + vec2(0.0, v_atlas_v) + v_card_uv * u_atlas_scale;
    if (texture(u_albedo_tex, uv).a < u_alpha_cutoff) {
        discard;
    }
    out_color = vec4(1.0);
}"#;

/// Full-screen dawn sky. The view ray is rebuilt from the inverse view-projection so
/// the gradient sits in the world rather than on the screen.
pub const SKY_VS: &str = r#"#version 150
in vec3 a_pos;
out vec2 v_ndc;
void main() {
    v_ndc = a_pos.xy;
    gl_Position = vec4(a_pos.xy, 1.0, 1.0);
}"#;

const SKY_FS_BODY: &str = r#"in vec2 v_ndc;
uniform mat4 u_inv_view_proj;
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
out vec4 out_color;

void main() {
    vec4 far = u_inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(far.xyz / far.w - u_cam_pos);
    vec3 col = sky_color(dir);
    // The sun itself, and the broad wash it throws across the sky beside it.
    float d = max(dot(dir, u_sun_dir), 0.0);
    col += u_sun_color * 0.22 * pow(d, 900.0);
    col += u_sun_color * 0.09 * pow(d, 18.0);
    col += u_sky_horizon * 0.35 * pow(d, 3.0);
    out_color = vec4(present(col), 1.0);
}"#;

pub fn sky_fs() -> String {
    format!("#version 150
{SKY_GLSL}{SKY_FS_BODY}")
}

/// A ground plane, so the tree casts onto something. It fades into the sky at range
/// rather than ending at a visible edge.
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
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_albedo_color;
uniform float u_fade_start;
uniform float u_fade_end;
uniform sampler2D u_shadow_tex;
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

float sample_shadow(vec4 sc, float ndl) {
    vec3 proj = sc.xyz / sc.w * 0.5 + 0.5;
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z > 1.0) {
        return 1.0;
    }
    float bias = max(0.0025 * (1.0 - ndl), 0.0008);
    vec2 texel = 1.0 / vec2(textureSize(u_shadow_tex, 0));
    float sum = 0.0;
    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            float d = texture(u_shadow_tex, proj.xy + vec2(float(x), float(y)) * texel).r;
            sum += (proj.z - bias > d) ? 0.0 : 1.0;
        }
    }
    return sum / 25.0;
}

void main() {
    vec3 N = vec3(0.0, 1.0, 0.0);
    float grain = vnoise(v_world.xz * 0.7) * 0.35 + vnoise(v_world.xz * 0.11) * 0.65;
    vec3 albedo = u_albedo_color * (0.84 + 0.32 * grain);
    float ndl = max(dot(N, u_sun_dir), 0.0);
    float shadow = sample_shadow(v_shadow, ndl);
    // Same convention as the bark and the leaves: u_sun_color is irradiance, so a
    // Lambert surface returns albedo / PI of it. Without that the ground came back
    // PI times brighter than everything standing on it.
    const float PI = 3.14159265359;
    vec3 color = albedo * (ndl * shadow * u_sun_color / PI + sky_ambient(N));

    // Dissolve into the sky at range so the plane has no visible rim.
    vec3 view = normalize(v_world - u_cam_pos);
    float dist = length(v_world.xz - u_cam_pos.xz);
    float fade = smoothstep(u_fade_start, u_fade_end, dist);
    color = mix(color, sky_color(view), fade);

    out_color = vec4(present(color), 1.0);
}"#;

pub fn ground_fs() -> String {
    format!("#version 150
{SKY_GLSL}{GROUND_FS_BODY}")
}
