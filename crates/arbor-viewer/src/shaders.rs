pub const MESH_VS: &str = r#"#version 150
in vec3 a_pos;
in vec3 a_normal;
in vec2 a_uv;
in vec4 a_tangent;
uniform mat4 u_view_proj;
uniform mat4 u_light_view_proj;
uniform float u_normal_bias;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
out vec4 v_tangent;
out vec4 v_shadow;
void main() {
    v_world = a_pos;
    v_normal = a_normal;
    v_uv = a_uv;
    v_tangent = a_tangent;
    // Looked up a little along the normal rather than at the surface itself. With a
    // low sun the light grazes everything, and that is where plain depth bias either
    // stripes the bark with acne or lifts the shadow off its caster.
    v_shadow = u_light_view_proj * vec4(a_pos + a_normal * u_normal_bias, 1.0);
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
}"#;

pub const MESH_FS: &str = r#"#version 150
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_tangent;
in vec4 v_shadow;
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_albedo_color;
uniform float u_roughness;
uniform float u_metallic;
uniform int u_mode;
uniform int u_use_normal_map;
uniform sampler2D u_albedo_tex;
uniform sampler2D u_normal_tex;
uniform sampler2D u_rough_tex;
uniform sampler2D u_shadow_tex;
out vec4 out_color;

const float PI = 3.14159265359;

// One sky model shared by everything that shades: a warm band at the horizon under a
// cold zenith, which is what a low sun does to the dome, plus the bounce coming back
// up off the ground.
uniform vec3 u_sky_zenith;
uniform vec3 u_sky_horizon;
uniform vec3 u_ground_bounce;

vec3 sky_color(vec3 dir) {
    float h = clamp(dir.y, -1.0, 1.0);
    vec3 dome = mix(u_sky_horizon, u_sky_zenith, pow(max(h, 0.0), 0.42));
    return mix(u_ground_bounce, dome, smoothstep(-0.25, 0.03, h));
}

// Ambient arriving at a surface: the dome above it, fading into ground bounce as the
// surface turns to face down.
vec3 sky_ambient(vec3 n) {
    return sky_color(n) * 0.55 + u_sky_horizon * 0.12;
}

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

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
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

    vec3 color = aces(direct + sky_ibl(albedo, f0, metallic, N, V, rough));
    color = pow(color, vec3(1.0 / 2.2));
    out_color = vec4(color, 1.0);
}"#;

pub const DEPTH_VS: &str = r#"#version 150
in vec3 a_pos;
uniform mat4 u_light_view_proj;
void main() {
    gl_Position = u_light_view_proj * vec4(a_pos, 1.0);
}"#;

pub const DEPTH_FS: &str = r#"#version 150
out vec4 out_color;
void main() {
    out_color = vec4(1.0);
}"#;

pub const COLOR_VS: &str = r#"#version 150
in vec3 a_pos;
uniform mat4 u_view_proj;
void main() {
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
}"#;

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

pub const LEAF_VS: &str = r#"#version 150
in vec3 a_pos;
in vec3 a_normal;
in vec2 a_uv;
in vec4 a_tint;
uniform mat4 u_view_proj;
uniform mat4 u_light_view_proj;
uniform float u_normal_bias;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_card_uv;
out vec4 v_tint;
out vec4 v_shadow;
void main() {
    v_world = a_pos;
    v_normal = a_normal;
    v_card_uv = a_uv;
    v_tint = a_tint;
    v_shadow = u_light_view_proj * vec4(a_pos + a_normal * u_normal_bias, 1.0);
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
}"#;

pub const LEAF_FS: &str = r#"#version 150
in vec3 v_world;
in vec3 v_normal;
in vec2 v_card_uv;
in vec4 v_tint;
in vec4 v_shadow;
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec2 u_atlas_scale;
uniform vec2 u_atlas_front;
uniform vec2 u_atlas_back;
uniform float u_alpha_cutoff;
// Mip level at which coverage starts giving way to a hard cutoff.
uniform float u_coverage_lod;
uniform float u_translucency;
uniform int u_mode;
uniform sampler2D u_albedo_tex;
uniform sampler2D u_rough_tex;
uniform sampler2D u_shadow_tex;
out vec4 out_color;

// One sky model shared by everything that shades: a warm band at the horizon under a
// cold zenith, which is what a low sun does to the dome, plus the bounce coming back
// up off the ground.
uniform vec3 u_sky_zenith;
uniform vec3 u_sky_horizon;
uniform vec3 u_ground_bounce;

vec3 sky_color(vec3 dir) {
    float h = clamp(dir.y, -1.0, 1.0);
    vec3 dome = mix(u_sky_horizon, u_sky_zenith, pow(max(h, 0.0), 0.42));
    return mix(u_ground_bounce, dome, smoothstep(-0.25, 0.03, h));
}

// Ambient arriving at a surface: the dome above it, fading into ground bounce as the
// surface turns to face down.
vec3 sky_ambient(vec3 n) {
    return sky_color(n) * 0.55 + u_sky_horizon * 0.12;
}

const float PI = 3.14159265359;

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

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    // A leaf is one quad seen from both sides: the lit face and the underside are
    // different cells of the same atlas, picked per fragment.
    vec2 cell = gl_FrontFacing ? u_atlas_front : u_atlas_back;
    vec2 uv = cell + v_card_uv * u_atlas_scale;
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
        N = -N;
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
    float shadow = sample_shadow(v_shadow, ndl);

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
    color = aces(color);
    color = pow(color, vec3(1.0 / 2.2));
    out_color = vec4(color, coverage);
}"#;

pub const LEAF_DEPTH_VS: &str = r#"#version 150
in vec3 a_pos;
in vec2 a_uv;
uniform mat4 u_light_view_proj;
out vec2 v_card_uv;
void main() {
    v_card_uv = a_uv;
    gl_Position = u_light_view_proj * vec4(a_pos, 1.0);
}"#;

pub const LEAF_DEPTH_FS: &str = r#"#version 150
in vec2 v_card_uv;
uniform vec2 u_atlas_scale;
uniform vec2 u_atlas_front;
uniform float u_alpha_cutoff;
uniform sampler2D u_albedo_tex;
out vec4 out_color;
void main() {
    // Without the same alpha test the depth pass uses, every leaf would cast the
    // shadow of its bounding quad.
    if (texture(u_albedo_tex, u_atlas_front + v_card_uv * u_atlas_scale).a < u_alpha_cutoff) {
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

pub const SKY_FS: &str = r#"#version 150
in vec2 v_ndc;
uniform mat4 u_inv_view_proj;
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
out vec4 out_color;

uniform vec3 u_sky_zenith;
uniform vec3 u_sky_horizon;
uniform vec3 u_ground_bounce;

vec3 sky_color(vec3 dir) {
    float h = clamp(dir.y, -1.0, 1.0);
    vec3 dome = mix(u_sky_horizon, u_sky_zenith, pow(max(h, 0.0), 0.42));
    return mix(u_ground_bounce, dome, smoothstep(-0.25, 0.03, h));
}

vec3 sky_ambient(vec3 n) {
    return sky_color(n) * 0.55 + u_sky_horizon * 0.12;
}

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec4 far = u_inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(far.xyz / far.w - u_cam_pos);
    vec3 col = sky_color(dir);
    // The sun itself, and the broad wash it throws across the sky beside it.
    float d = max(dot(dir, u_sun_dir), 0.0);
    col += u_sun_color * 0.22 * pow(d, 900.0);
    col += u_sun_color * 0.09 * pow(d, 18.0);
    col += u_sky_horizon * 0.35 * pow(d, 3.0);
    col = aces(col);
    out_color = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}"#;

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

pub const GROUND_FS: &str = r#"#version 150
in vec3 v_world;
in vec4 v_shadow;
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_albedo_color;
uniform float u_fade_start;
uniform float u_fade_end;
uniform sampler2D u_shadow_tex;
out vec4 out_color;

uniform vec3 u_sky_zenith;
uniform vec3 u_sky_horizon;
uniform vec3 u_ground_bounce;

vec3 sky_color(vec3 dir) {
    float h = clamp(dir.y, -1.0, 1.0);
    vec3 dome = mix(u_sky_horizon, u_sky_zenith, pow(max(h, 0.0), 0.42));
    return mix(u_ground_bounce, dome, smoothstep(-0.25, 0.03, h));
}

vec3 sky_ambient(vec3 n) {
    return sky_color(n) * 0.55 + u_sky_horizon * 0.12;
}

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

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

    color = aces(color);
    out_color = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}"#;
