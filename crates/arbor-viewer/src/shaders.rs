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

const float PI = 3.14159265;

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
    float rough = clamp(texture(u_rough_tex, v_uv).r * u_roughness, 0.04, 1.0);
    float ndl = max(dot(N, u_sun_dir), 0.0);
    float shadow = 1.0;
    if (ndl > -0.05) {
        shadow = sample_shadow(v_shadow, ndl);
    }
    vec3 H = normalize(V + u_sun_dir);
    float ndv = max(dot(N, V), 0.0001);
    float ndh = max(dot(N, H), 0.0);
    float vdh = max(dot(V, H), 0.0);
    float a = rough * rough;
    float a2 = a * a;
    float den = ndh * ndh * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * den * den);
    float k = rough + 1.0;
    k = k * k / 8.0;
    float G = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k));
    vec3 F0 = mix(vec3(0.04), albedo, u_metallic);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - vdh, 5.0);
    vec3 spec = (D * G * F) / (4.0 * ndv * max(ndl, 0.001) + 0.001) * ndl;
    vec3 diff = albedo * (1.0 - u_metallic) / PI * ndl;
    vec3 color = (diff + spec) * u_sun_color * shadow + albedo * sky_ambient(N);
    color = aces(color);
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
    // Resolve the cutout edge over roughly one pixel instead of snapping to it. With
    // alpha-to-coverage on, this hands the hardware a real coverage fraction, so a
    // leaf thins out smoothly at distance rather than flickering in and out.
    float edge = max(fwidth(tex.a), 1e-4);
    float mask = clamp((tex.a - u_alpha_cutoff) / edge + 0.5, 0.0, 1.0);
    if (mask <= 0.0) {
        discard;
    }
    if (u_mode == 1) {
        float c = mod(floor(v_card_uv.x * 4.0) + floor(v_card_uv.y * 4.0), 2.0);
        out_color = vec4(mix(vec3(0.2, 0.5, 0.2), vec3(0.85, 0.9, 0.5), c), 1.0);
        return;
    }
    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing) {
        N = -N;
    }
    if (u_mode == 2) {
        out_color = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }

    vec3 albedo = tex.rgb * v_tint.rgb;
    vec3 V = normalize(u_cam_pos - v_world);
    float rough = clamp(texture(u_rough_tex, uv).r, 0.15, 1.0);
    float ndl = max(dot(N, u_sun_dir), 0.0);
    float shadow = sample_shadow(v_shadow, ndl);

    // Wrapped diffuse: a thin blade scatters enough that it never goes fully black
    // at grazing angles, and hard terminators across a canopy read as faceted.
    float wrapped = max((dot(N, u_sun_dir) + 0.5) / 1.5, 0.0);
    vec3 diffuse = albedo * wrapped * shadow;

    // Light coming through the blade from behind. The view lobe peaks when the
    // camera looks into the sun, but it keeps a floor so a leaf turned away from
    // the sun is lit from behind instead of going black.
    float through = max(dot(-N, u_sun_dir), 0.0);
    float lobe = 0.35 + 0.65 * pow(max(dot(V, -u_sun_dir), 0.0), 3.0);
    vec3 transmitted = albedo * u_translucency * through * lobe * shadow;

    vec3 H = normalize(V + u_sun_dir);
    float spec = pow(max(dot(N, H), 0.0), mix(60.0, 6.0, rough)) * (1.0 - rough) * 0.25 * shadow;

    // Foliage picks up the dome from both faces, so the ambient uses the geometric
    // side rather than the flipped one and never goes flat black underneath.
    vec3 color = (diffuse + transmitted + vec3(spec)) * u_sun_color + albedo * sky_ambient(N);
    // Leaves buried in the crown get less sky than the ones on the outside.
    color *= v_tint.a;
    color = aces(color);
    color = pow(color, vec3(1.0 / 2.2));
    out_color = vec4(color, mask);
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
    vec3 color = albedo * (ndl * shadow * u_sun_color + sky_ambient(N));

    // Dissolve into the sky at range so the plane has no visible rim.
    vec3 view = normalize(v_world - u_cam_pos);
    float dist = length(v_world.xz - u_cam_pos.xz);
    float fade = smoothstep(u_fade_start, u_fade_end, dist);
    color = mix(color, sky_color(view), fade);

    color = aces(color);
    out_color = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}"#;
