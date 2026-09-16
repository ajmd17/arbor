pub const MESH_VS: &str = r#"#version 150
in vec3 a_pos;
in vec3 a_normal;
in vec2 a_uv;
in vec4 a_tangent;
uniform mat4 u_view_proj;
uniform mat4 u_light_view_proj;
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
    v_shadow = u_light_view_proj * vec4(a_pos, 1.0);
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
    if (dot(N, V) < 0.0) {
        N = -N;
    }
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
    vec3 hemi = mix(vec3(0.14, 0.13, 0.11), vec3(0.34, 0.42, 0.56), N.y * 0.5 + 0.5);
    vec3 color = (diff + spec) * u_sun_color * shadow + albedo * hemi;
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
