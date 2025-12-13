#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec4 f_light_space_pos;

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    mat4 light_view_projection; // Block 1: Light space matrix
    float ambient_intensity;
    uint point_light_count;
    uint spotlight_count;
    float _pad1;

    vec3 light_direction;
    float _pad3;
    vec3 light_color;
    float light_intensity;

    vec3 camera_position;
    float _pad4;
} scene;

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad0;
    vec3 specular_color;
    float _pad1;
    float shininess;
} object;

void main() {
    vec4 position = object.model * vec4(v_position, 1.0f);

    // NOTE: Use normal matrix for correct normal transformation with non-uniform scaling
    mat3 normal_matrix = transpose(inverse(mat3(object.model)));
    vec3 normal = normal_matrix * v_normal;

    gl_Position = scene.view_projection * position;

    f_position = position.xyz;
    f_normal = normal;
    f_uv = v_uv;
    f_light_space_pos = scene.light_view_projection * position;
}
