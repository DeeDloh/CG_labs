#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
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

struct PointLight {
    vec3 position;
    float _pad0;
    vec3 color;
    float intensity;
};

struct Spotlight {
    vec3 position;
    float _pad0;
    vec3 direction;
    float _pad1;
    vec3 color;
    float intensity;
    float inner_cutoff;  // cos(inner_angle)
    float outer_cutoff;  // cos(outer_angle)
    float _pad2, _pad3;
};

layout (binding = 2, std430) readonly buffer PointLights {
    PointLight lights[];
} point_lights;

layout (binding = 3, std430) readonly buffer SpotLights {
    Spotlight lights[];
} spot_lights;

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(scene.camera_position - f_position);

    // NOTE: Ambient component
    vec3 ambient = scene.ambient_intensity * object.albedo_color;

    // NOTE: Directional light - Diffuse component
    float diff = max(dot(normal, scene.light_direction), 0.0);
    vec3 diffuse = diff * scene.light_intensity * scene.light_color * object.albedo_color;
    
    // NOTE: Directional light - Specular component (Blinn-Phong)
    vec3 halfway_dir = normalize(scene.light_direction + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), object.shininess);
    vec3 specular = spec * scene.light_intensity * scene.light_color * object.specular_color;
    
    // NOTE: Point lights contribution
    vec3 point_lighting = vec3(0.0);
    for (uint i = 0; i < scene.point_light_count; ++i) {
        PointLight light = point_lights.lights[i];
        
        vec3 light_dir = light.position - f_position;
        float distance = length(light_dir);
        light_dir = normalize(light_dir);
        
        // NOTE: Attenuation (inverse square law with epsilon to avoid division by zero)
        float attenuation = light.intensity / (distance * distance + 0.01);
        
        // NOTE: Diffuse
        float point_diff = max(dot(normal, light_dir), 0.0);
        vec3 point_diffuse = point_diff * attenuation * light.color * object.albedo_color;
        
        // NOTE: Specular
        vec3 point_halfway = normalize(light_dir + view_dir);
        float point_spec = pow(max(dot(normal, point_halfway), 0.0), object.shininess);
        vec3 point_specular = point_spec * attenuation * light.color * object.specular_color;
        
        point_lighting += point_diffuse + point_specular;
    }

    // NOTE: Spotlights contribution
    vec3 spotlight_lighting = vec3(0.0);
    for (uint i = 0; i < scene.spotlight_count; ++i) {
        Spotlight light = spot_lights.lights[i];
        
        vec3 light_dir = light.position - f_position;
        float distance = length(light_dir);
        light_dir = normalize(light_dir);
        
        // NOTE: Check if fragment is within spotlight cone
        float theta = dot(light_dir, normalize(-light.direction));
        float epsilon = light.inner_cutoff - light.outer_cutoff;
        float spot_intensity = clamp((theta - light.outer_cutoff) / epsilon, 0.0, 1.0);
        
        // NOTE: Distance attenuation
        float attenuation = light.intensity / (distance * distance + 0.01);
        
        // NOTE: Combine spot and distance attenuation
        float total_attenuation = attenuation * spot_intensity;
        
        // NOTE: Diffuse
        float spot_diff = max(dot(normal, light_dir), 0.0);
        vec3 spot_diffuse = spot_diff * total_attenuation * light.color * object.albedo_color;
        
        // NOTE: Specular
        vec3 spot_halfway = normalize(light_dir + view_dir);
        float spot_spec = pow(max(dot(normal, spot_halfway), 0.0), object.shininess);
        vec3 spot_specular = spot_spec * total_attenuation * light.color * object.specular_color;
        
        spotlight_lighting += spot_diffuse + spot_specular;
    }

    // NOTE: Combine all lighting
    vec3 result = ambient + diffuse + specular + point_lighting + spotlight_lighting;
    
    final_color = vec4(result, 1.0);
}
