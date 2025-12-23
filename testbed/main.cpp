#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

#define MINIAUDIO_IMPLEMENTATION
#include "../miniaudio.h"

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t SHADOW_MAP_SIZE = 4096;

enum class CameraMode {
    LookAt,
    Transform
};

struct CameraState {
    veekay::vec3 position;
    veekay::vec3 rotation;
};

struct Vertex {
    veekay::vec3 position;
    veekay::vec3 normal;
    veekay::vec2 uv;
    // NOTE: You can add more attributes
};

struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::mat4 shadow_projection;

	veekay::vec3 view_position; float _pad0;

	veekay::vec3 ambient_light_intensity; float _pad1;

	veekay::vec3 sun_light_direction; float _pad2;
	veekay::vec3 sun_light_color; float _pad3;
	uint32_t point_lights_count;
    uint32_t spot_lights_count;
    uint32_t _pad4[2];
};

struct PointLight {
    veekay::vec3 position;
    float radius;
    veekay::vec3 color;
    float _pad0;
};

struct SpotLight {
    veekay::vec3 position;
    float radius;

    veekay::vec3 direction;
    float angle;

    veekay::vec3 color;
    float _pad0;
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color; float _pad0;
	veekay::vec3 specular_color; float _pad1;
    float shininess; float _pad2, _pad3, _pad4;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color;
	uint32_t material_id = 0;
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};
    veekay::vec3 target = {0.0f, -0.5f, 0.0f};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	veekay::mat4 view() const;
	veekay::mat4 view_projection(float aspect_ratio) const;
};

// NOTE: Scene objects
inline namespace {
	Camera camera{
		.position = {0.0f, -0.5f, 10.0f},
	};

	std::vector<Model> models;

	// NOTE: Camera mode management
    CameraMode current_camera_mode = CameraMode::Transform;
    CameraState saved_lookat_state{
        .position = {0.0f, -0.5f, 10.0f},
        .rotation = {0.0f, 0.0f, 0.0f}
    };
    CameraState saved_transform_state{
        .position = {0.0f, -0.5f, 10.0f},
        .rotation = {0.0f, 0.0f, 0.0f}
    };
}

// NOTE: Vulkan objects
inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	std::vector<VkDescriptorSet> material_descriptor_sets;
	std::vector<veekay::graphics::Texture*> textures;
	std::vector<VkSampler> samplers;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;
	veekay::graphics::Buffer* point_lights_buffer;


	Mesh plane_mesh;
	Mesh cube_mesh;

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;

	std::vector<PointLight> point_lights;
	std::vector<SpotLight> spot_lights;
	veekay::graphics::Buffer* spot_lights_buffer;

	struct ShadowPass {
		VkFormat depth_image_format;
		VkImage depth_image;
		VkDeviceMemory depth_image_memory;
		VkImageView depth_image_view;

		VkShaderModule vertex_shader;

		VkDescriptorSetLayout descriptor_set_layout;
		VkDescriptorSet descriptor_set;
		VkPipelineLayout pipeline_layout;
		VkPipeline pipeline;

		veekay::graphics::Buffer* uniform_buffer;
		VkSampler sampler;

		veekay::mat4 matrix;
	};

	ShadowPass shadow;

	PFN_vkCmdBeginRenderingKHR vkCmdBeginRenderingKHR;
	PFN_vkCmdEndRenderingKHR   vkCmdEndRenderingKHR;

	constexpr uint32_t shadow_map_size = 4096;

	ma_engine audio_engine;
	bool audio_initialized = false;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

// NOTE: Helper function to clamp values (C++11 compatible)
template<typename T>
T clamp(T value, T min_val, T max_val) {
    return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
}

veekay::graphics::Texture* loadTexture(VkCommandBuffer cmd, const char* path) {
    std::vector<unsigned char> image;
    unsigned width, height;

    unsigned error = lodepng::decode(image, width, height, path);
    if (error) {
        std::cerr << "Failed to load texture: " << path << " (error " << error << "): " << lodepng_error_text(error) << "\n";
        return nullptr;
    }

    if (width == 0 || height == 0 || width > 8192 || height > 8192) {
        std::cerr << "Invalid texture dimensions: " << width << "x" << height << " for " << path << "\n";
        return nullptr;
    }

    std::cout << "Loaded texture: " << path << " (" << width << "x" << height << ")\n";

    return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, image.data());
}

static constexpr veekay::vec3 WORLD_UP{0.0f, -1.0f, 0.0f};

veekay::mat4 lookat(veekay::vec3 forward, veekay::vec3 position) {
	using veekay::vec3;
    using veekay::mat4;

	vec3 target = forward;
	vec3 eye    = position;
    vec3 center = { eye.x + target.x,
                    eye.y + target.y,
                    eye.z + target.z };

    vec3 f = forward;
    vec3 r = vec3::normalized(vec3::cross(f, WORLD_UP));
    vec3 u = vec3::cross(r, f);

    mat4 m{};
    m[0][0] = -r.x; m[0][1] = -u.x; m[0][2] = -f.x; m[0][3] = 0.0f;
    m[1][0] = -r.y; m[1][1] = -u.y; m[1][2] = -f.y; m[1][3] = 0.0f;
    m[2][0] = -r.z; m[2][1] = -u.z; m[2][2] = -f.z; m[2][3] = 0.0f;
    m[3][0] =  vec3::dot(r, eye);
    m[3][1] =  vec3::dot(u, eye);
    m[3][2] =  vec3::dot(f, eye);
    m[3][3] =  1.0f;

    return m;
}

veekay::mat4 Transform::matrix() const {
    veekay::mat4 trans = veekay::mat4::translation(position);
    veekay::mat4 rot_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
    veekay::mat4 rot_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
    veekay::mat4 rot_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));
    veekay::mat4 scale_mat = veekay::mat4::scaling(scale);

    return trans * rot_z * rot_y * rot_x * scale_mat;
}

veekay::mat4 Camera::view() const {
	if (current_camera_mode == CameraMode::LookAt) {
		// NOTE: Calculate forward direction from pitch and yaw
		float pitch = toRadians(rotation.x);
		float yaw = toRadians(rotation.y);

		veekay::vec3 forward{
			std::sin(yaw),
			std::sin(pitch),
			std::cos(yaw)
		};
		forward = veekay::vec3::normalized(forward);

		// NOTE: World up vector (camera's local up is -Y in world space)
		veekay::vec3 world_up{0.0f, -1.0f, 0.0f};

		// NOTE: Calculate right vector (perpendicular to forward and up)
		veekay::vec3 right = veekay::vec3::cross(forward, world_up);
		right = veekay::vec3::normalized(right);
		right = right * -1.0f;

		// NOTE: Recalculate up vector (perpendicular to forward and right)
		veekay::vec3 up = veekay::vec3::cross(right, forward);

		// NOTE: Build look-at view matrix manually
		veekay::mat4 result = veekay::mat4::identity();

		// NOTE: Set rotation part (basis vectors as rows)
		result.elements[0][0] = right.x;
		result.elements[1][0] = right.y;
		result.elements[2][0] = right.z;

		result.elements[0][1] = up.x;
		result.elements[1][1] = up.y;
		result.elements[2][1] = up.z;

		result.elements[0][2] = -forward.x;
		result.elements[1][2] = -forward.y;
		result.elements[2][2] = -forward.z;

		// NOTE: Set translation part (dot products with position)
		result.elements[3][0] = -veekay::vec3::dot(right, position);
		result.elements[3][1] = -veekay::vec3::dot(up, position);
		result.elements[3][2] = veekay::vec3::dot(forward, position);

		return result;
	}
	auto t = veekay::mat4::translation(-position);
    auto rx = veekay::mat4::rotation({1, 0, 0}, toRadians(-rotation.x));
    auto ry = veekay::mat4::rotation({0, 1, 0}, toRadians(-rotation.y - 180.0f));
    auto rz = veekay::mat4::rotation({0, 0, 1}, toRadians(-rotation.z));
    return t * rz * ry * rx;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
	return view() * projection;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
		vkGetDeviceProcAddr(device, "vkCmdBeginRenderingKHR"));
	vkCmdEndRenderingKHR = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(
		vkGetDeviceProcAddr(device, "vkCmdEndRenderingKHR"));

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		shadow.vertex_shader = loadShaderModule("./shaders/shadow.vert.spv");
		if (!shadow.vertex_shader) {
			std::cerr << "Failed to load shadow vertex shader\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
		};

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 16,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 16,
				},
				{
                    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 16,
                },
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 32,
				}
			};

			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 10,  // Increased for shadow descriptor set
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Descriptor set layout specification
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
                    .binding = 2,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
				{
    				.binding = 3,
    				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    				.descriptorCount = 1,
    				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 4,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 5,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}

		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	{
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				},
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = uint32_t(std::size(bindings)),
				.pBindings = bindings,
			};

			vkCreateDescriptorSetLayout(device, &info, nullptr,
										&shadow.descriptor_set_layout);
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &shadow.descriptor_set_layout,
			};
			vkAllocateDescriptorSets(device, &info, &shadow.descriptor_set);
		}

		{
			VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 1,
				.pSetLayouts = &shadow.descriptor_set_layout,
			};
			vkCreatePipelineLayout(device, &layout_info, nullptr,
								&shadow.pipeline_layout);
		}

		{
			if (shadow.depth_image_format == VK_FORMAT_UNDEFINED) {
				VkFormat candidates[] = {
					VK_FORMAT_D32_SFLOAT,
					VK_FORMAT_D32_SFLOAT_S8_UINT,
					VK_FORMAT_D24_UNORM_S8_UINT,
				};

				for (auto f : candidates) {
					VkFormatProperties props;
					vkGetPhysicalDeviceFormatProperties(physical_device, f, &props);
					if (props.optimalTilingFeatures &
						VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
						shadow.depth_image_format = f;
						break;
					}
				}
			}

			VkVertexInputBindingDescription shadow_binding{
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			};

			VkVertexInputAttributeDescription shadow_attribute{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, position),
			};

			VkPipelineVertexInputStateCreateInfo shadow_input_state{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &shadow_binding,
				.vertexAttributeDescriptionCount = 1,
				.pVertexAttributeDescriptions = &shadow_attribute,
			};

			VkPipelineShaderStageCreateInfo stage{
				.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage  = VK_SHADER_STAGE_VERTEX_BIT,
				.module = shadow.vertex_shader,
				.pName  = "main",
			};

			VkPipelineRasterizationStateCreateInfo raster_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_FRONT_BIT,
				.frontFace = VK_FRONT_FACE_CLOCKWISE,
				.depthBiasEnable = VK_TRUE,
				.lineWidth = 1.0f,
			};

			VkPipelineColorBlendStateCreateInfo blend_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				.attachmentCount = 0,
			};

			VkDynamicState dyn_states[] = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR,
				VK_DYNAMIC_STATE_DEPTH_BIAS,
			};
			VkPipelineDynamicStateCreateInfo dyn_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
				.dynamicStateCount = uint32_t(std::size(dyn_states)),
				.pDynamicStates = dyn_states,
			};

			VkPipelineViewportStateCreateInfo viewport_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1,
				.scissorCount  = 1,
			};

			VkPipelineDepthStencilStateCreateInfo depth_stencil{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = VK_TRUE,
				.depthWriteEnable = VK_TRUE,
				.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			};

			VkPipelineInputAssemblyStateCreateInfo assembly{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			};

			VkPipelineMultisampleStateCreateInfo multisample{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			};

			VkPipelineRenderingCreateInfoKHR format_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
				.depthAttachmentFormat = shadow.depth_image_format,
			};

			VkGraphicsPipelineCreateInfo pipeline_info{
				.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.pNext = &format_info,
				.stageCount = 1,
				.pStages = &stage,
				.pVertexInputState   = &shadow_input_state,
				.pInputAssemblyState = &assembly,
				.pViewportState      = &viewport_info,
				.pRasterizationState = &raster_info,
				.pMultisampleState   = &multisample,
				.pDepthStencilState  = &depth_stencil,
				.pColorBlendState    = &blend_info,
				.pDynamicState       = &dyn_state_info,
				.layout = shadow.pipeline_layout,
			};

			vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_info, nullptr,
									&shadow.pipeline);
		}

		{
			VkImageCreateInfo image_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
				.imageType = VK_IMAGE_TYPE_2D,
				.format = shadow.depth_image_format,
				.extent = { shadow_map_size, shadow_map_size, 1 },
				.mipLevels = 1,
				.arrayLayers = 1,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.tiling = VK_IMAGE_TILING_OPTIMAL,
				.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
						VK_IMAGE_USAGE_SAMPLED_BIT,
			};

			vkCreateImage(device, &image_info, nullptr, &shadow.depth_image);

			VkMemoryRequirements req;
			vkGetImageMemoryRequirements(device, shadow.depth_image, &req);

			VkPhysicalDeviceMemoryProperties mem_props;
			vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

			uint32_t index = UINT32_MAX;
			for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
				if ((req.memoryTypeBits & (1 << i)) &&
					(mem_props.memoryTypes[i].propertyFlags &
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
					index = i;
					break;
				}
			}

			VkMemoryAllocateInfo alloc{
				.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
				.allocationSize = req.size,
				.memoryTypeIndex = index,
			};
			vkAllocateMemory(device, &alloc, nullptr, &shadow.depth_image_memory);
			vkBindImageMemory(device, shadow.depth_image, shadow.depth_image_memory, 0);

			VkImageViewCreateInfo view_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = shadow.depth_image,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = shadow.depth_image_format,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};
			vkCreateImageView(device, &view_info, nullptr, &shadow.depth_image_view);
		}
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	point_lights_buffer = new veekay::graphics::Buffer(
	16 * sizeof(PointLight),
	nullptr,
	VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);


	spot_lights_buffer = new veekay::graphics::Buffer(
    8 * sizeof(SpotLight),
    nullptr,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
	);

	shadow.uniform_buffer = new veekay::graphics::Buffer(
    sizeof(veekay::mat4), nullptr,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = shadow.uniform_buffer->buffer,
				.range  = sizeof(veekay::mat4),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.range  = sizeof(ModelUniforms),
			},
		};

		VkWriteDescriptorSet writes[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = shadow.descriptor_set,
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = shadow.descriptor_set,
				.dstBinding = 1,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
		};

		vkUpdateDescriptorSets(device,
							uint32_t(std::size(writes)), writes,
							0, nullptr);
	}

	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.compareEnable = VK_TRUE,
			.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			.minLod = 0.0f,
			.maxLod = VK_LOD_CLAMP_NONE,
			.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
		};

		vkCreateSampler(device, &info, nullptr, &shadow.sampler);
	}

	// NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	// NOTE: Load textures from assets folder
	{
		const char* texture_paths[] = {
			"assets/dexter.png",      // 0 - for floor
			"assets/dex-1.png",    // 1
			"assets/dex.png",   // 2
			"assets/dex-4.png"    // 3
		};

		for (const char* path : texture_paths) {
			veekay::graphics::Texture* tex = loadTexture(cmd, path);
			if (tex) {
				textures.push_back(tex);
			} else {
				std::cerr << "Failed to load texture, using missing_texture instead\n";
				textures.push_back(missing_texture);
			}
		}

		// NOTE: Create samplers for each texture
		VkSamplerCreateInfo sampler_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.anisotropyEnable = VK_TRUE,
			.maxAnisotropy = 16.0f,
		};

		for (size_t i = 0; i < textures.size(); i++) {
			VkSampler sampler;
			if (vkCreateSampler(device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
				std::cerr << "Failed to create texture sampler\n";
				veekay::app.running = false;
				return;
			}
			samplers.push_back(sampler);
		}
	}

	// NOTE: Create descriptor sets for each material
	{
		material_descriptor_sets.resize(textures.size());

		std::vector<VkDescriptorSetLayout> layouts(textures.size(), descriptor_set_layout);

		VkDescriptorSetAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = static_cast<uint32_t>(textures.size()),
			.pSetLayouts = layouts.data(),
		};

		if (vkAllocateDescriptorSets(device, &alloc_info, material_descriptor_sets.data()) != VK_SUCCESS) {
			std::cerr << "Failed to allocate material descriptor sets\n";
			veekay::app.running = false;
			return;
		}

		// Update descriptor sets for each material
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
			{
                .buffer = point_lights_buffer->buffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE,
            },
			{
				.buffer = spot_lights_buffer->buffer,
				.offset = 0,
				.range = VK_WHOLE_SIZE,
			},
		};

		for (size_t i = 0; i < textures.size(); i++) {
			VkDescriptorImageInfo image_info{
				.sampler = samplers[i],
				.imageView = textures[i]->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			VkDescriptorImageInfo shadow_image_info{
				.sampler = shadow.sampler,
				.imageView = shadow.depth_image_view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			VkWriteDescriptorSet writes[] = {
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = material_descriptor_sets[i],
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &buffer_infos[0],
				},
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = material_descriptor_sets[i],
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.pBufferInfo = &buffer_infos[1],
				},
				{
	                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
	                .dstSet = material_descriptor_sets[i],
	                .dstBinding = 2,
	                .dstArrayElement = 0,
	                .descriptorCount = 1,
	                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                .pBufferInfo = &buffer_infos[2],
	            },
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = material_descriptor_sets[i],
					.dstBinding = 3,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &buffer_infos[3],
				},
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = material_descriptor_sets[i],
					.dstBinding = 4,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &image_info,
				},
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = material_descriptor_sets[i],
					.dstBinding = 5,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &shadow_image_info,
				},
			};

			vkUpdateDescriptorSets(device, sizeof(writes) / sizeof(writes[0]), writes, 0, nullptr);
		}
	}

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0
		};

		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Cube mesh initialization
	{
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
			8, 9, 10, 10, 11, 8,
			12, 13, 14, 14, 15, 12,
			16, 17, 18, 18, 19, 16,
			20, 21, 22, 22, 23, 20,
		};

		cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		cube_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Add models to scene
	models.emplace_back(Model{
		.mesh = plane_mesh,
		.transform = Transform{},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.material_id = 0  // dex.jpg
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {-2.5f, -0.5f, 1.8f},
			.scale = {1.0f, 1.0f, 1.0f},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.material_id = 1  // dex-1.jpg
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {3.2f, -0.5f, -1.5f},
			.scale = {1.0f, 1.0f, 1.0f},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.material_id = 2  // dex-2.jpeg
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {0.8f, -0.5f, 2.7f},
			.scale = {1.0f, 1.0f, 1.0f},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.material_id = 3  // dex-3.jpeg
	});

    // NOTE: Add point lights to the scene
    // point_lights.push_back(PointLight{
    //     .position = {-5.0f, -5.0f, 0.0f},
    //     .color = {0.0f, 0.0f, 1.0f},
    //     .intensity = 20.0f,
    // });

    point_lights.push_back(PointLight{
        .position = {5.0f, -5.0f, 0.0f},
        .radius = 10.0f,
        .color = {1.0f, 1.0f, 1.0f},
    });

	spot_lights.push_back(SpotLight{
		.position = {0.0f, -3.0f, 0.0f},
		.radius = 10.0f,
		.direction = {0.0f, 1.0f, 0.0f},
		.angle = std::cos(toRadians(25.0f)),
		.color = {1.0f, 1.0f, 0.8f},
	});

	// Initialize audio engine
	ma_result audio_result = ma_engine_init(NULL, &audio_engine);
	if (audio_result == MA_SUCCESS) {
		audio_initialized = true;
		std::cout << "Audio system initialized successfully\n";
	} else {
		std::cerr << "Failed to initialize audio system: " << audio_result << "\n";
		audio_initialized = false;
	}
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	// Clean up textures and samplers
	for (veekay::graphics::Texture* tex : textures) {
		if (tex != missing_texture) {
			delete tex;
		}
	}
	for (VkSampler sampler : samplers) {
		vkDestroySampler(device, sampler, nullptr);
	}

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete plane_mesh.index_buffer;
	delete plane_mesh.vertex_buffer;

    delete point_lights_buffer;
	delete spot_lights_buffer;
	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	// Clean up shadow resources
	delete shadow.uniform_buffer;
	vkDestroySampler(device, shadow.sampler, nullptr);
	vkDestroyImageView(device, shadow.depth_image_view, nullptr);
	vkDestroyImage(device, shadow.depth_image, nullptr);
	vkFreeMemory(device, shadow.depth_image_memory, nullptr);
	vkDestroyPipeline(device, shadow.pipeline, nullptr);
	vkDestroyPipelineLayout(device, shadow.pipeline_layout, nullptr);
	vkDestroyShaderModule(device, shadow.vertex_shader, nullptr);
	vkDestroyDescriptorSetLayout(device, shadow.descriptor_set_layout, nullptr);

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);

	// Cleanup audio
	if (audio_initialized) {
		ma_engine_uninit(&audio_engine);
	}
}

void update(double time) {
    ImGui::Begin("Lighting Controls");

    const char* mode_names[] = { "Look-At", "Transform" };
    int current_mode = static_cast<int>(current_camera_mode);

    if (ImGui::Combo("Mode", &current_mode, mode_names, 2)) {
        // NOTE: Save current state before switching
        if (current_camera_mode == CameraMode::LookAt) {
            saved_lookat_state.position = camera.position;
            saved_lookat_state.rotation = camera.rotation;
        } else {
            saved_transform_state.position = camera.position;
            saved_transform_state.rotation = camera.rotation;
        }

        // NOTE: Switch mode
        current_camera_mode = static_cast<CameraMode>(current_mode);

        // NOTE: Restore saved state for new mode
        if (current_camera_mode == CameraMode::LookAt) {
            camera.position = saved_lookat_state.position;
            camera.rotation = saved_lookat_state.rotation;
        } else {
            camera.position = saved_transform_state.position;
            camera.rotation = saved_transform_state.rotation;
        }
    }

    // NOTE: Display current rotation for debugging
    ImGui::Text("Position: (%.2f, %.2f, %.2f)", camera.position.x, camera.position.y, camera.position.z);
    ImGui::Text("Rotation: (%.2f, %.2f, %.2f)", camera.rotation.x, camera.rotation.y, camera.rotation.z);
    ImGui::Separator();

    // NOTE: Ambient light control
    static veekay::vec3 g_ambient = { 0.15f, 0.15f, 0.15f };
    static float ambient_intensity = 0.15f;
    ImGui::Text("Ambient Light");
    ImGui::SliderFloat("Intensity##ambient", &ambient_intensity, 0.0f, 1.0f);
    ImGui::Separator();

    // NOTE: Directional light control (sun)
    static veekay::vec3 g_sun_dir = veekay::vec3::normalized({ -0.3f, 0.5f, -0.2f });
    static veekay::vec3 g_sun_color = {0.5f, 0.5f, 0.5f};
    static float dir_light_intensity = 1.0f;

    ImGui::Text("Directional Light");
    ImGui::SliderFloat3("Direction", &g_sun_dir.x, -1.0f, 1.0f);
    g_sun_dir = veekay::vec3::normalized(g_sun_dir);
    ImGui::ColorEdit3("Color##directional", &g_sun_color.x);
    ImGui::SliderFloat("Intensity##directional", &dir_light_intensity, 0.0f, 5.0f);
    ImGui::Separator();

    // NOTE: Point lights control
    ImGui::Text("Point Lights");

    for (size_t i = 0; i < point_lights.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));

        if (ImGui::TreeNode("Light", "Light %zu", i)) {
            ImGui::SliderFloat3("Position", &point_lights[i].position.x, -10.0f, 10.0f);
            ImGui::ColorEdit3("Color", &point_lights[i].color.x);
            ImGui::SliderFloat("Radius", &point_lights[i].radius, 0.1f, 50.0f);

            if (ImGui::Button("Remove")) {
                point_lights.erase(point_lights.begin() + i);
                ImGui::TreePop();
                ImGui::PopID();
                break;
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    if (point_lights.size() < 16 && ImGui::Button("Add Point Light")) {
        point_lights.push_back(PointLight{
            .position = {0.0f, -2.0f, 0.0f},
            .radius = 10.0f,
            .color = {1.0f, 1.0f, 1.0f},
        });
    }

    ImGui::Separator();

    // NOTE: Spotlights control
    ImGui::Text("Spotlights");

    // NOTE: Store per-spotlight angle state
    static std::vector<float> spotlight_angles;

    // NOTE: Sync angles vector with spotlights vector
    if (spotlight_angles.size() != spot_lights.size()) {
        spotlight_angles.resize(spot_lights.size());
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            // NOTE: Convert cosine back to degrees
            float angle_cos = spot_lights[i].angle;
            spotlight_angles[i] = std::acos(angle_cos) * 180.0f / float(M_PI);
        }
    }

    for (size_t i = 0; i < spot_lights.size(); ++i) {
        ImGui::PushID(static_cast<int>(i + 1000));

        if (ImGui::TreeNode("Spotlight", "Spotlight %zu", i)) {
            ImGui::SliderFloat3("Position", &spot_lights[i].position.x, -10.0f, 10.0f);
            ImGui::SliderFloat3("Direction", &spot_lights[i].direction.x, -1.0f, 1.0f);
            spot_lights[i].direction = veekay::vec3::normalized(spot_lights[i].direction);

            ImGui::ColorEdit3("Color", &spot_lights[i].color.x);
            ImGui::SliderFloat("Radius", &spot_lights[i].radius, 0.1f, 50.0f);

            // NOTE: Use per-spotlight angle storage
            float& angle = spotlight_angles[i];

            if (ImGui::SliderFloat("Cone Angle", &angle, 5.0f, 90.0f)) {
                spot_lights[i].angle = std::cos(toRadians(angle));
            }

            if (ImGui::Button("Remove")) {
                spot_lights.erase(spot_lights.begin() + i);
                spotlight_angles.erase(spotlight_angles.begin() + i);
                ImGui::TreePop();
                ImGui::PopID();
                break;
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    if (spot_lights.size() < 8 && ImGui::Button("Add Spotlight")) {
        spot_lights.push_back(SpotLight{
            .position = {0.0f, -2.0f, 0.0f},
            .radius = 10.0f,
            .direction = {0.0f, 1.0f, 0.0f},
            .angle = std::cos(toRadians(25.0f)),
            .color = {1.0f, 1.0f, 1.0f},
        });
        spotlight_angles.push_back(25.0f);
    }

    ImGui::Separator();

    // NOTE: Material properties control
    // static float material_shininess = 8.0f;
    // ImGui::Text("Material Properties");
    // ImGui::SliderFloat("Shininess", &material_shininess, 1.0f, 128.0f);

    ImGui::End();

    if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
        using namespace veekay::input;

        if (mouse::isButtonDown(mouse::Button::left)) {
            auto move_delta = mouse::cursorDelta();

			camera.rotation.y -= move_delta.x * 0.2f;
			camera.rotation.x -= move_delta.y * 0.2f;

            constexpr float max_pitch = 90.0f;
            if (camera.rotation.x > max_pitch) camera.rotation.x = max_pitch;
            if (camera.rotation.x < -max_pitch) camera.rotation.x = -max_pitch;
        }

        auto view = camera.view();

        veekay::vec3 right, up, front;

        right = {view.elements[0][0], view.elements[1][0], view.elements[2][0]};
        up = {view.elements[0][1], view.elements[1][1], view.elements[2][1]};
        front = {view.elements[0][2], view.elements[1][2], view.elements[2][2]};


		// // NOTE: Project forward vector onto horizontal plane for WASD movement
        // veekay::vec3 forward_horizontal = {front.x, 0.0f, front.z};
        // forward_horizontal = veekay::vec3::normalized(forward_horizontal);

        if (keyboard::isKeyDown(keyboard::Key::w))
            camera.position += front * 0.1f;

        if (keyboard::isKeyDown(keyboard::Key::s))
            camera.position -= front * 0.1f;

        if (keyboard::isKeyDown(keyboard::Key::d))
            camera.position += right * 0.1f;

        if (keyboard::isKeyDown(keyboard::Key::a))
            camera.position -= right * 0.1f;

        if (keyboard::isKeyDown(keyboard::Key::q))
            camera.position -= up * 0.1f;

        if (keyboard::isKeyDown(keyboard::Key::e))
            camera.position += up * 0.1f;
    }

    // NOTE: Calculate shadow matrix
    veekay::mat4 shadow_matrix;
    {
        // NOTE: Direction FROM scene TO sun
        veekay::vec3 sun_dir = -g_sun_dir;
        veekay::vec3 scene_center{0.0f, 0.0f, 0.0f};
        float light_distance = 15.0f;

        // NOTE: Light position: FROM scene TO sun
        veekay::vec3 light_position = scene_center + sun_dir * light_distance;

        // NOTE: Light forward direction: FROM light TO scene
        veekay::vec3 light_forward = sun_dir;

        veekay::mat4 light_view = lookat(light_forward, light_position);

        // NOTE: Create orthographic projection for shadow map
        float ortho_size = 10.0f;
        float z_near = 1.0f;
        float z_far = 30.0f;

        veekay::mat4 light_ortho = veekay::mat4::orthographic(
            -ortho_size, ortho_size,
            -ortho_size, ortho_size,
            z_near, z_far
        );

        shadow_matrix = light_view * light_ortho;

        // NOTE: Update shadow uniform buffer
        *(veekay::mat4*)shadow.uniform_buffer->mapped_region = shadow_matrix;
    }

    float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);

    // NOTE: Apply intensities to light colors
    veekay::vec3 final_ambient = g_ambient * ambient_intensity;
    veekay::vec3 final_sun_color = g_sun_color * dir_light_intensity;

    SceneUniforms scene_uniforms{
        .view_projection = camera.view_projection(aspect_ratio),
        .shadow_projection = shadow_matrix,
        .view_position = camera.position,
        .ambient_light_intensity = final_ambient,
        .sun_light_direction = g_sun_dir,
        .sun_light_color = final_sun_color,
        .point_lights_count = static_cast<uint32_t>(point_lights.size()),
        .spot_lights_count = static_cast<uint32_t>(spot_lights.size()),
    };

    std::vector<ModelUniforms> model_uniforms(models.size());
    for (size_t i = 0, n = models.size(); i < n; ++i) {
        const Model& model = models[i];
        ModelUniforms& uniforms = model_uniforms[i];

        uniforms.model = model.transform.matrix();
        uniforms.albedo_color = model.albedo_color;
        uniforms.specular_color = {1.0f, 1.0f, 1.0f};
        uniforms.shininess = 8.0f;
    }

    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

    const size_t alignment =
        veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

    for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
        const ModelUniforms& uniforms = model_uniforms[i];

        char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
        *reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
    }

    // NOTE: Update point lights buffer
    if (!point_lights.empty()) {
        std::memcpy(point_lights_buffer->mapped_region, point_lights.data(),
                    point_lights.size() * sizeof(PointLight));
    }

	//
	if (!spot_lights.empty()) {
    std::memcpy(spot_lights_buffer->mapped_region, spot_lights.data(),
                spot_lights.size() * sizeof(SpotLight));
}
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{
		VkImageMemoryBarrier barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow.depth_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};

		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);

		VkRenderingAttachmentInfoKHR depth_attachment{
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
			.imageView = shadow.depth_image_view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = {.depthStencil = {1.0f, 0}},
		};

		VkRenderingInfoKHR rendering_info{
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
			.renderArea = {
				.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE},
			},
			.layerCount = 1,
			.pDepthAttachment = &depth_attachment,
		};

		vkCmdBeginRenderingKHR(cmd, &rendering_info);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline);

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = float(SHADOW_MAP_SIZE),
			.height = float(SHADOW_MAP_SIZE),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		vkCmdSetViewport(cmd, 0, 1, &viewport);

		VkRect2D scissor{
			.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE},
		};
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		vkCmdSetDepthBias(cmd, 1.25f, 0.0f, 1.0f);

		VkDeviceSize zero_offset = 0;
		VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
		VkBuffer current_index_buffer = VK_NULL_HANDLE;

		const size_t model_uniforms_alignment =
			veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

		for (size_t i = 0, n = models.size(); i < n; ++i) {
			const Model& model = models[i];
			const Mesh& mesh = model.mesh;

			if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
				current_vertex_buffer = mesh.vertex_buffer->buffer;
				vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
			}

			if (current_index_buffer != mesh.index_buffer->buffer) {
				current_index_buffer = mesh.index_buffer->buffer;
				vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
			}

			uint32_t offset = i * model_uniforms_alignment;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline_layout,
									0, 1, &shadow.descriptor_set, 1, &offset);

			vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
		}

		vkCmdEndRenderingKHR(cmd);

		barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * model_uniorms_alignment;
		VkDescriptorSet material_descriptor_set = material_descriptor_sets[model.material_id];
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &material_descriptor_set, 1, &offset);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}