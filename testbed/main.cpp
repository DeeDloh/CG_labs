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

namespace {

constexpr uint32_t max_models = 1024;

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
	float ambient_intensity;
	uint32_t point_light_count;
	uint32_t spotlight_count; 
    float _pad1; // NOTE: Padding for alignment

	veekay::vec3 light_direction; // NOTE: Direction TO the light (normalized)
    float _pad3;
    veekay::vec3 light_color;
    float light_intensity;
	    
    veekay::vec3 camera_position;
    float _pad4;
};

struct PointLight {
    veekay::vec3 position;
    float _pad0;
    veekay::vec3 color;
    float intensity;
};


struct Spotlight {
    veekay::vec3 position;
    float _pad0;
    veekay::vec3 direction;
    float _pad1;
    veekay::vec3 color;
    float intensity;
    float inner_cutoff;  // cos(inner_angle)
    float outer_cutoff;  // cos(outer_angle)
    float _pad2, _pad3;
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
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};
    veekay::vec3 target = {0.0f, -0.5f, 0.0f};
    // bool is_look_at = true;

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
    CameraMode current_camera_mode = CameraMode::LookAt;
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
	std::vector<Spotlight> spotlights;
	veekay::graphics::Buffer* spotlights_buffer;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

// NOTE: Helper function to clamp values (C++11 compatible)
template<typename T>
T clamp(T value, T min_val, T max_val) {
    return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
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
			-std::sin(pitch),
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
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
                    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 8,
                },
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				}
			};
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,
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
	

	spotlights_buffer = new veekay::graphics::Buffer(
    8 * sizeof(Spotlight),  // 8 прожекторов
    nullptr,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
	);

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

	{
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
				.buffer = spotlights_buffer->buffer,
				.offset = 0,
				.range = VK_WHOLE_SIZE,
			},
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[2],
            },
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[3],
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
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
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f}
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {-2.0f, -0.5f, -1.5f},
			.rotation = {0.0f, 45.0f, 0.0f},  // NOTE: Поворот на 45° вокруг Y

		},
		.albedo_color = veekay::vec3{1.0f, 0.0f, 0.0f}
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {1.5f, -0.5f, -0.5f},
			.scale = {1.5f, 0.5f, 1.0f},  // NOTE: Растянутый куб
		},
		.albedo_color = veekay::vec3{0.0f, 1.0f, 0.0f}
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {0.0f, -0.5f, 1.0f},
		},
		.albedo_color = veekay::vec3{0.0f, 0.0f, 1.0f}
	});

    // NOTE: Add point lights to the scene
    // point_lights.push_back(PointLight{
    //     .position = {-5.0f, -5.0f, 0.0f},
    //     .color = {0.0f, 0.0f, 1.0f},
    //     .intensity = 20.0f,
    // });

    point_lights.push_back(PointLight{
        .position = {5.0f, -5.0f, 0.0f},
        .color = {1.0f, 1.0f, 1.0f},
        .intensity = 100.0f,
    });

	spotlights.push_back(Spotlight{
		.position = {0.0f, -3.0f, 0.0f},
		.direction = {0.0f, 1.0f, 0.0f},  // Светит вверх
		.color = {1.0f, 1.0f, 0.8f},      // Тёплый белый
		.intensity = 50.0f,
		.inner_cutoff = std::cos(toRadians(0.0f)),
		.outer_cutoff = std::cos(toRadians(0.0f)),
	});
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete plane_mesh.index_buffer;
	delete plane_mesh.vertex_buffer;

    delete point_lights_buffer;
	delete spotlights_buffer;
	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
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
    static float ambient_intensity = 0.1f;
    ImGui::Text("Ambient Light");
    ImGui::SliderFloat("Intensity##ambient", &ambient_intensity, 0.0f, 1.0f);
    ImGui::Separator();
    
    // NOTE: Directional light control
    static float dir_light_dir[3] = {0.0f, -1.0f, 0.0f};
    static float dir_light_color[3] = {1.0f, 1.0f, 1.0f};
    static float dir_light_intensity = 2.0f;
    
    ImGui::Text("Directional Light");
    ImGui::SliderFloat3("Direction", dir_light_dir, -1.0f, 1.0f);
    ImGui::ColorEdit3("Color##directional", dir_light_color);
    ImGui::SliderFloat("Intensity##directional", &dir_light_intensity, 0.0f, 5.0f);
    ImGui::Separator();
    
    // NOTE: Point lights control
    ImGui::Text("Point Lights");
    
    for (size_t i = 0; i < point_lights.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        
        if (ImGui::TreeNode("Light", "Light %zu", i)) {
            ImGui::SliderFloat3("Position", &point_lights[i].position.x, -10.0f, 10.0f);
            ImGui::ColorEdit3("Color", &point_lights[i].color.x);
            ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 100.0f);
            
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
            .color = {1.0f, 1.0f, 1.0f},
            .intensity = 5.0f,
        });
    }

    ImGui::Separator();

    // NOTE: Spotlights control
    ImGui::Text("Spotlights");

    // NOTE: Store per-spotlight angle state
    static std::vector<std::pair<float, float>> spotlight_angles;
    
    // NOTE: Sync angles vector with spotlights vector
    if (spotlight_angles.size() != spotlights.size()) {
        spotlight_angles.resize(spotlights.size());
        for (size_t i = 0; i < spotlights.size(); ++i) {
            // NOTE: Convert cosine back to degrees, with clamping to prevent acos domain errors
            float inner_cos = clamp(spotlights[i].inner_cutoff, 0.0f, 1.0f);
            float outer_cos = clamp(spotlights[i].outer_cutoff, 0.0f, 1.0f);
            spotlight_angles[i].first = std::acos(inner_cos) * 180.0f / float(M_PI);
            spotlight_angles[i].second = std::acos(outer_cos) * 180.0f / float(M_PI);
        }
    }

    for (size_t i = 0; i < spotlights.size(); ++i) {
        ImGui::PushID(static_cast<int>(i + 1000));
        
        if (ImGui::TreeNode("Spotlight", "Spotlight %zu", i)) {
            ImGui::SliderFloat3("Position", &spotlights[i].position.x, -10.0f, 10.0f);
            ImGui::SliderFloat3("Direction", &spotlights[i].direction.x, -1.0f, 1.0f);
            spotlights[i].direction = veekay::vec3::normalized(spotlights[i].direction);
            
            ImGui::ColorEdit3("Color", &spotlights[i].color.x);
            ImGui::SliderFloat("Intensity", &spotlights[i].intensity, 0.0f, 200.0f);
            
            // NOTE: Use per-spotlight angle storage
            float& inner_angle = spotlight_angles[i].first;
            float& outer_angle = spotlight_angles[i].second;
            
            if (ImGui::SliderFloat("Inner Angle", &inner_angle, 0.0f, 45.0f)) {
                spotlights[i].inner_cutoff = std::cos(toRadians(inner_angle));
                if (outer_angle < inner_angle + 1.0f) {
                    outer_angle = inner_angle + 1.0f;
                    spotlights[i].outer_cutoff = std::cos(toRadians(outer_angle));
                }
            }
            
            if (ImGui::SliderFloat("Outer Angle", &outer_angle, inner_angle + 1.0f, 45.0f)) {
                spotlights[i].outer_cutoff = std::cos(toRadians(outer_angle));
            }
            
            if (ImGui::Button("Remove")) {
                spotlights.erase(spotlights.begin() + i);
                spotlight_angles.erase(spotlight_angles.begin() + i);
                ImGui::TreePop();
                ImGui::PopID();
                break;
            }
            
            ImGui::TreePop();
        }
        
        ImGui::PopID();
    }

    if (spotlights.size() < 8 && ImGui::Button("Add Spotlight")) {
        spotlights.push_back(Spotlight{
            .position = {0.0f, -2.0f, 0.0f},
            .direction = {0.0f, 1.0f, 0.0f},
            .color = {1.0f, 1.0f, 1.0f},
            .intensity = 50.0f,
            .inner_cutoff = std::cos(toRadians(12.5f)),
            .outer_cutoff = std::cos(toRadians(17.5f)),
        });
        spotlight_angles.push_back({12.5f, 17.5f});
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

			camera.rotation.y -= move_delta.x * 0.2f;  // Инверсия для Transform
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

    float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
    SceneUniforms scene_uniforms{
        .view_projection = camera.view_projection(aspect_ratio),
        .ambient_intensity = ambient_intensity,
        .point_light_count = static_cast<uint32_t>(point_lights.size()),
	    .spotlight_count = static_cast<uint32_t>(spotlights.size()),


        .light_direction = veekay::vec3::normalized({dir_light_dir[0], dir_light_dir[1], dir_light_dir[2]}),
        .light_color = {dir_light_color[0], dir_light_color[1], dir_light_color[2]},
        .light_intensity = dir_light_intensity,

        .camera_position = camera.position,
    };

    std::vector<ModelUniforms> model_uniforms(models.size());
    for (size_t i = 0, n = models.size(); i < n; ++i) {
        const Model& model = models[i];
        ModelUniforms& uniforms = model_uniforms[i];

        uniforms.model = model.transform.matrix();
        uniforms.albedo_color = model.albedo_color;
        uniforms.specular_color = {1.0f, 1.0f, 1.0f};
        uniforms.shininess = 8.0f;
		// uniforms.shininess = material_shininess;
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
	if (!spotlights.empty()) {
    std::memcpy(spotlights_buffer->mapped_region, spotlights.data(),
                spotlights.size() * sizeof(Spotlight));
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
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &descriptor_set, 1, &offset);

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
