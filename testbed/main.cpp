#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace {

constexpr float camera_fov = 70.0f;
constexpr float camera_near_plane = 0.01f;
constexpr float camera_far_plane = 100.0f;

struct Vertex {
    veekay::vec3 position;
    veekay::vec3 color;  // Закомментируйте пока что
};

// NOTE: These variable will be available to shaders through push constant uniform
struct ShaderConstants {
	veekay::mat4 projection;
	veekay::mat4 transform;
	veekay::vec3 color;
};

VkShaderModule vertex_shader_module;
VkShaderModule fragment_shader_module;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

// NOTE: Declare buffers and other variables here
veekay::graphics::Buffer* vertex_buffer;
veekay::graphics::Buffer* index_buffer;

veekay::vec3 model_position = {0.0f, 0.0f, 5.0f};
float model_rotation;
veekay::vec3 model_color = {0.5f, 1.0f, 0.7f };
bool model_spin = true;
float view_angle = 0.0f;  // Угол для вида сбоку (в радианах)


// Параметры для дочерней пирамиды
veekay::vec3 child_offset = {2.0f, 0.0f, 5.0f};  // Смещение относительно родительской
float child_scale = 0.5f;                        // Масштаб дочерней пирамиды

// Для управления анимацией
bool animation_paused = false;
bool animation_reversed = false;
float animation_speed = 1.0f;
double accumulated_time = 0.0;
double last_time = 0.0;

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

void initialize() {
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
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, position),
			},
			{
				.location = 1,  // Атрибут цвета
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, color),
			},
		};
		// NOTE: Bring
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
			.cullMode = VK_CULL_MODE_NONE, // Доя корректного отображения дна квадрата
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

		// NOTE: Declare constant memory region visible to vertex and fragment shaders
		VkPushConstantRange push_constants{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
			              VK_SHADER_STAGE_FRAGMENT_BIT,
			.size = sizeof(ShaderConstants),
		};

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_constants,
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

	// TODO: You define model vertices and create buffers here
	// TODO: Index buffer has to be created here too
	// NOTE: Look for createBuffer function

	// (v0)------(v1)
	//  |  \       |
	//  |   `--,   |
	//  |       \  |
	// (v3)------(v2)
	Vertex vertices[] = {
		// Основание пирамиды - 2 треугольника (синий)
		{{-1.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 1.0f}}, // v0
		{{1.0f, 0.0f, -1.0f},  {0.0f, 0.0f, 1.0f}}, // v1
		{{1.0f, 0.0f, 1.0f},   {0.0f, 0.0f, 1.0f}}, // v2

		{{1.0f, 0.0f, 1.0f},   {0.0f, 0.0f, 1.0f}}, // v3
		{{-1.0f, 0.0f, 1.0f},  {0.0f, 0.0f, 1.0f}}, // v4
		{{-1.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 1.0f}}, // v5

		// Задняя грань - красный
		{{-1.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f}}, // v6
		{{1.0f, 0.0f, -1.0f},  {1.0f, 0.0f, 0.0f}}, // v7
		{{0.0f, -1.5f, 0.0f},   {1.0f, 0.0f, 0.0f}}, // v8

		// Правая грань - зеленый
		{{1.0f, 0.0f, -1.0f},  {0.0f, 1.0f, 0.0f}}, // v9
		{{1.0f, 0.0f, 1.0f},   {0.0f, 1.0f, 0.0f}}, // v10
		{{0.0f, -1.5f, 0.0f},   {0.0f, 1.0f, 0.0f}}, // v11

		// Передняя грань - желтый
		{{1.0f, 0.0f, 1.0f},   {1.0f, 1.0f, 0.0f}}, // v12
		{{-1.0f, 0.0f, 1.0f},  {1.0f, 1.0f, 0.0f}}, // v13
		{{0.0f, -1.5f, 0.0f},   {1.0f, 1.0f, 0.0f}}, // v14

		// Левая грань - фиолетовый
		{{-1.0f, 0.0f, 1.0f},  {1.0f, 0.0f, 1.0f}}, // v15
		{{-1.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 1.0f}}, // v16
		{{0.0f, -1.5f, 0.0f},   {1.0f, 0.0f, 1.0f}}, // v17
	};

	// Индексы для треугольников: основание + 4 боковых грани
	uint32_t indices[] = {
		// Основание (2 треугольника)
		0, 1, 2,
		3, 4, 5,

		// Боковые грани (по 1 треугольнику каждая)
		6, 7, 8,    // Задняя
		9, 10, 11,  // Правая
		12, 13, 14, // Передняя
		15, 16, 17  // Левая
	};

	vertex_buffer = new veekay::graphics::Buffer(sizeof(vertices), vertices,
	                                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	index_buffer = new veekay::graphics::Buffer(sizeof(indices), indices,
	                                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	// NOTE: Destroy resources here, do not cause leaks in your program!
	delete index_buffer;
	delete vertex_buffer;

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
    // Вычисляем дельту времени
    double delta_time = time - last_time;
    last_time = time;

    // Накопление времени только если анимация не на паузе
    if (!animation_paused) {
        accumulated_time += delta_time * animation_speed * (animation_reversed ? -1.0 : 1.0);
    }

	ImGui::Begin("Controls:");
	ImGui::InputFloat3("Translation", reinterpret_cast<float*>(&model_position));
	ImGui::SliderFloat("Rotation", &model_rotation, 0.0f, 2.0f * M_PI);

    // Управление анимацией
    ImGui::Separator();
    ImGui::Text("Animation Control:");

    // Кнопка паузы/возобновления
    if (ImGui::Button(animation_paused ? "Resume" : "Pause")) {
        animation_paused = !animation_paused;
    }

    // Кнопка реверса направления
    ImGui::SameLine();
    if (ImGui::Button(animation_reversed ? "Forward" : "Reverse")) {
        animation_reversed = !animation_reversed;
    }

    // Слайдер скорости анимации
    ImGui::SliderFloat("Animation Speed", &animation_speed, 0.0f, 3.0f, "%.1f");

    // Кнопка сброса анимации
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        accumulated_time = 0.0;
        animation_paused = false;
        animation_reversed = false;
        animation_speed = 1.0f;
    }



	// ImGui::Checkbox("Spin?", &model_spin);
    // Слайдер для вида сбоку (в градусах)
    ImGui::SliderFloat("View Angle", &view_angle, -90.0f, 90.0f, "%.1f deg");

    // UI для дочерней пирамиды
    ImGui::Separator();
    ImGui::Text("Child Pyramid:");
    ImGui::SliderFloat("Child Scale", &child_scale, 0.1f, 2.0f, "%.2f"); // диапазон до 2.0
    ImGui::InputFloat3("Child Offset", reinterpret_cast<float*>(&child_offset)); // смещение

    ImGui::End();


	// NOTE: Animation code and other runtime variable updates go here
    if (!animation_paused) {
        model_rotation = fmodf(float(accumulated_time), 2.0f * M_PI);
    }
    // model_rotation = fmodf(model_rotation, 2.0f * M_PI);

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

	// TODO: Vulkan rendering code here
	// NOTE: ShaderConstant updates, vkCmdXXX expected to be here
	{
		// NOTE: Use our new shiny graphics pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);


		// NOTE: Use our quad vertex buffer
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer->buffer, &offset);

		// NOTE: Use our quad index buffer
		vkCmdBindIndexBuffer(cmd, index_buffer->buffer, offset, VK_INDEX_TYPE_UINT32);

		// NOTE: Variables like model_XXX were declared globally
		veekay::mat4 base_transform =
			veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, view_angle * M_PI / 180.0f) *
			veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, model_rotation) *
			veekay::mat4::translation(model_position);

		ShaderConstants constants{
			.projection = veekay::mat4::projection(
				camera_fov,
				float(veekay::app.window_width) / float(veekay::app.window_height),
				camera_near_plane, camera_far_plane),

			.transform = base_transform,

			.color = {0.0f, 1.0f, 0.0f}, // Ярко-зеленый для родительской
		};

		// NOTE: Update constant memory with new shader constants
		vkCmdPushConstants(cmd, pipeline_layout,
		                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(ShaderConstants), &constants);

		// NOTE: Draw 18 indices (6 треугольников * 3 вершины), 1 group, no offsets
		vkCmdDrawIndexed(cmd, 18, 1, 0, 0, 0);


		// ДОЧЕРНЯЯ пирамиду
		float test_scale = 0.3f; // Фиксированный маленький масштаб


		ShaderConstants child_constants{
			.projection = veekay::mat4::projection(
				camera_fov,
				float(veekay::app.window_width) / float(veekay::app.window_height),
				camera_near_plane, camera_far_plane),

			.transform = veekay::mat4::scaling({child_scale, child_scale, child_scale}) * // слайдер для масштаба
						 veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, view_angle * M_PI / 180.0f) *
						 veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, model_rotation) *

						 veekay::mat4::translation(child_offset),                    // родительские преобразования


			.color = {1.0f, 0.0f, 0.0f}, // Ярко-красный для дочерней
		};

		vkCmdPushConstants(cmd, pipeline_layout,
						VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
						0, sizeof(ShaderConstants), &child_constants);

		vkCmdDrawIndexed(cmd, 18, 1, 0, 0, 0);
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
