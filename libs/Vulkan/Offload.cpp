#include "Offload.h"

namespace Vulkan
{
  template class Offload<int>;
  template class Offload<float>;
  template class Offload<double>;

  template <typename T>
  Offload<T>::Offload(Device &dev, std::vector<Array<T>*> &data, std::string shader_path)
  {
    buffers = data;
    device = dev.device;
    device_limits = dev.device_limits;
    queue = dev.queue;
    std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i)
    {
      VkDescriptorSetLayoutBinding descriptor_set_layout_binding = {};
      descriptor_set_layout_binding.binding = i;
      descriptor_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_set_layout_binding.descriptorCount = 1;
      descriptor_set_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      descriptor_set_layout_bindings[i] = descriptor_set_layout_binding;
    }

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};
    descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_set_layout_create_info.bindingCount = (uint32_t) descriptor_set_layout_bindings.size(); // only a single binding in this descriptor set layout. 
    descriptor_set_layout_create_info.pBindings = descriptor_set_layout_bindings.data(); 

    if (vkCreateDescriptorSetLayout(dev.device, &descriptor_set_layout_create_info, nullptr, &descriptor_set_layout) != VK_SUCCESS)
      throw std::runtime_error("Can't create DescriptorSetLayout.");

    VkDescriptorPoolSize descriptor_pool_size = {};
    descriptor_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_pool_size.descriptorCount = (uint32_t) buffers.size();

    VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
    descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_create_info.maxSets = 1;
    descriptor_pool_create_info.poolSizeCount = 1;
    descriptor_pool_create_info.pPoolSizes = &descriptor_pool_size;

    if (vkCreateDescriptorPool(dev.device, &descriptor_pool_create_info, nullptr, &descriptor_pool) != VK_SUCCESS)
      throw std::runtime_error("Can't creat DescriptorPool.");
    
    VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {};
    descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; 
    descriptor_set_allocate_info.descriptorPool = descriptor_pool; // pool to allocate from.
    descriptor_set_allocate_info.descriptorSetCount = 1; // allocate a single descriptor set.
    descriptor_set_allocate_info.pSetLayouts = &descriptor_set_layout;

    if (vkAllocateDescriptorSets(dev.device, &descriptor_set_allocate_info, &descriptor_set) != VK_SUCCESS)
      throw std::runtime_error("Can't allocate DescriptorSets.");

    std::vector<VkDescriptorBufferInfo> descriptor_buffer_infos(buffers.size());
    std::vector<VkWriteDescriptorSet> write_descriptor_set(buffers.size());

    for (size_t i = 0; i < buffers.size(); ++i)
    {
      VkDescriptorBufferInfo descriptor_buffer_info = {};
      descriptor_buffer_info.buffer = (*buffers[i]).buffer;
      descriptor_buffer_info.offset = 0;
      descriptor_buffer_info.range = VK_WHOLE_SIZE;
      descriptor_buffer_infos[i] = descriptor_buffer_info;

      VkWriteDescriptorSet write_descriptor = {};
      write_descriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write_descriptor.dstSet = descriptor_set;
      write_descriptor.dstBinding = i;
      write_descriptor.descriptorCount = 1;
      write_descriptor.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      write_descriptor.pBufferInfo = &descriptor_buffer_infos[i];

      write_descriptor_set[i] = write_descriptor;
    }

    vkUpdateDescriptorSets(dev.device, (uint32_t) write_descriptor_set.size(), write_descriptor_set.data(), 0, nullptr);

    if (Supply::LoadPrecompiledShaderFromFile(dev.device, shader_path, compute_shader) != VK_SUCCESS)
      throw std::runtime_error("Can't Shader from file.");

    VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
    shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_create_info.module = compute_shader;
    shader_stage_create_info.pName = "main";

    VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
    pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_create_info.setLayoutCount = 1;
    pipeline_layout_create_info.pSetLayouts = &descriptor_set_layout; 
    if (vkCreatePipelineLayout(dev.device, &pipeline_layout_create_info, nullptr, &pipeline_layout) != VK_SUCCESS)
      throw std::runtime_error("Can't create pipeline layout.");

    VkComputePipelineCreateInfo pipeline_create_info = {};
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.stage = shader_stage_create_info;
    pipeline_create_info.layout = pipeline_layout;

    if (vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &pipeline) != VK_SUCCESS)
      throw std::runtime_error("Can't create compute pipelines.");

    VkCommandPoolCreateInfo command_pool_create_info = {};
    command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_create_info.flags = 0;
    command_pool_create_info.queueFamilyIndex = dev.family_queue;

    if (vkCreateCommandPool(dev.device, &command_pool_create_info, NULL, &command_pool) != VK_SUCCESS)
      throw std::runtime_error("Can't create command pool.");    
      
    VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
    command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.commandPool = command_pool; 
    command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_allocate_info.commandBufferCount = 1; 
    if (vkAllocateCommandBuffers(dev.device, &command_buffer_allocate_info, &command_buffer) != VK_SUCCESS)
      throw std::runtime_error("Can't allocate command buffers.");
  }

  template <typename T>
  void Offload<T>::Run()
  {
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS)
      throw std::runtime_error("Can't begin command buffer.");

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    //device_limits.maxComputeWorkGroupCount[0]
    vkCmdDispatch(command_buffer, 64, 1, 1);

    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
      throw std::runtime_error("Can't end command buffer.");

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    VkFence fence;
    VkFenceCreateInfo fence_create_info = {};
    fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_create_info.flags = 0;
    if (vkCreateFence(device, &fence_create_info, nullptr, &fence) != VK_SUCCESS)
      throw std::runtime_error("Can't create fence.");

    if (vkQueueSubmit(queue, 1, &submit_info, fence) != VK_SUCCESS)
      throw std::runtime_error("Can't submit queue.");

    if (vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000) != VK_SUCCESS)
      throw std::runtime_error("WaitForFences error");

    vkDestroyFence(device, fence, nullptr);
  }
}