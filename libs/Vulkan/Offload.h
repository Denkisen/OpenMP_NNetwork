#ifndef __CPU_NW_LIBS_VULKAN_OFFLOAD_H
#define __CPU_NW_LIBS_VULKAN_OFFLOAD_H

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>

#include "Instance.h"
#include "Device.h"
#include "Array.h"

namespace Vulkan
{
  template <typename T> class Offload
  {
  private:
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule compute_shader = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkPhysicalDeviceLimits device_limits = {};
    std::vector<Array<T>*> buffers;
  public:
    Offload() = delete;
    Offload(Device &dev, std::vector<Array<T>*> &data, std::string shader_path);
    ~Offload()
    {
#ifdef DEBUG
      std::cout << __func__ << std::endl;
#endif
      if (device != VK_NULL_HANDLE)
      {
        vkDestroyShaderModule(device, compute_shader, NULL);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, command_pool, NULL);
      }
      device = VK_NULL_HANDLE;
    }
    void Run();
  };
}

#endif