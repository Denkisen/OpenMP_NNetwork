#ifndef __CPU_NW_LIBS_VULKAN_ISTORAGE_H
#define __CPU_NW_LIBS_VULKAN_ISTORAGE_H

#include <vulkan/vulkan.h>
#include <iostream>

namespace Vulkan
{
  enum class StorageType
  {
    Default,
    Uniform
  };

  class IStorage
  {
  protected:
    StorageType type = StorageType::Default;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice p_device = VK_NULL_HANDLE;
    uint32_t buffer_size = 0;
    uint32_t family_queue = 0;

    template <typename O> friend class Offload;
  public:
      IStorage() = default;
      ~IStorage()
      {
#ifdef DEBUG
        std::cout << __func__ << std::endl;
#endif
        if (device != VK_NULL_HANDLE)
        {
          vkFreeMemory(device, buffer_memory, NULL);
          vkDestroyBuffer(device, buffer, NULL);
          device = VK_NULL_HANDLE;
        }
      }
  };
}

#endif