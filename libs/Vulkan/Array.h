#ifndef __CPU_NW_LIBS_VULKAN_ARRAY_H
#define __CPU_NW_LIBS_VULKAN_ARRAY_H

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>

#include "Device.h"


namespace Vulkan
{
  template <typename T> class Array
  {
  private:
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice p_device = VK_NULL_HANDLE;
    uint32_t buffer_size = 0;
    uint32_t family_queue = 0;
    void Create(Device &dev, T *data, size_t len);
    void Create(VkDevice dev, VkPhysicalDevice p_dev, T *data, size_t len, uint32_t f_queue);
    template <typename O> friend class Offload;
  public:
    Array() = delete;
    Array(Device &dev, std::vector<T> &data);
    Array(Device &dev, T *data, size_t len);
    Array(const Array<T> &array);
    Array<T>& operator= (const Array<T> &obj);
    std::vector<T> Extract() const;
    ~Array()
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