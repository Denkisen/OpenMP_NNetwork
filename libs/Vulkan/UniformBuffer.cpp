#include "UniformBuffer.h"
#include <cstring>

namespace Vulkan
{
  void UniformBuffer::Create(Device &dev, void *data, size_t len)
  {
    Create(dev.device, dev.p_device, data, len, dev.family_queue);
  }

  void UniformBuffer::Create(VkDevice dev, VkPhysicalDevice p_dev, void *data, size_t len, uint32_t f_queue)
  {
    if (len == 0 || data == nullptr || p_dev == VK_NULL_HANDLE)
      throw std::runtime_error("Data array is empty.");

    buffer_size = len;
    VkPhysicalDeviceMemoryProperties properties;
    vkGetPhysicalDeviceMemoryProperties(p_dev, &properties);

    size_t memory_type_index = VK_MAX_MEMORY_TYPES;

    for (size_t i = 0; i < properties.memoryTypeCount; i++) 
    {
      if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & properties.memoryTypes[i].propertyFlags) &&
         (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & properties.memoryTypes[i].propertyFlags) &&
         (buffer_size < properties.memoryHeaps[properties.memoryTypes[i].heapIndex].size)) 
      {
        memory_type_index = i;
        break;
      }
    }

    if (memory_type_index == VK_MAX_MEMORY_TYPES)
      throw std::runtime_error("Out of memory.");

    VkMemoryAllocateInfo memory_allocate_info = {
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      0,
      buffer_size,
      (uint32_t) memory_type_index
    };

    if (vkAllocateMemory(dev, &memory_allocate_info, nullptr, &buffer_memory) != VK_SUCCESS)
      throw std::runtime_error("Can't allocate memory");
    
    void *payload = nullptr;
    if (vkMapMemory(dev, buffer_memory, 0, VK_WHOLE_SIZE, 0, &payload) != VK_SUCCESS)
      throw std::runtime_error("Can't map memory.");
    
    std::memcpy(payload, data, buffer_size);
    vkUnmapMemory(dev, buffer_memory);

    VkBufferCreateInfo buffer_create_info = {
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      nullptr,
      0,
      buffer_size,
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_SHARING_MODE_EXCLUSIVE,
      1,
      &f_queue
    };

    if (vkCreateBuffer(dev, &buffer_create_info, nullptr, &buffer) != VK_SUCCESS)
      throw std::runtime_error("Can't create Buffer.");

    VkMemoryRequirements mem_req = {};
    vkGetBufferMemoryRequirements(dev, buffer, &mem_req);
    if (buffer_size < mem_req.size)
      throw std::runtime_error("Buffer size is incorrect, required is " + std::to_string(mem_req.size));

    if (vkBindBufferMemory(dev, buffer, buffer_memory, 0) != VK_SUCCESS)
      throw std::runtime_error("Can't bind memory to buffer.");

    device = dev;
    family_queue = f_queue;
    p_device = p_dev;
    type = StorageType::Uniform; // VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
  }

  UniformBuffer::UniformBuffer(Device &dev, void *data, size_t len)
  {
    Create(dev, data, len);
  }

  UniformBuffer::UniformBuffer(const UniformBuffer &obj)
  {
    if (device != VK_NULL_HANDLE)
    {
      vkFreeMemory(device, buffer_memory, NULL);
      vkDestroyBuffer(device, buffer, NULL);
      device = VK_NULL_HANDLE;
    }

    char *tmp = new char[buffer_size];
    Extract((void *) tmp);
    Create(obj.device, obj.p_device, (void *) tmp, buffer_size, obj.family_queue);
    delete tmp;
  }

  UniformBuffer& UniformBuffer::operator= (const UniformBuffer &obj)
  {
    if (device != VK_NULL_HANDLE)
    {
      vkFreeMemory(device, buffer_memory, NULL);
      vkDestroyBuffer(device, buffer, NULL);
      device = VK_NULL_HANDLE;
    }
    
    char *tmp = new char[buffer_size];
    Extract((void *) tmp);
    Create(obj.device, obj.p_device, (void *) tmp, buffer_size, obj.family_queue);
    delete tmp;
    
    return *this;
  }

  void UniformBuffer::Extract(void *out) const
  {
    void *payload = nullptr;
    if (vkMapMemory(device, buffer_memory, 0, VK_WHOLE_SIZE, 0, &payload) != VK_SUCCESS)
      throw std::runtime_error("Can't map memory.");
    
    std::memcpy(out, payload, buffer_size);
    vkUnmapMemory(device, buffer_memory);
  }
}