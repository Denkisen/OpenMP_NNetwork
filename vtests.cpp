#include "Units.h"
#include "libs/SpeedTest.h"
#include "libs/Vulkan/Array.h"
#include "libs/Vulkan/UniformBuffer.h"
#include "libs/Vulkan/Instance.h"
#include "libs/Vulkan/Device.h"
#include "libs/Vulkan/Offload.h"

#include <iostream>
#include <cmath>
#include <unistd.h>
#include <algorithm>
#include <random>
#include <vector>

int vtest_main()
{
  try 
  { 
    Vulkan::Instance instance;
    Vulkan::Device device(instance, Vulkan::Discrete);
    
    std::vector<float> inp(64, 5.0);
    Vulkan::Array<float> input(device, inp);
    std::vector<float> out(64, 0.0);
    Vulkan::Array<float> output(device, out);
    struct UniformData
    {
      unsigned mul;
      unsigned times;
      unsigned val[62];
    };
    UniformData global_data = {};
    global_data.mul = 4;
    global_data.times = 1;
    Vulkan::UniformBuffer global(device, &global_data, sizeof(UniformData));
    std::vector<Vulkan::IStorage*> data;
    data.push_back(&input);
    data.push_back(&output);
    data.push_back(&global);
    Vulkan::Offload<float> offload(device, data, "bin/comp.spv");

    Vulkan::OffloadPipelineOptions opts;
    opts.DispatchTimes = 3;
    Vulkan::UpdateBufferOpt uni_opts;
    uni_opts.index = data.size() - 1;
    uni_opts.OnDispatchEndEvent = [](const size_t i, const size_t index, Vulkan::IStorage &buff)
    {
      if (buff.Type() == Vulkan::StorageType::Uniform)
      {
        UniformData tmp = {};
        reinterpret_cast<Vulkan::UniformBuffer&> (buff).Extract(&tmp);
        tmp.mul += i;
        reinterpret_cast<Vulkan::UniformBuffer&> (buff).Update(&tmp);  
      }
    };
    opts.DispatchEndEvents.push_back(uni_opts);

    offload.SetPipelineOptions(opts);
    offload.Run(64, 1, 1);
    out = output.Extract();
    std::cout << "Output:" << std::endl;
    for (size_t i = 0; i < out.size(); ++i)
    {
      std::cout << out[i] << " ";
    }
    std::cout << std::endl;
  }
  catch(const std::runtime_error& e)
  {
    std::cout << e.what() << std::endl;
  }
  std::cout << "Spok" << std::endl;

  return 0;
}