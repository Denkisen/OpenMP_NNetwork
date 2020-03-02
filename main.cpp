#include <iostream>
#include <cmath>
#include "Networks/MultiLayerPerceptron.h"
#include "Learning/BackProp.h"
#include "Functions/activation_functions.h"
#include "DataProviders/Images.h"
#include "libs/Math/SimpleMath.h"
#include "libs/SpeedTest.h"
#include "libs/Vulkan/Array.h"
#include "libs/Vulkan/Instance.h"
#include "libs/Vulkan/Device.h"

#include "libs/Vulkan/Offload.h"

#define CONF_FILE "conf.txt"
#define LOG_FILE "log.txt"
#define COMMENTS "Fix image 40x40 MLP."
#define TRAIN_SET "/home/marko/Documents/Projects/NeuralNetworks/Training_Set/set.txt"
#define TRAIN_SPEED 0.01
#define MOMENTUM 0.01


double ActFunc(double x)
{
  return Tanh(x, 0.1);//Leaky_ReLu(x, 0.01);
}

double DerFunc(double x)
{
  return Derivative_Tanh(x, 0.1);//Derivative_Leaky_ReLu(x, 0.01);
}

int main(int argc, char const *argv[])
{
//#pragma omp target teams distribute parallel for reduction(+:sum) map(tofrom:sum)
//#pragma omp parallel for reduction(+:sum)

  try 
  {
    Vulkan::Instance instance;
    Vulkan::Device device(instance, Vulkan::Discrete);
    std::vector<float> inp(64, 5.0);
    Vulkan::Array<float> input(device, inp);
    std::vector<float> out(64, 0.0);
    Vulkan::Array<float> output(device, out);
    std::vector<Vulkan::Array<float>*> data;
    data.push_back(&input);
    data.push_back(&output);
    Vulkan::Offload<float> offload(device, data, "bin/comp.spv");
    offload.Run();
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
  MultiLayerPerceptron net;
  BackProp train;
  net.SetActivationFunc(ActFunc);
  if (!net.Load(CONF_FILE))
  {
    net.SetLayersCount(4);
    net.AddLayer(4800);
    net.AddLayer(5400);
    net.AddLayer(6400);
    net.AddLayer(4800);
  }

  train.SetNetwork(&net);
  train.LogFile(LOG_FILE);
  train.SetActivationFunction(ActFunc, DerFunc);
  train.LearningSpeed(TRAIN_SPEED);
  train.Momentum(MOMENTUM);

  std::ifstream f(TRAIN_SET, std::ifstream::in);

  if (f.is_open())
  {
    std::string line = "";
    Image input;
    std::vector<double> inp;
    std::vector<double> out;
    std::vector<double> exp;
    while (true)
    {
      f >> line;
      if (line == "") break;

      input = OpenImage(line);
      if (input.canva != nullptr)
      {
        
        for (int y = 0; y < input.height - 40; y += 40)
        {
          for (int x = 0; x < input.width - 40; x += 40)
          {
            Image res = CutOfImage(input, x, y, 40, 40);
            Image cut = CorruptImage(res, 80, StripCorruptionFunction());

            inp.resize(cut.height * cut.width * cut.bpp);
            std::copy(cut.canva, &cut.canva[inp.size()], inp.begin());
            inp = DivVector(inp, 255.0);

            exp.resize(res.height * res.width * res.bpp);
            std::copy(res.canva, &res.canva[exp.size()], exp.begin());
            exp = DivVector(exp, 255);

            STOP_WATCH(start);
            out = train.DoItteration(inp, exp);
            STOP_WATCH(stop);
            PRINT_DIFF_WATCH(start, stop);
            double err = 0;
            for (size_t i = 0; i < out.size(); ++i)
            {
              err += ((out[i] - exp[i]) * (out[i] - exp[i]));
            }
            err = sqrt(err);
            std::cout << "X, Y :" << x << " " << y << std::endl;
            std::cout << "Error: " << err << std::endl;

            //SaveImage("test/res/res" + std::to_string(x) + ".jpg", cut);

            FreeImage(cut);
            FreeImage(res);
          }
        }
        net.Save(CONF_FILE, COMMENTS);
      }
    }
    net.Save(CONF_FILE, COMMENTS);
  }

  return 0;
}
