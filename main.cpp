#include <iostream>
#include "Networks/MultiLayerPerceptron.h"
#include "Learning/BackProp.h"
#include "Functions/activation_functions.h"
#include "DataProviders/Images.h"

double ActFunc(double x)
{
  return Leaky_ReLu(x, 0.01);
}

double DerFunc(double x)
{
  return Derivative_Leaky_ReLu(x, 0.01);
}

int main(int argc, char const *argv[])
{
//#pragma omp target teams distribute parallel for reduction(+:sum) map(tofrom:sum)
//#pragma omp parallel for reduction(+:sum)

  Image im = OpenImage("test/175238.jpg");
  if (im.canva != nullptr)
  {
    std::cout << "Size: " << im.width << "x" << im.height << std::endl;  
    Image res = GrayscaleImage(im);
    SaveImage("test/res.jpg", res);
    FreeImage(im);
    FreeImage(res);
  }


  return 0;

  MultiLayerPerceptron net;
  net.SetActivationFunc(ActFunc); 
  net.SetWeightInitFunc([](){ return 0.5; });
  //net.Load("test.txt");
  net.AddLayer(5);
  net.AddLayer(10);
  net.AddLayer(6);

  BackProp back;
  back.SetNetwork(&net);
  back.SetActivationFunction(ActFunc, DerFunc);
  back.LearningSpeed(0.01);
  back.LogFile("log.txt");
  Weights w = nullptr;
  std::vector<size_t> w_layout;  
  //ValueTable temp = nullptr;
  //std::vector<size_t> t_layout;

  
  w = net.GetWeights(w_layout);
  std::vector<double> input;
  input.push_back(5.0);
  input.push_back(5.0);
  input.push_back(5.0);
  input.push_back(5.0);
  input.push_back(5.0);

  std::vector<double> expect;
  expect.push_back(5.0);
  expect.push_back(5.0);
  expect.push_back(5.0);
  expect.push_back(5.0);
  expect.push_back(5.0);
  expect.push_back(5.0);

  std::cout << std::endl;
  std::cout << std::endl;
  
  // for (size_t i = 0; i < w_layout.size(); ++i) // Layer
  // {
  //   for (size_t j = 0; j < w_layout[i]; ++j) // Neuron
  //   {
  //     if (i > 0)
  //     {
  //       std::cout << "Neuron:" << std::endl;
  //       for (size_t l = 0; l < w_layout[i - 1]; ++l) // Back
  //       {
  //         std::cout << w[i][j][l] << " ";
  //       }
  //       std::cout << std::endl;
  //     }
  //     else
  //     {
  //       std::cout << w[i][j][0] << std::endl;
  //     }
  //   }
  // }


  for (size_t i = 0; i < 2; ++i)
  {
    back.DoItteration(input, expect);
  }

  std::cout << std::endl;
  std::cout << std::endl;
  
  for (size_t i = 0; i < w_layout.size(); ++i) // Layer
  {
    for (size_t j = 0; j < w_layout[i]; ++j) // Neuron
    {
      if (i > 0)
      {
        std::cout << "Neuron:" << std::endl;
        for (size_t l = 0; l < w_layout[i - 1]; ++l) // Back
        {
          std::cout << w[i][j][l] << " ";
        }
        std::cout << std::endl;
      }
      else
      {
        std::cout << w[i][j][0] << std::endl;
      }
    }
  }
  //std::vector<double> result = net.Pass(input);

  // if (temp != nullptr)
  // {
  //   std::cout << "Temp: " << t_layout.size() << std::endl;
  //   for (size_t i = 0; i < t_layout.size(); ++i)
  //   {
  //     std::cout << "Temp layer: " << t_layout[i] << std::endl;
  //     for (size_t j = 0; j < t_layout[i]; ++j)
  //     {
  //       std::cout << temp[i][j] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }

  // std::cout << "Result:" << std::endl;
  // for (size_t i = 0; i < result.size(); ++i)
  // {
  //   std::cout << result[i] << " ";
  // }
  // std::cout << std::endl;
  //net.Save("test.txt");
  return 0;
}
