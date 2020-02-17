#include <iostream>
#include "Networks/MultiLayerPerceptron.h"

int main(int argc, char const *argv[])
{
//#pragma omp target teams distribute parallel for reduction(+:sum) map(tofrom:sum)
//#pragma omp parallel for reduction(+:sum)
  MultiLayerPerceptron net;
  Weights w = nullptr;
  std::vector<size_t> w_layout;  
  ValueTable temp = nullptr;
  std::vector<size_t> t_layout;
  net.Load("test.txt");
  // net.AddLayer(5);
  // net.AddLayer(10);
  // net.AddLayer(6);
  
  w = net.GetWeights(w_layout);
  std::vector<double> input;
  input.push_back(1);
  input.push_back(2);
  input.push_back(3);
  input.push_back(4);
  input.push_back(5);

  std::cout << std::endl;
  std::cout << std::endl;
  
  // for (size_t i = 0; i < w_layout.size(); ++i) // Layer
  // {
  //   //std::cout << "Layer" << i << " layout:" << std::endl;
  //   for (size_t j = 0; j < w_layout[i]; ++j) // Neuron
  //   {
  //     //std::cout << "Neuron" << j << " back layout:" << std::endl;
  //     if (i > 0)
  //     {
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

  std::vector<double> result = net.Pass(input, temp, t_layout);

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

  std::cout << "Result:" << std::endl;
  for (size_t i = 0; i < result.size(); ++i)
  {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  net.Bias(false);
  result.clear();
  result = net.Pass(input, temp, t_layout);

  if (temp != nullptr)
  {
    std::cout << "Temp: " << t_layout.size() << std::endl;
    for (size_t i = 0; i < t_layout.size(); ++i)
    {
      std::cout << "Temp layer: " << t_layout[i] << std::endl;
      for (size_t j = 0; j < t_layout[i]; ++j)
      {
        std::cout << temp[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "Result:" << std::endl;
  for (size_t i = 0; i < result.size(); ++i)
  {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;
  //net.Save("test.txt");
  return 0;
}
