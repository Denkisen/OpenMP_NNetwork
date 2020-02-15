#ifndef __CPU_NW_NETWORKS_INETWORK_H
#define __CPU_NW_NETWORKS_INETWORK_H

#include <iostream>
#include <vector>
#include <exception>
#include <omp.h>
#include <mutex>

typedef double ActivationFunc(double);
typedef double WeightInitFunc();
typedef double *** Weights;
typedef double ** ValueTable;

class INetwork
{
protected:
  Weights weights = nullptr;
  std::vector<size_t> weights_layout;
  std::mutex pass_forward_mutex;
public:
  INetwork() = default;
  ~INetwork() 
  {
    if (weights != nullptr)
    {
      for (size_t i = 0; i < weights_layout.size(); ++i)
      {
        for (size_t j = 0; j < weights_layout[i]; ++j)
          delete[] weights[i][j];
        delete[] weights[i];
      }
      delete[] weights;
    }
  }
  virtual std::vector<double> Pass(std::vector<double> input) = 0;
  virtual std::vector<double> Pass(std::vector<double> input, ValueTable &temporary_layers_values, std::vector<size_t> &layout) = 0;
  virtual void AddLayer(size_t size) = 0;
  virtual Weights GetWeights(std::vector<size_t> &layout) = 0;
  virtual void Load(std::string file_path) = 0;
  virtual void Save(std::string file_path) = 0;
};


#endif