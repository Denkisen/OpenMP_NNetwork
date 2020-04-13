#ifndef __CPU_NW_NETWORKS_INETWORK_H
#define __CPU_NW_NETWORKS_INETWORK_H

#include <iostream>
#include <fstream>
#include <vector>
#include <exception>
#include <omp.h>
#include <mutex>

typedef double ActivationFunc(double);
typedef double WeightInitFunc();
typedef double *** Weights;
typedef std::vector<std::vector<double>> ValueTable;

enum class NetworkType
{
  MLP,
  LSTM
};

class INetwork
{
protected:
  NetworkType type;
  Weights weights = nullptr;
  std::vector<size_t> weights_layout;
  std::mutex pass_forward_mutex;
  void Clean()
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
    weights_layout.clear();
    weights = nullptr;
  }
public:
  INetwork() = default;
  ~INetwork() 
  {
    Clean();
  }
  NetworkType Type() { return type; }
  virtual std::vector<double> Pass(std::vector<double> input) = 0;
  virtual std::vector<double> Pass(std::vector<double> input, ValueTable &temporary_layers_values, std::vector<size_t> &layout) = 0;
  virtual Weights GetWeights(std::vector<size_t> &layout) = 0;
  virtual bool Load(std::string file_path) = 0;
  virtual void Save(std::string file_path, std::string comments) = 0;
};


#endif