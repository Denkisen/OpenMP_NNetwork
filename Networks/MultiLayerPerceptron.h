#ifndef __CPU_NW_NETWORKS_MUlTILAYERPERCEPTRON_H
#define __CPU_NW_NETWORKS_MUlTILAYERPERCEPTRON_H

#include "INetwork.h"
#include <cstdlib>
#include <ctime>

class MultiLayerPerceptron : public INetwork
{
private:
  bool bias = true;
  ActivationFunc *activation_func = [](double x){ return x < 0 ? x * 0.01 : x; };
  WeightInitFunc *weight_init_func = [](){ return double(std::rand()) / (double(RAND_MAX) + 1.0); };
public:
  MultiLayerPerceptron();
  ~MultiLayerPerceptron();

  std::vector<double> Pass(std::vector<double> input);
  std::vector<double> Pass(std::vector<double> input, ValueTable &temporary_layers_values, std::vector<size_t> &layout);
  void AddLayer(size_t size);
  Weights GetWeights(std::vector<size_t> &layout);
  void Load(std::string file_path);
  void Save(std::string file_path);

  bool Bias();
  void Bias(bool state);
  void SetActivationFunc(ActivationFunc func);
  void SetWeightInitFunc(WeightInitFunc func);
  
};


#endif