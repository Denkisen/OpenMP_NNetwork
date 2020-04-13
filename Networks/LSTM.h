#ifndef __CPU_NW_NETWORKS_LSTM_H
#define __CPU_NW_NETWORKS_LSTM_H

#include "INetwork.h"

#include <cstdlib>
#include <ctime>

class LSTM : public INetwork
{
private:
  const std::string version = "LSTM:1.0";
  std::vector<double> mem_cell;
  std::vector<double> last_output;
  WeightInitFunc *weight_init_func = []() { return ((double) std::rand() / RAND_MAX) - 0.5; };
public:
  LSTM();
  ~LSTM();
  std::vector<double> Pass(std::vector<double> input);
  std::vector<double> Pass(std::vector<double> input, ValueTable &temporary_layers_values, std::vector<size_t> &layout);
  Weights GetWeights(std::vector<size_t> &layout);
  bool Load(std::string file_path);
  void Save(std::string file_path, std::string comments);
  void SetLayout(size_t size);
};

#endif