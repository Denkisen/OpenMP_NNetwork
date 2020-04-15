#ifndef __CPU_NW_NETWORKS_LSTM_H
#define __CPU_NW_NETWORKS_LSTM_H

#include "INetwork.h"
#include "../Functions/activation_functions.h"

#include <cstdlib>
#include <ctime>

#define FORGET_INDEX 0
#define INPUT_INDEX 1
#define CELL_INDEX 2
#define OUTPUT_INDEX 3

/* 
  weights[f,i,c,o][output_size][]
  f[output_size][input_size + output_size + b] - forget
  i[output_size][input_size + output_size + b] - input
  c[output_size][input_size + output_size + b] - cell
  o[output_size][input_size + output_size + b] - output
  output_size = cell_size
*/

class LSTM : public INetwork
{
private:
  const std::string version = "LSTM:1.1";
  std::vector<double> mem_cell;
  std::vector<double> last_output;
  size_t input_size = 0;
  WeightInitFunc *weight_init_func = []() { return ((double) std::rand() / RAND_MAX) - 0.5; };
  ActivationFunc *activation_func = [](double x) { return Sigmoid(x, 1.0); };
public:
  LSTM();
  ~LSTM();
  std::vector<double> Pass(std::vector<double> input);
  std::vector<double> Pass(std::vector<double> input, ValueTable &temporary_layers_values, std::vector<size_t> &layout);
  Weights GetWeights(std::vector<size_t> &layout);
  bool Load(std::string file_path);
  void Save(std::string file_path, std::string comments);
  void SetLayout(size_t input_size, size_t output_size);
};

#endif