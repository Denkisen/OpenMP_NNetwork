#ifndef __CPU_NW_NETWORKS_LSTM_H
#define __CPU_NW_NETWORKS_LSTM_H

#include "INetwork.h"

class LSTM
{
private:
    
public:
  LSTM();
  ~LSTM();
  std::vector<double> Pass(std::vector<double> input);
  std::vector<double> Pass(std::vector<double> input, ValueTable &temporary_layers_values, std::vector<size_t> &layout);
  Weights GetWeights(std::vector<size_t> &layout);
  bool Load(std::string file_path);
  void Save(std::string file_path, std::string comments);
};

#endif