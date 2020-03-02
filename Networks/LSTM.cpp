#include "LSTM.h"

LSTM::LSTM()
{
  
}

LSTM::~LSTM()
{

}

std::vector<double> LSTM::Pass(std::vector<double> input)
{
  return std::vector<double>();
}

std::vector<double> LSTM::Pass(std::vector<double> input, ValueTable &temporary_layers_values, std::vector<size_t> &layout)
{
  return std::vector<double>();
}

Weights LSTM::GetWeights(std::vector<size_t> &layout)
{
  return nullptr;
}

bool LSTM::Load(std::string file_path)
{
  return false;
}

void LSTM::Save(std::string file_path, std::string comments)
{

}