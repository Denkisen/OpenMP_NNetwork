#include "BPTT.h"
#include "../libs/Math/SimpleMath.h"
#include <math.h>

BPTT::BPTT()
{

}

BPTT::~BPTT()
{
  if (log.is_open()) log.close();
}

void BPTT::SetNetwork(INetwork *net)
{
  std::lock_guard<std::mutex> lock(network_mutex);
  network = net;
}

void BPTT::LogFile(std::string file_path)
{
  if (log.is_open()) log.close();
  log.open(file_path, std::ios::app);
}

void BPTT::ToLog(std::string text)
{
  if (log.is_open())
    log << text << std::endl;
}

void BPTT::LearningSpeed(double val)
{
  std::lock_guard<std::mutex> lock(network_mutex);
  nu = val;
}

std::vector<ValueTable> BPTT::LSTMItteration(ValueTable input, ValueTable expect)
{
  if (input.size() < expect.size())
    throw std::runtime_error("Error: input.size() < expect.size()");
  std::lock_guard<std::mutex> lock(network_mutex);
  ValueTable result(expect.size());
  std::vector<ValueTable>  delta_x(expect.size());
  std::vector<size_t> layout; 
  ValueTable memories(result.size());
  std::vector<ValueTable> temp_values(result.size());
  size_t input_offset = input.size() - expect.size();

  for (size_t i = 0; i < input_offset; ++i)
  {
    result[i] = network->Pass(input[i]);
  }

  for (size_t i = 0; i < result.size(); ++i)
  {
    result[i] = network->Pass(input[i + input_offset], temp_values[i], layout);
    memories[i] = network->GetMemory();
  }

  Weights w = network->GetWeights(layout);

  ValueTable dW(input[0].size(), std::vector<double>(layout.size(), 0.0));
  ValueTable dU(expect[0].size(), std::vector<double>(layout.size(), 0.0));
  std::vector<double> dB(layout.size(), 0.0);

  double delta_priv = 0;
  std::vector<double> delta_priv_cell(expect[0].size(), 0.0);
  ValueTable mods(result[0].size(), std::vector<double>(layout.size(), 0.0));
  for (int i = result.size() - 1; i >= 0; --i)
  {
    for (size_t j = 0; j < result[i].size(); ++j)
    {
      if (i != (int) result.size() - 1)
      {
        for (size_t z = 0; z < layout.size(); ++z)
        {
          dU[j][z] += mods[j][z] * result[i][j];
        }
      }
      mods[j][3] = (result[i][j] - expect[i][j]) + delta_priv;
      delta_priv_cell[j] = (mods[j][3] * temp_values[i][OUTPUT_INDEX][j] * (1 - std::pow(std::tanh(memories[i][j]), 2.0)))
                      + (delta_priv_cell[j] * (i == (int) result.size() - 1 ? 0.0 : temp_values[i + 1][FORGET_INDEX][j]));
  
      mods[j][2] = delta_priv_cell[j] * temp_values[i][INPUT_INDEX][j] * Derivative_Tanh(temp_values[i][CELL_INDEX][j], 1.0);
      mods[j][1] = delta_priv_cell[j] * temp_values[i][CELL_INDEX][j] * Derivative_Sigmoid(temp_values[i][INPUT_INDEX][j], 1.0);
      mods[j][0] = delta_priv_cell[j] * (i > 0 ? memories[i - 1][j] : 0.0) * Derivative_Sigmoid(temp_values[i][FORGET_INDEX][j], 1.0);
      mods[j][3] *= std::tanh(memories[i][j]) * Derivative_Sigmoid(temp_values[i][OUTPUT_INDEX][j], 1.0);
      delta_priv = 0;
      for (size_t z = 0; z < layout.size(); ++z)
      {
        delta_priv += (w[z][j][input[0].size() + j] * mods[j][z]);
        dB[z] += mods[j][z];
        for (size_t k = 0; k < input[0].size(); ++k)
        {
          dW[k][z] += mods[j][z] * input[i + input_offset][k];
        } 
      }
    }
    
    delta_x[i].resize(input[0].size(), std::vector<double>(mods.size(), 0.0));

    for (size_t k = 0; k < delta_x[i].size(); ++k)
    {
      for (size_t j = 0; j < mods.size(); ++j)
      {
        for (size_t z = 0; z < layout.size(); ++z)
        {
          delta_x[i][k][j] += (w[z][j][k] * mods[j][z]);
        }
      }
    }

    for (auto x : delta_x[i])
    {
      for (auto d : x)
      {
        std::cout << d << " ";
      } 
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  for (size_t i = 0; i < layout.size(); ++i)
  {
    for (size_t j = 0; j < layout[i]; ++j)
    {
      for (size_t z = 0; z < input[0].size(); ++z)
      {
        w[i][j][z] -= nu * dW[z][i];
      }
      for (size_t z = 0; z < layout[i]; ++z)
      {
        w[i][j][input[0].size() + z] -= nu * dU[z][i];
      }
      w[i][j][input[0].size() + layout[i]] -= nu * dB[i];
    }
  }

  return delta_x;
}