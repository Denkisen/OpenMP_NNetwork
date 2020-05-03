#include "BackProp.h"

BackProp::BackProp()
{
}

BackProp::~BackProp()
{
  if (log.is_open()) log.close();
}

void BackProp::SetNetwork(INetwork *net)
{
  std::lock_guard<std::mutex> lock(network_mutex);
  network = net;
}

void BackProp::SetActivationFunction(ActivationFunc func, ActivationFunc derivative)
{
  std::lock_guard<std::mutex> lock(network_mutex);
  activation_func = func;
  activation_func_deriv = derivative;
}

void BackProp::LearningSpeed(double val)
{
  std::lock_guard<std::mutex> lock(network_mutex);
  nu = val;
}

void BackProp::Momentum(double val)
{
  std::lock_guard<std::mutex> lock(network_mutex);
  m = val;
}

void BackProp::LogFile(std::string file_path)
{
  if (log.is_open()) log.close();
  log.open(file_path, std::ios::app);
}

void BackProp::ToLog(std::string text)
{
  if (log.is_open())
    log << text << std::endl;
}

std::vector<double> BackProp::DoItteration(std::vector<double> input, std::vector<double> expect)
{
  std::vector<double> result;
  switch (network->Type())
  {
  case NetworkType::MLP:
    result = MLPItteration(input, expect);
    break;
  default:
    break;
  }
  return result;
}

std::vector<double> BackProp::MLPItteration(std::vector<double> input, std::vector<double> expect)
{
  std::lock_guard<std::mutex> lock(network_mutex);
  std::vector<double> result;
  Weights w = nullptr;
  ValueTable v;
  std::vector<size_t> w_layout;  

  if (input.size() == 0 || expect.size() == 0)
  {
    ToLog(std::string(__func__) + "() Error: input.size() == 0 || expect.size() == 0");
    return result;
  }

  if (network == nullptr)
  {
    ToLog(std::string(__func__) + "() Error: network == nullptr");
    return result;
  }
  
  try
  {
    result = network->Pass(input, v, w_layout);
    w = network->GetWeights(w_layout);
  }
  catch (...)
  {
    ToLog(std::string(__func__) + "() Error: std::vector<double> result = network->Pass(input, v, t_layout)");
    return result;
  }
  std::vector<double> correction;
  std::vector<double> correction_priv;
  momentum_correction.resize(w_layout.size());
  for (size_t i = w_layout.size() - 1; i > 0; --i)
  {
    correction.resize(w_layout[i] - 1);
    correction[correction.size() - 1] = 0;
    momentum_correction[i].resize(w_layout[i]);

    #pragma omp parallel for
    for (size_t j = 0; j < w_layout[i] - 1; ++j)
    {    
      if (i == w_layout.size() - 1)
      {
        correction[j] = ((expect[j] - v[i][j]) * activation_func_deriv(v[i][j]));
        momentum_correction[i][j] = correction[j];
        for (size_t l = 0; l < w_layout[i - 1]; ++l)
        {
          w[i][j][l] += (((1.0 - m) * (correction[j] * nu * v[i - 1][l])) + (m * momentum_correction[i][j]));
        }
      }
      else
      {
        double sum = 0;
        for (size_t l = 0; l < w_layout[i + 1] - 1; ++l)
        {
          sum += w[i + 1][l][j] * correction_priv[l];
        }
        correction[j] = activation_func_deriv(v[i][j]) * sum;
        momentum_correction[i][j] = correction[j];
        for (size_t l = 0; l < w_layout[i - 1]; ++l)
        {
          w[i][j][l] += (((1.0 - m) * (correction[j] * nu * v[i - 1][l])) + (m * momentum_correction[i][j]));
        }
      }
    }
    correction_priv = correction;
  }

  return result;
}