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
  std::lock_guard<std::mutex> lock(network_mutex);
  std::vector<double> result;
  if (input.size() == 0 || expect.size() == 0)
  {
    ToLog(std::string(__func__) + "() Error: input.size() == 0 || expect.size() == 0");
    return result;
  }

  Weights w = nullptr;
  ValueTable v = nullptr;
  std::vector<size_t> t_layout;
  std::vector<size_t> w_layout;

  if (network == nullptr)
  {
    ToLog(std::string(__func__) + "() Error: network == nullptr");
    return result;
  }
  
  try
  {
    result = network->Pass(input, v, t_layout);
    w = network->GetWeights(w_layout);
  }
  catch (std::exception e)
  {
    ToLog(std::string(__func__) + "() Error: std::vector<double> result = network->Pass(input, v, t_layout)");
    return result;
  }
  std::vector<double> correction;
  std::vector<double> correction_priv;
  for (size_t i = w_layout.size() - 1; i > 0; --i)
  {
    correction.resize(w_layout[i]);
    #pragma omp parallel for
    for (size_t j = 0; j < w_layout[i] - 1; ++j)
    {    
      if (i == w_layout.size() - 1)
      {
        #pragma omp parallel for
        for (size_t l = 0; l < w_layout[i - 1]; ++l)
        {
          correction[j] = activation_func_deriv(expect[j] - v[i][j]);
          w[i][j][l] += (correction[j] * nu * v[i - 1][l]);
        }
      }
      else
      {
        double sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t l = 0; l < w_layout[i + 1] - 1; ++l)
        {
          sum += w[i + 1][l][j] * correction_priv[l];
        }
        #pragma omp parallel for
        for (size_t l = 0; l < w_layout[i - 1]; ++l)
        {
          correction[j] = activation_func_deriv(sum);
          w[i][j][l] += (correction[j] * nu * v[i - 1][l]);
        }
      }
    }
    correction_priv = correction;
  }

  // std::cout << "Result:" << std::endl;
  // for (size_t i = 0; i < result.size(); ++i)
  // {
  //   std::cout << result[i] << " ";
  // }
  // std::cout << std::endl;

  if (v != nullptr)
  {
    for (size_t i = 0; i < t_layout.size(); ++i)
      delete[] v[i];
    delete[] v;
  }

  return result;
}

double BackProp::DoBatch(std::vector<std::vector<double>> input, std::vector<std::vector<double>> expect)
{
  return 0;
}