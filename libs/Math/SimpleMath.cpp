#include "SimpleMath.h"
#include <cmath>

std::vector<double> NormalizeVector(std::vector<double> input)
{
  std::vector<double> result(input);
  double len = 0;
  #pragma omp parallel for reduction(+:len)
  for (size_t i = 0; i < input.size(); ++i)
  {
    len += input[i] * input[i];
  }
  len = sqrt(len);

  #pragma omp parallel for
  for (size_t i = 0; i < result.size(); ++i)
  {
    result[i] = input[i] / len;
  }
  return result;
}

std::vector<double> DivVector(std::vector<double> input, double val)
{
  std::vector<double> result(input);
  #pragma omp parallel for
  for (size_t i = 0; i < result.size(); ++i)
  {
    result[i] = input[i] / val;
  }
  return result;
}

std::vector<double> MulVector(std::vector<double> input, double val)
{
  std::vector<double> result(input);
  #pragma omp parallel for
  for (size_t i = 0; i < result.size(); ++i)
  {
    result[i] = input[i] * val;
  }
  return result;
}