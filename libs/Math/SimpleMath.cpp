#include "SimpleMath.h"
#include <math.h>

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

std::vector<double> MatrixMulVector(std::vector<std::vector<double>> mx, std::vector<double> v)
{
  std::vector<double> result(mx.size(), 0.0);
  for (size_t i = 0; i < result.size(); ++i)
  {
    for (size_t j = 0; j < v.size(); ++j)
    {
      result[i] += (mx[i][j] * v[j]);
    }
  }
  return result;
}

double MatrixMulVector(std::vector<double> mx, std::vector<double> v)
{
  double result = 0.0;
  for (size_t i = 0; i < mx.size(); ++i)
  {
    result += (mx[i] * v[i]);
  }
  return result;
}

std::vector<std::vector<double>> MatrixMulVector(std::vector<std::vector<double>> mx, std::vector<std::vector<double>> v)
{
  std::vector<std::vector<double>> result(mx.size(), std::vector<double>(v.size(), 0.0));
  for (size_t i = 0; i < mx.size(); ++i)
  {
    for (size_t j = 0; j < v.size(); ++j)
    {
      for (size_t z = 0; z < mx[i].size(); ++z)
      {
        result[i][j] += (mx[i][z] * v[j][z]);
      }
    }
  }
  return result;
}

std::vector<std::vector<double>> OuterMulVector(std::vector<double> f, std::vector<double> s)
{
  std::vector<std::vector<double>> result(f.size(), std::vector<double>(s.size(), 0.0));

  for (size_t i = 0; i < f.size(); ++i)
    for (size_t j = 0; j < s.size(); ++j)
      result[i][j] = f[i] * s[j];

  return result;
}