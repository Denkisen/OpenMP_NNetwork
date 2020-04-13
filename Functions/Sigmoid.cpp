#include "activation_functions.h"

#include <math.h>

double Sigmoid(double x, double a)
{
  return 1.0 / (1.0 + std::exp((-x * a)));
}

double Derivative_Sigmoid(double x, double a)
{
  return x * (1 - x) * a;
}