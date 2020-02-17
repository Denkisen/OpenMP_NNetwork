#include "activation_functions.h"



double ReLu(double x)
{
  return x <= 0 ? 0 : x;
}

double Derivative_ReLu(double x)
{
  return x >= 0 ? x : 0;
}

double Leaky_ReLu(double x, double a)
{
  return x < 0 ? x * a : x;
}

double Derivative_Leaky_ReLu(double x, double a)
{
  return x >= 0 ? x : a * x;
}