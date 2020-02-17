#include "activation_functions.h"

#include <math.h>



double Tanh(double x, double a)
{
  return tanh(x*a);
}

double Derivative_Tanh(double x, double a)
{
  return (1 - (x * x)) * a;
}