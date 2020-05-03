#include "Units.h"
#include "libs/Math/SimpleMath.h"
#include "libs/SpeedTest.h"
#include "Networks/MLP.h"
#include "Learning/BackProp.h"
#include "Functions/activation_functions.h"

#include <iostream>
#include <cmath>
#include <unistd.h>
#include <algorithm>
#include <random>

#define CONF_FILE "conf.txt"
#define LOG_FILE "log.txt"
#define COMMENTS "Test."
#define TRAIN_SET ""
#define TRAIN_SPEED 0.01
#define MOMENTUM 0.01

double ActFunc(double x)
{
  return Tanh(x, 1);
  //return Leaky_ReLu(x, 1.0);
}

double DerFunc(double x)
{
  return Derivative_Tanh(x, 1);
  //return Derivative_Leaky_ReLu(x, 1.0);
}

int cputest_main()
{
  std::vector<double> inp = {1, -1, 2, -2, 3, -3};
  inp = DivVector(inp, 3);
  std::vector<double> exp = {-1, 1, -2, 2, -3, 3};
  exp = DivVector(exp, 3);
  std::vector<double> out;

  MLP net;
  BackProp train;
  
  net.SetActivationFunc(ActFunc);
  if (!net.Load(CONF_FILE))
  {
    net.SetLayersCount(3);
    net.AddLayer(inp.size());
    net.AddLayer(4);
    net.AddLayer(exp.size());
  }
  train.LearningSpeed(TRAIN_SPEED);
  train.Momentum(MOMENTUM);
  train.LogFile(LOG_FILE);
  train.SetActivationFunction(ActFunc, DerFunc);
  train.SetNetwork(&net);

  int i = 0;
  while (i < 10000)
  {
    STOP_WATCH(start);
    out = train.DoItteration(inp, exp);
    STOP_WATCH(end);
    PRINT_DIFF_WATCH(start, end);
    out = MulVector(out, 3);
    for (auto d : out)
    {
      std::cout << d << " ";
    }
    std::cout << std::endl;
    i++;
  }
  net.Save(CONF_FILE, COMMENTS);
  return 0;
}