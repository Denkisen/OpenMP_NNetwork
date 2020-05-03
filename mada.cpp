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
#define COMMENTS "MLP."
#define TRAIN_SET "data.txt"
#define TRAIN_SPEED 0.01
#define MOMENTUM 0.01

double ActFunc(double x)
{
  return Tanh(x, 1.0);
  //return Leaky_ReLu(x, 1.0);
}

double DerFunc(double x)
{
  return Derivative_Tanh(x, 1.0);
  //return Derivative_Leaky_ReLu(x, 1.0);
}

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

int mada_main()
{
  MLP net;
  BackProp train;
  net.SetActivationFunc(ActFunc);
  if (!net.Load(CONF_FILE))
  {
    net.SetLayersCount(3);
    net.AddLayer(10);
    net.AddLayer(15);
    net.AddLayer(1);
  }

  train.SetNetwork(&net);
  train.LogFile(LOG_FILE);
  train.SetActivationFunction(ActFunc, DerFunc);
  train.LearningSpeed(TRAIN_SPEED);
  train.Momentum(MOMENTUM);

  std::vector<std::vector<double>> max_min;
  max_min.push_back({50.0, 180.0, 1500.0, 200.0, 1.7, 48230.0, 121940.0, 26840.0, 750.0, 3700.0});
  max_min.push_back({0.5, 1.8, 15.0, 2.0, 0.017, 482.3, 1219.4, 268.4, 45.0, 150.0});
  std::vector<std::pair<double, std::vector<double>>> data;
  std::vector<double> output(1);
  while (true)
  {
    std::ifstream inp(TRAIN_SET);
    if (!inp.fail())
    {
      std::string line = "";
      while ((inp >> line) && !inp.eof())
      {
        std::vector<std::string> spt = split(line, ";");
        std::vector<double> add(spt.size() - 1);
        std::transform(spt.begin(), spt.end() - 1, add.begin(), [](auto x){ return std::stod(x); });
        #pragma omp parallel for
        for (size_t i = 0; i < add.size(); ++i)
        {
          add[i] = (add[i] - max_min[1][i]) / (max_min[0][i] - max_min[1][i]);
        }
        data.push_back(std::make_pair(std::stod(spt[spt.size() - 1]), add));
      }
      inp.close();

      size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
      shuffle (data.begin(), data.end(), std::default_random_engine(seed));

      bool zero = true;
      for (size_t i = 0; i < data.size(); ++i)
      { 
        std::vector<double> ex(1);
        ex[0] = data[i].first;
        output = train.DoItteration(data[i].second, ex);
        output[0] = roundf(output[0] * 100) / 100;
        std::cout << "Out: " << output[0] << " Exp: " << data[i].first << " Acc: " << (data[i].first - output[0]) << std::endl; 
        if (std::abs(data[i].first - output[0]) >= 0.02)
          zero = false;
        
      }
      if (zero) break;
      net.Save(CONF_FILE, COMMENTS);
    }
    else
    {
      break;
    }
  }
  return 0;
}