#ifndef __CPU_NW_LEARNING_BACKPROP_H
#define __CPU_NW_LEARNING_BACKPROP_H

#include "../Networks/MLP.h"

class BackProp
{
private:
  INetwork *network;
  std::mutex network_mutex;
  ActivationFunc *activation_func = [](double x){ return x; };
  ActivationFunc *activation_func_deriv = [](double x){ return x; };
  double nu = 0.01; // learning speed
  double m = 0.01; // momentum
  std::vector<std::vector<double>> momentum_correction;
  std::string log_file_path = "";
  std::ofstream log;
  void ToLog(std::string text);
  std::vector<double> MLPItteration(std::vector<double> input, std::vector<double> expect);
public:
  BackProp();
  void SetNetwork(INetwork *net);
  void LearningSpeed(double val);
  double LearningSpeed() { return nu; }
  void Momentum(double val);
  double Momentum() { return m; }
  void CleanMomentumData() { momentum_correction.clear(); }
  void LogFile(std::string file_path);
  std::vector<double> DoItteration(std::vector<double> input, std::vector<double> expect);
  ValueTable DoBatch(ValueTable input, ValueTable expect);
  void SetActivationFunction(ActivationFunc func, ActivationFunc derivative);
  ~BackProp();
};

#endif