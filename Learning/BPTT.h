#ifndef __CPU_NW_LEARNING_BPTT_H
#define __CPU_NW_LEARNING_BPTT_H

#include "../Networks/LSTM.h"

class BPTT
{
private:
  INetwork *network;
  std::mutex network_mutex;
  double nu = 0.01; // learning speed
  std::string log_file_path = "";
  std::ofstream log;
  void ToLog(std::string text);
public:
  BPTT();
  ~BPTT();
  void SetNetwork(INetwork *net);
  void LearningSpeed(double val);
  double LearningSpeed() { return nu; }
  void LogFile(std::string file_path);
  std::vector<ValueTable> LSTMItteration(ValueTable input, ValueTable expect);
};

#endif