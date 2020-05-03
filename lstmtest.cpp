#include "Units.h"
#include "Networks/LSTM.h"
#include "libs/SpeedTest.h"
#include "Learning/BPTT.h"

#include <iostream>
#include <vector>

#define CONF_FILE "conf.txt"
#define COMMENTS "LSTM test."
#define TRAIN_SPEED 0.1
#define MOMENTUM 0.01

int lstmtest_main()
{
  LSTM net;
  BPTT train;
  //net.SetLayout(2, 1);
  net.Load(CONF_FILE);
  ValueTable input(2);
  ValueTable exp(2);
  input[0] = { 1.0, 2.0 };
  input[1] = { 0.5, 3.0 };

  exp[0] = { 0.5 };
  exp[1] = { 1.25 };

  train.LearningSpeed(TRAIN_SPEED);
  train.LogFile("log.txt");
  train.SetNetwork(&net);
  STOP_WATCH(start);
  std::vector<ValueTable> res = train.LSTMItteration(input, exp);
  STOP_WATCH(stop);
  PRINT_DIFF_WATCH(start, stop);
  //net.Save(CONF_FILE, COMMENTS);
  return 0;
}