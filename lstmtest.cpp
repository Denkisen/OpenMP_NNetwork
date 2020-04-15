#include "Units.h"
#include "Networks/LSTM.h"
#include "libs/SpeedTest.h"

#include <iostream>
#include <vector>

#define CONF_FILE "conf.txt"
#define COMMENTS "LSTM test."

int lstmtest_main()
{
  LSTM net;
  //net.Load(CONF_FILE);
  net.SetLayout(10, 6);
  std::vector<double> inp(10, 1.0);
  ValueTable tab;
  std::vector<size_t> layout;
  STOP_WATCH(start);
  std::vector<double> out = net.Pass(inp, tab, layout);
  STOP_WATCH(stop);
  PRINT_DIFF_WATCH(start, stop);
  
  for (auto o : out)
    std::cout << o << " ";
  std::cout << std::endl << std::endl;

  for (auto l : tab)
  {
    for (auto t : l)
      std::cout << t << " ";
    std::cout << std::endl;
  }
  net.Save(CONF_FILE, COMMENTS);
  return 0;
}