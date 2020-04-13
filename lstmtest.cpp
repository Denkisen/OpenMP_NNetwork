#include "Units.h"
#include "Networks/LSTM.h"

#include <iostream>
#include <vector>

#define CONF_FILE "conf.txt"
#define COMMENTS "LSTM test."

int lstmtest_main()
{
  LSTM net;
  //net.SetLayout(5);
  net.Load(CONF_FILE);
  return 0;
  std::vector<double> inp = { 1, 1, 1, 1, 1};
  ValueTable tab;
  std::vector<size_t> layout;

  std::vector<double> out = net.Pass(inp, tab, layout);
  
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