#ifndef __CPU_NW_LEARNING_BACKPROP_H
#define __CPU_NW_LEARNING_BACKPROP_H

#include "../Networks/MultiLayerPerceptron.h"

class BackProp
{
private:
  INetwork *network;
public:
  BackProp(/* args */);
  void SetNetwork(INetwork &net);
  ~BackProp();
};

#endif