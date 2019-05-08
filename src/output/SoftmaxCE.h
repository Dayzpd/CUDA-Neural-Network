
#ifndef SOFTMAX_CE_H
#define SOFTMAX_CE_H

#include "../neurons/Neurons.h"

#include <string>

class SoftmaxCE
{
private:
  Neurons input;
  Neurons output;
  Neurons delta;

  std::string name;
  bool verbose;

public:
  SoftmaxCE(std::string name, bool verbose = false);

  ~SoftmaxCE();

  float loss(Neurons& actual);

  Neurons& forward_prop(Neurons& input);

  Neurons& back_prop(Neurons& actual);
};

#endif
