
#ifndef RELU_H
#define RELU_H

#include "Layer.h"
#include "../neurons/Neurons.h"

#include <string>

class ReLU : public Layer
{
private:
  Neurons input;
  Neurons output;
  Neurons delta;

  std::string name;

public:
  ReLU(std::string name);

  ~ReLU();

  Neurons& forward_prop(Neurons& input);

  Neurons& back_prop(Neurons& input, float learning_rate);
};


#endif
