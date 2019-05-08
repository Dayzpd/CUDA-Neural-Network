
#ifndef LAYER_H
#define LAYER_H

#include "../neurons/Neurons.h"


class Layer
{
public:
  virtual Neurons& forward_prop(Neurons& input) = 0;

  virtual Neurons& back_prop(Neurons& input, float learning_rate) = 0;
};


#endif
