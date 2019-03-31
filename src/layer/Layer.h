
#ifndef LAYER_H
#define LAYER_H

#include "../neurons/Neurons.h"

namespace neural_network
{

  class Layer
  {
    public:
      virtual ~Layer() = 0;

      virtual Neurons& forward_prop(Neurons& input) = 0;

      virtual Neurons& back_prop(Neurons& input, float learning_rate) = 0;
  }

}

#endif
