
#ifndef RELU_H
#define RELU_H

#include "Layer.h"
#include "../neurons/Neurons.h"

namespace neural_network {

  class ReLU : public Layer
  {
    private:
      Neurons input;
      Neurons output;
      Neurons backprop_deriv;

    public:
      ReLU();

      ~ReLU();

      Neurons& forward_prop(Neurons& input);

      Neurons& back_prop(Neurons& input, float learning_rate);
  }

}

#endif
