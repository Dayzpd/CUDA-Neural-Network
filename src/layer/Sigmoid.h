
#ifndef SIGMOID_H
#define SIGMOID_H

#include "Layer.h"
#include "../neurons/Neurons.h"

namespace neural_network {

  class Sigmoid : public Layer
  {
    private:
      Neurons input;
      Neurons output;

    public:
      Sigmoid();

      ~Sigmoid();

      Neurons& forward_prop(Neurons& input);

      Neurons& back_prop(Neurons& input, float learning_rate);
  }

}

#endif
