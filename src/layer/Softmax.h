
#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "Layer.h"
#include "../neurons/Neurons.h"

namespace neural_network {

  class Softmax : public Layer
  {
    private:
      Neurons input;
      Neurons output;

    public:
      Softmax();

      ~Softmax();

      Neurons& forward_prop(Neurons& input);

      Neurons& back_prop(Neurons& input, float learning_rate);
  }

}

#endif
