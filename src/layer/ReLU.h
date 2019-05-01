
#ifndef RELU_H
#define RELU_H

#include "Layer.h"
#include "../neurons/Neurons.h"

namespace cuda_net
{

  class ReLU : public Layer
  {
    private:
      Neurons input;
      Neurons output;
      Neurons delta;

    public:
      ReLU();

      ~ReLU();

      Neurons& forward_prop(Neurons& input);

      Neurons& back_prop(Neurons& input, float learning_rate);
  };

}

#endif
