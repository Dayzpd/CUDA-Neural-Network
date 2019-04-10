
#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "Layer.h"
#include "../neurons/Dim.h"
#include "../neurons/Neurons.h"

namespace neural_network {

  class FullyConnected : public Layer
  {
    private:
      Neurons input;
      Neurons weights;
      Neurons biases;
      Neurons output;
      Neurons backprop_deriv;

    public:
      FullyConnected(Dim dim);

      ~FullyConnected();

      Neurons& forward_prop(Neurons& input);

      Neurons& back_prop(Neurons& input, float learning_rate);

      void init_weights();

      void init_biases();
  }

}

#endif
