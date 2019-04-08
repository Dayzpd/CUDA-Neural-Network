
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
      Neurons input_deriv;

    public:
      FullyConnected(Dim dim, std::string act_type);

      ~FullyConnected();

      Neurons& forward_prop(Neurons& input);

      Neurons& back_prop(Neurons& input, float learning_rate);

      void initialize_neurons();

      void init_weights();

      void init_biases();
  }

}

#endif
