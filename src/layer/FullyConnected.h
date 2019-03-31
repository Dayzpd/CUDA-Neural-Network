
#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "Layer.h"
#include "../neurons/Dim.h"
#include "../neurons/Neurons.h"

namespace neural_network {

  class FullyConnected : public Layer
  {
    private:
      Neurons weights;
      Neurons biases;
      Neurons layer_ouput;
      Neurons optimized;

    public:
      FullyConnected(Dim dim, std::string act_type);

      ~FullyConnected();

      Neurons& forward_prop(Neurons& input);

      Neurons& back_prop(Neurons& input, float learning_rate);

      void initialize_neurons();

      void init_weights();

      void init_biases();

      void init_layer_ouput();

      void init_optimized();
  }

}

#endif
