

#ifndef NETWORK_H
#define NETWORK_H

#include "../neuron/Neurons.h"

namespace neural_network
{

  class Network
  {
    private:
      double learning_rate;
      Neurons* head_layer; // First hidden layer.
      Neurons* tail_layer; // Should always have output layer strategy.

    public:
      Network(double learning_rate, LossFunction::Type loss_function);

      ~Network();

      void add_layer();

      Neurons* get_prev_layer();

      Neurons* get_next_layer();
  }

}

#endif
