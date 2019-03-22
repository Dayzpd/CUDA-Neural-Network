
#ifndef NETWORK_H
#define NETWORK_H

#include "../layer/Layer.h"
#include "../loss/LossFunction.h"
#include "../optimize/OptimizeFunction.h"
#include "../neuron/Neurons.h"

namespace neural_network
{

  class Network
  {
    private:
      int input_size;
      std::vector<string> classes;
      double learning_rate;
      LossFunction* loss_function;
      OptimizeFunction* optimize_function;
      Neurons* head_layer; // First hidden layer.
      Neurons* tail_layer; // Should always have output layer strategy.

    public:
      Network(int input_size, std::vector<string> classes, double learning_rate,
        LossFunction::Type loss_strategy,
        OptimizeFunction::Type optimize_strategy);

      ~Network();

      void add_layer(Layer::Type layer_strategy,
        ActivationFunction::Type activation_strategy, int num_neurons);

      void prev_layer();

      void next_layer();
  }

}

#endif
