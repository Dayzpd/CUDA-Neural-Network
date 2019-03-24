
#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "Layer.h"

namespace neural_network {

  class FullyConnected : public Layer
  {
    private:
      int input_size;
      int num_neurons;

      std::vector<double> weights;
      std::vector<double> biases;

      ActivationFunction* act_func;

      Layer* prev_layer;
      Layer* next_layer;

      void build_neurons();

    public:
      FullyConnected(int num_neurons, std::string act_type, Layer& p_layer);

      FullyConnected(int num_neurons, std::string act_type, int in_size);

      const int& get_num_outputs();

      const std::vector<double>& activate();
  }

}

#endif
