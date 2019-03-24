
#ifndef LAYER_H
#define LAYER_H

#include "../activation/ActivationFunction.h"

#include <string>
#include <vector>

namespace neural_network
{

  class Layer
  {
    public:
      double rand_norm();

      const int& get_num_neurons();

      void set_prev_layer(Layer* p_layer);

      void set_next_layer(Layer* n_layer);

      void set_activation_function(std::string act_type);

      virtual int calculate_num_outputs();

      virtual const int& get_num_outputs() = 0;

      virtual const std::vector<double>& activate() = 0;
  }

}

#endif
