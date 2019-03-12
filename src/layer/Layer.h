
#ifndef LAYER_H
#define LAYER_H

#include "ActivationFunction.h"
#include "Neuron.h"

namespace neural_network {

  class Layer
  {
    public:
      virtual ~Layer();

      virtual void add_neuron(Neuron* neuron) = 0;

      virtual void populate_layer() = 0;

      virtual void connect_neurons();

      virtual int get_activation_function();

      virtual void set_activation_function(ActivationFunction* af);

      virtual int get_input_size();

      virtual void set_input_size(int input_size) = 0;

      virtual int get_num_neurons();

      virtual void set_num_neurons(int num_neurons) = 0;

      virtual Layer* get_prev_layer();

      virtual void set_prev_layer(Layer* prev_layer);
  }

}

#endif
