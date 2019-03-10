
#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"

namespace neural_network {

  class Layer
  {
    public:
      virtual ~Layer();

      virtual void add_neuron(Neuron* neuron) = 0;

      virtual void populate_layer() = 0;

      virtual void connect_neurons();

      virtual int get_num_neurons();

      virtual void set_num_neurons(int num_neurons) = 0;

      virtual void set_prev_layer(Layer* prev_layer);
  }

}

#endif
