
#include "Neuron.h"

using namespace std;

#ifndef LAYER_H
#define LAYER_H

namespace neural_network {

  class Layer
  {
    public:
      enum {
        INPUT = "INPUT",
        HIDDEN = "HIDDEN",
        OUTPUT = "OUTPUT"
      };

      Layer();

      ~Layer();

      virtual void add_neuron(Neuron* neuron) = 0;

      virtual void populate_layer() = 0;

      virtual void populate_layer() = 0;

      virtual int get_num_neurons();

      virtual void set_num_neurons(int num_neurons);

      virtual Layer* get_prev_layer();

      virtual void set_prev_layer(Layer* prev_layer);
  }

}

#endif
