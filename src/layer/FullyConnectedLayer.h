
#include "Layer.h"

#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

namespace neural_network {

  class FullyConnectedLayer : public Layer
  {
    public:
      FullyConnectedLayer();

      ~FullyConnectedLayer();

      void add_neuron(HiddenNeuron neuron);

      void populate_layer();

      void connect_neurons();

      int get_num_neurons();

      void set_num_neurons(int num_neurons);

      Layer* get_prev_layer();

      void set_prev_layer(Layer* prev_layer);
  }

}

#endif
