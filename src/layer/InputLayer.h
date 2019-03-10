
#include "Layer.h"
#include "InputNeuron.h"

#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

namespace neural_network {

  class InputLayer : public Layer
  {
    public:
      InputLayer();

      ~InputLayer();

      void add_neuron(InputNeuron neuron, BiasNeuron bias_neuron);

      void populate_layer();

      int get_num_neurons();

      void set_num_neurons(int num_neurons);
  }

}

#endif
