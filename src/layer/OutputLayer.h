
#include "Layer.h"

using namespace std;

#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H

namespace neural_network {

  class OutputLayer : public Layer
  {
    private:
      int num_output_classes;
      vector<OutputNeuron> neurons;

    public:
      OutputLayer();

      ~OutputLayer();

      void add_neuron(OutputNeuron neuron);

      void populate_layer();

      void connect_neurons();

      int get_num_neurons();

      void set_num_neurons(int num_classes);

      Layer* get_prev_layer();

      void set_prev_layer(Layer* prev_layer);
  }

}

#endif
