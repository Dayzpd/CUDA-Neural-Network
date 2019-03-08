
#include "Layer.h"

using namespace std;

#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

namespace neural_network {

  class InputLayer : public Layer
  {
    private:
      int num_input_neurons;
      vector<InputNeuron> neurons;

    public:
      Layer(int num_neurons)
      {
        this->num_input_neurons = num_neurons;
        vector<InputNeuron> neurons(this->num_input_neurons);
      }

      void add_neuron(InputNeuron neuron)
      {
        this->neurons.push_back(neuron);
      }

      void populate_layer()
      {
        for (int x = 0; x < this->num_input_neurons; x++)
        {
          InputNeuron neuron;
          this->add_neuron(neuron);
        }
      }
  }

}

#endif
