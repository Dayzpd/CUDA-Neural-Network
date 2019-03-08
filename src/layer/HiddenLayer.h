
#include "Layer.h"

using namespace std;

#ifndef HIDDEN_LAYER_H
#define HIDDEN_LAYER_H

namespace neural_network {

  class HiddenLayer : public Layer
  {
    private:
      int num_hidden_neurons;
      vector<HiddenNeuron> neurons;

    public:
      const string LAYER_TYPE = Layer::ALLOWED_LAYER_TYPES[1];
  }

}

#endif
