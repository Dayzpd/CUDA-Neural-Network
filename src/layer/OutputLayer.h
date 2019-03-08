
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
      const string LAYER_TYPE = Layer::ALLOWED_LAYER_TYPES[2];
  }

}

#endif
