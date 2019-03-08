
#include "Layer.h"

using namespace std;

#ifndef LAYER_FACTORY_H
#define LAYER_FACTORY_H

namespace neural_network
{

  class LayerFactory
  {
    public:
      virtual Layer* create(const str layer_type, int num_neurons,
        Layer* prev_layer = 0) = 0;

      virtual bool is_accepted_type(const str layer_type) = 0;

      virtual string accepted_types_to_string() = 0;
  }

}

#endif
