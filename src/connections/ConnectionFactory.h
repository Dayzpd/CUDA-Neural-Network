
#ifndef LAYER_FACTORY_H
#define LAYER_FACTORY_H

#include "Layer.h"

namespace neural_network
{

  class LayerFactory
  {
    public:
      enum Type {
        AVG_POOL = "AVG_POOL",
        CONVOLUTIONAL = "CONVOLUTIONAL",
        FULLY_CONNECTED = "FULLY_CONNECTED",
        MAX_POOL = "MAX_POOL",
        OUTPUT = "OUTPUT"
      };

      static Layer* create(Type layer_type);
  }

}

#endif
