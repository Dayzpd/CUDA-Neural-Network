
#include "Layer.h"

#ifndef LAYER_FACTORY_H
#define LAYER_FACTORY_H

namespace neural_network
{

  class LayerFactory
  {
    public:
      enum LayerType {
        INPUT, FULLY_CONNECTED, OUTPUT
      };

      virtual Layer* create(LayerType layer_type) = 0;
  }

}

#endif
