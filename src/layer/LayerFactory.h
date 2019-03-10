
#include "Layer.h"

#ifndef LAYER_FACTORY_H
#define LAYER_FACTORY_H

namespace neural_network
{

  class LayerFactory
  {
    public:
      virtual Layer* create(const str layer_type) = 0;
  }

}

#endif
