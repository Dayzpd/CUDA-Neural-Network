
#ifndef LOSS_FACTORY_H
#define LOSS_FACTORY_H

#include "LossFunction.h"

namespace neural_network
{

  class LossFunction
  {
    public:
      enum Type {
        BINARY_CROSS_ENTROPY = "BINARY_CROSS_ENTROPY",
        HINGE = "HINGE",
        MEAN_SQUARED_ERROR = "MEAN_SQUARED_ERROR",
        MULTI_CLASS_CROSS_ENTROPY = "MULTI_CLASS_CROSS_ENTROPY"
      };

      static LossFunction* create(Type loss_type);
  }

}

#endif
