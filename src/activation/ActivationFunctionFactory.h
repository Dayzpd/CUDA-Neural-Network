
#include "Layer.h"

#ifndef ACTIVATION_FUNCTION_FACTORY_H
#define ACTIVATION_FUNCTION_FACTORY_H

namespace neural_network
{

  class ActivationFunctionFactory
  {
    public:
      enum ActivationType {
        FAST_SIGMOID, LEAKY_RELU, LINEAR, RELU, TANH
      };

      virtual ActivationFunction* create(ActivationType activation_type) = 0;
  }

}

#endif
