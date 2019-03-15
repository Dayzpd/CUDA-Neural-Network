
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "FastSigmoid.h"
#include "LeakyReLU.h"
#include "Linear.h"
#include "ReLU.h"
#include "TanH.h"

#include <memory>
#include <stdexcept>

namespace neural_network
{

  class ActivationFunction
  {
    public:
      enum Type {
        FAST_SIGMOID = "FAST_SIGMOID",
        LEAKY_RELU = "LEAKY_RELU",
        LINEAR = "LINEAR",
        RELU = "RELU",
        TANH = "TANH"
      };

      static ActivationFunction* create(Type activation_type);

      virtual double calculate(double& x) = 0;

      virtual double calculate_deriv(double& x) = 0;
  }

}

#endif
