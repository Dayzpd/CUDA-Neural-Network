
#ifndef OPTIMIZE_FUNCTION_H
#define OPTIMIZE_FUNCTION_H

#include "GradientDescent.h"

#include <memory>
#include <stdexcept>

namespace neural_network
{

  class LossFunction
  {
    public:
      enum Type {
        GRADIENT_DESCENT = "GRADIENT_DESCENT"
      };

      static OptimizeFunction* create(Type optimize_type);

      virtual double calculate(double& x) = 0;
  }

}

#endif
