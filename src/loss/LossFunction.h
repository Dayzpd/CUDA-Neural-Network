
#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "BinaryCrossEntropy.h"
#include "Hinge.h"
#include "MeanSquaredError.h"
#include "MultiClassCrossEntropy.h"

#include <memory>
#include <stdexcept>

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

      virtual double calculate(double& x) = 0;
  }

}

#endif
