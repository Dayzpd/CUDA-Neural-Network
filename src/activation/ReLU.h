
#ifndef RELU_H
#define RELU_H

#include "ActivationFunction.h"

namespace neural_network {

  class ReLU : public ActivationFunction
  {
    public:
      ReLU();

      ~ReLU();

      double calculate(double& x);

      int calculate_deriv(double& x);
  }

}

#endif
