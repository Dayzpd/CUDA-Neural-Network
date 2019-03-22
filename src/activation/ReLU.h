
#ifndef RELU_H
#define RELU_H

#include "ActivationFunction.h"

namespace neural_network {

  class ReLU : public ActivationFunction<ReLU>
  {
    public:
      double calculate(double& x);

      double calculate_deriv(double& x);
  }

}

#endif
