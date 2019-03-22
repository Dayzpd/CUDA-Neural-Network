
#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

#include "ActivationFunction.h"

namespace neural_network {

  class LeakyReLU : public ActivationFunction<LeakyReLU>
  {
    public:
      double calculate(double& x);

      double calculate_deriv(double& x);
  }

}

#endif
