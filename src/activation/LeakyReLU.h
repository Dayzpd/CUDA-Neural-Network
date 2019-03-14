
#include "ActivationFunction.h"

#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

namespace neural_network {

  class LeakyReLU : public ActivationFunction
  {
    public:
      LeakyReLU();
      
      ~LeakyReLU();

      double calculate(double& x);

      double calculate_deriv(double& x);
  }

}

#endif
