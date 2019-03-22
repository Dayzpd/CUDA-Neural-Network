
#ifndef FAST_SIGMOID_H
#define FAST_SIGMOID_H

#include "ActivationFunction.h"

namespace neural_network {

  class FastSigmoid : public ActivationFunction<FastSigmoid>
  {
    public:
      double calculate(double& x);

      double calculate_deriv(double& x);
  }

}

#endif
