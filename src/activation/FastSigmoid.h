
#ifndef FAST_SIGMOID_H
#define FAST_SIGMOID_H

#include "ActivationFunction.h"

namespace neural_network {

  class FastSigmoid : public ActivationFunction
  {
    public:
      FastSigmoid();

      ~FastSigmoid();

      double calculate(double& x);

      double calculate_deriv(double& x);
  }

}

#endif
