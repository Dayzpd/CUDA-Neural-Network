
#include "ActivationFunction.h"

#ifndef TANH_H
#define TANH_H

namespace neural_network {

  class TanH : public ActivationFunction
  {
    public:
      ~TanH();

      double calculate(double& x);

      double calculate_deriv(double& x)
  }

}

#endif
