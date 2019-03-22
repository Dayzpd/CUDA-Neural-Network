
#ifndef TANH_H
#define TANH_H

#include "ActivationFunction.h"

namespace neural_network {

  class TanH : public ActivationFunction<TanH>
  {
    public:
      double calculate(double& x);

      double calculate_deriv(double& x)
  }

}

#endif
