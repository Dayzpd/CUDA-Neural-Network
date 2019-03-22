
#ifndef LINEAR_H
#define LINEAR_H

#include "ActivationFunction.h"

namespace neural_network {

  class Linear : public ActivationFunction<Linear>
  {
    public:
      double calculate(double& x);

      double calculate_deriv(double& x);
  }

}

#endif
