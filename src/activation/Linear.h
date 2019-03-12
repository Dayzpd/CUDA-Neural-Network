
#include "ActivationFunction.h"

#ifndef LINEAR_H
#define LINEAR_H

namespace neural_network {

  class Linear : public ActivationFunction
  {
    public:
      ~Linear();

      double calculate(double& x);

      int calculate_deriv(double& x);
  }

}

#endif
