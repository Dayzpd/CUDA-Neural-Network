
#ifndef LINEAR_H
#define LINEAR_H

#include "ActivationFunction.h"

namespace neural_network {

  class Linear : public ActivationFunction
  {
    public:
      Linear();

      ~Linear();

      double calculate(double& x);

      int calculate_deriv(double& x);
  }

}

#endif
