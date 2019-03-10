
#include "ActivationFunction.h"

#ifndef RELU_H
#define RELU_H

namespace neural_network {

  class ReLU : public ActivationFunction
  {
    public:
      ~ReLU();

      double calculate(double& x);
  }

}

#endif
