
#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "LossFunction.h"

namespace neural_network
{

  class CrossEntropy
  {
    public:
      float calculate(Neurons& prediction, Neurons& actual);

      float calculate_deriv(Neurons& prediction, Neurons& actual);
  }

}

#endif
