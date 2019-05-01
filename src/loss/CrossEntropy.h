
#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "LossFunction.h"
#include "../neurons/Neurons.H"

namespace cuda_net
{

  class CrossEntropy : public LossFunction
  {
    public:
      float calculate(Neurons& prediction, Neurons& actual);

      Neurons calculate_deriv(Neurons& prediction, Neurons& actual,
        Neurons& prob_delta);
  };

}

#endif
