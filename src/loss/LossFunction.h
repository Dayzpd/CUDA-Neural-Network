
#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "../neurons/Neurons.h"

namespace cuda_net
{

  class LossFunction
  {
    public:
      virtual float calculate(Neurons& prob, Neurons& actual) = 0;

      virtual Neurons calculate_deriv(Neurons& prob, Neurons& actual,
        Neurons& prob_delta) = 0;
  };

}

#endif
