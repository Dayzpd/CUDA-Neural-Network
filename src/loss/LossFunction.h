
#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

namespace neural_network
{

  class LossFunction
  {
    public:
      virtual float calculate(Neurons& prediction, Neurons& actual) = 0;
  }

}

#endif
