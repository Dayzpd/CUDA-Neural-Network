
#ifndef OPTIMIZE_FACTORY_H
#define OPTIMIZE_FACTORY_H

#include "GradientFunction.h"

namespace neural_network
{

  class LossFactory
  {
    public:
      enum Type {
        GRAD_DESC = "GRAD_DESC"
      };

      static OptimizeFunction* create(Type optimize_type);
  }

}

#endif
