
#ifndef LOSS_FACTORY_H
#define LOSS_FACTORY_H

#include "LossFunction.h"

namespace neural_network
{

  class LossFactory
  {
    private:
      LossFactory() {}

    public:
      enum Type {
        CROSS_ENTROPY = "CROSS_ENTROPY"
      };

      static LossFactory& get_instance();

      static LossFunction& create(Type loss_type);
  }

}

#endif
