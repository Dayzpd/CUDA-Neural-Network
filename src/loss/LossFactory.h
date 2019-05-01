
#ifndef LOSS_FACTORY_H
#define LOSS_FACTORY_H

#include "LossFunction.h"

#include <string>

namespace cuda_net
{

  class LossFactory
  {
    private:
      LossFactory() {}

    public:
      static const std::string CROSS_ENTROPY;

      static LossFactory& get_instance();

      static LossFunction* create(std::string loss_type);
  };

}

#endif
