
#ifndef ACTIVATION_FACTORY_H
#define ACTIVATION_FACTORY_H

#include "Layer.h"
#include "../neurons/Dim.h"

namespace neural_network
{

  class LayerFactory
  {
    private:
      LayerFactory() {}

    public:
      enum Activation {
        SOFTMAX = "SOFTMAX",
        RELU = "RELU"
      };

      enum Connection {
        FULLY_CONNECTED = "FULLY_CONNECTED",
        NORMALIZE = "NORMALIZE",
        NORMALIZE_IMG = "NORMALIZE_IMG"
      };

      static ActivationFactory& get_instance();

      Layer* create(Activation activation_type);

      Layer* create(Connection connection_type, Dim dim);
  }

}

#endif
