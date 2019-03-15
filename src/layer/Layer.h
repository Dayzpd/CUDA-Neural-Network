
#ifndef LAYER_H
#define LAYER_H

#include "Convolutional.h"
#include "FullyConnected.h"
#include "Output.h"
#include "Pooling.h"

#include <memory>
#include <stdexcept>

namespace neural_network {

  class Layer
  {
    public:
      enum Type {
        CONVOLUTIONAL = "CONVOLUTIONAL",
        FULLY_CONNECTED = "FULLY_CONNECTED",
        OUTPUT = "OUTPUT",
        POOLING = "POOLING"
      };

      static Layer* create(Type layer_type);
  }

}

#endif
