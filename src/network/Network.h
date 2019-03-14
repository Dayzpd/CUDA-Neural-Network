
#include "../helpers/DoublyLinkedList.h"
#include "../layer/Layer.h"

#ifndef NETWORK_H
#define NETWORK_H

namespace neural_network
{

  class Network : DoublyLinkedList
  {
    private:
      int num_layers;

    public:
      enum LayerType {
        INPUT, FULLY_CONNECTED, OUTPUT
      };

      virtual Layer* add_layer(LayerType layer_type) = 0;
  }

}

#endif
