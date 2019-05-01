
#ifndef LAYER_FACTORY_H
#define LAYER_FACTORY_H

#include "Layer.h"
#include "../neurons/Dim.h"

#include <string>

namespace cuda_net
{

  class LayerFactory
  {
    private:
      LayerFactory() {}

    public:
        static const std::string SOFTMAX;
        static const std::string RELU;
        static const std::string FULLY_CONNECTED;


      static LayerFactory& get_instance();

      static std::unique_ptr<Layer> create(std::string activation_type);

      static std::unique_ptr<Layer> create(std::string connection_type, Dim dim);
  };

}

#endif
