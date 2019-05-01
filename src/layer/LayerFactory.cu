
#include "LayerFactory.h"
#include "Softmax.h"
#include "ReLU.h"
#include "FullyConnected.h"

#include "../neurons/Dim.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace cuda_net
{

  const std::string SOFTMAX = "SOFTMAX";
  const std::string RELU = "RELU";
  const std::string FULLY_CONNECTED = "FULLY_CONNECTED";

  LayerFactory& LayerFactory::get_instance()
  {
    static LayerFactory factory_instance;
    return factory_instance;
  }

  std::unique_ptr<Layer> LayerFactory::create(std::string activation_type)
  {

    if (activation_type == SOFTMAX)
    {
      return std::make_unique<Softmax>();
    }

    if (activation_type == RELU)
    {
      return std::make_unique<ReLU>();
    }

    throw std::runtime_error("An invalid activation layer type was given. \
      Accepted types include: SOFTMAX, RELU.");
  }

  std::unique_ptr<Layer> LayerFactory::create(std::string connection_type,
    Dim dim
  ) {

    if (connection_type == FULLY_CONNECTED)
    {
      return std::make_unique<FullyConnected>(dim);
    }

    throw std::runtime_error("An invalid connection layer type was given. \
      Accepted types include: FULLY_CONNECTED.");
  }
}
