
#include "LayerFactory.h"
#include "Softmax.h"
#include "ReLU.h"
#include "FullyConnected.h"
#include "Normalize.h"
#include "NormalizeImage.h"

#include "Dim.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace neural_network
{

  ActivationFactory& ActivationFactory::get_instance()
  {
    static ActivationFactory factory_instance;
    return factory_instance;
  }

  std::unique_ptr<Layer> LayerFactory::create(std::string activation_type)
  {
    switch (activation_type)
    {
      case SOFTMAX:
        return std::make_unique<Softmax>();
      case RELU:
        return std::make_unique<ReLU>();
    }
    throw runtime_error("An invalid activation layer type was given. " +
      "Accepted types include: SIGMOID, RELU.");
  }

  std::unique_ptr<Layer> LayerFactory::create(std::string connection_type
    Dim dim
  ) {
    switch (connection_type)
    {
      case FULLY_CONNECTED:
        return std::make_unique<FullyConnected>(dim);
      case NORMALIZE:
        return std::make_unique<Normalize>(dim);
      case NORMALIZE_IMG:
        return std::make_unique<NormalizeImage>(dim);
    }
    throw runtime_error("An invalid connection layer type was given. " +
      "Accepted types include: FULLY_CONNECTED, NORMALIZE, NORMALIZE_IMG.");
  }
}
