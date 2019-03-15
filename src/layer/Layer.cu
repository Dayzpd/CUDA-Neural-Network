
#include "Layer.h"

namespace neural_network
{
  /// <summary><c>create</c> serves as a factory function to establish
  /// strategies for connecting Neurons in a given layer to Neurons in a
  /// previous Layer.</summary>
  /// <param name="layer_type">The <c>LayerFactory</c> class supplies
  /// constants that must to be used (e.g. <c>LayerFactory::INPUT</c>,
  /// <c>LayerFactory::HIDDEN</c>, or <c>LayerFactory::OUTPUT</c>).
  /// </param name>
  /// <returns>Returns a pointer to a layer of the specified type.</returns>
  static std::unique_ptr<Layer> Layer::create(Type layer_type)
  {
    switch (layer_type)
    {
      case CONVOLUTIONAL:
        return std::make_unique<Convolutional>();
      case FULLY_CONNECTED:
        return std::make_unique<FullyConnected>();
      case OUTPUT:
        return std::make_unique<Output>();
      case POOLING:
        return std::make_unique<Pooling>();
    }
    throw runtime_error(layer_type + " is an invalid layer type. " +
      "Accepted types include: " +
      "LayerFactory::CONVOLUTIONAL, " +
      "LayerFactory::FULLY_CONNECTED, " +
      "LayerFactory::OUTPUT, " +
      "LayerFactory::POOLING." +);
  }

}
