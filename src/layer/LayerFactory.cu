
#include "Layer.h"
#include "LayerFactory.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"

#include <memory>
#include <stdexcept>

namespace neural_network
{

  /// <summary>The <c>LayerFactory</c> allows for dynamic creation of a layer
  /// given a user-specified type. For more information on the design of this
  /// class, see Factories at
  /// en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns</summary>
  class LayerFactory
  {
    public:

      enum LayerType {
        INPUT, HIDDEN, OUTPUT
      };

      /// <summary><c>create</c> serves as the factory function for the
      /// <c>LayerFactory</c> class that returns a layer of the specified type.
      /// See en.cppreference.com/w/cpp/memory/unique_ptr/make_unique for more
      /// information.</summary>
      /// <param name="layer_type">The <c>LayerFactory</c> class supplies
      /// constants that must to be used (e.g. <c>LayerFactory::INPUT</c>,
      /// <c>LayerFactory::HIDDEN</c>, or <c>LayerFactory::OUTPUT</c>).
      /// </param name>
      /// <returns>Returns a pointer to a layer of the specified type.</returns>
      static std::unique_ptr<Layer> create_layer(LayerType layer_type)
      {
        switch (layer_type)
        {
          case INPUT:
            return std::make_unique<InputLayer>();
          case HIDDEN:
            return std::make_unique<HiddenLayer>();
          case OUTPUT:
            return std::make_unique<OutputLayer>();
        }
        throw runtime_error(layer_type + " is an invalid layer type. " +
          "Accepted types include: LayerFactory::INPUT, LayerFactory::HIDDEN," +
          " LayerFactory::OUTPUT.");
      }
  }

}
