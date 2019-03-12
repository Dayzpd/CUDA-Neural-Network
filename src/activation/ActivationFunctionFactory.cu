#include "ActivationFunction.h"
#include "FastSigmoid.h"
#include "LeakyReLU.h"
#include "Linear.h"
#include "ReLU.h"
#include "TanH.h"

#include <memory>
#include <stdexcept>

namespace neural_network
{

  /// <summary>The <c>ActivationFunctionFactory</c> allows for dynamic creation
  /// of an activation function given a user-specified type. For more
  /// information on the design of this class, see Factories at
  /// en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns</summary>
  class ActivationFunctionFactory
  {
    public:
      enum ActivationType {
        FAST_SIGMOID, LEAKY_RELU, LINEAR, RELU, TANH
      };

      /// <summary><c>create</c> serves as the factory function for the
      /// <c>ActivationFunctionFactory</c> class that returns an activation
      /// function of the specified type. See
      /// en.cppreference.com/w/cpp/memory/unique_ptr/make_unique for more
      /// information.</summary>
      /// <param name="activation_type">The <c>ActivationFunctionFactory</c>
      /// class supplies constants that must to be used (e.g.
      /// <c>ActivationFunctionFactory::FAST_SIGMOID</c>,
      /// <c>ActivationFunctionFactory::LEAKY_RELU</c>,
      /// <c>ActivationFunctionFactory::LINEAR</c>,
      /// <c>ActivationFunctionFactory::RELU</c>,
      /// <c>ActivationFunctionFactory::SIGMOID</c>,
      /// <c>ActivationFunctionFactory::TANH</c>).</param name>
      /// <returns>Returns a pointer to an activation function of the specified
      /// type.</returns>
      static std::unique_ptr<ActivationFunction> create(
        ActivationType activation_type
      ) {
        switch (activation_type)
        {
          case FAST_SIGMOID:
            return std::make_unique<FastSigmoid>();
          case LEAKY_RELU:
            return std::make_unique<LeakyReLU>();
          case LINEAR:
            return std::make_unique<Linear>();
          case RELU:
            return std::make_unique<ReLU>();
          case TANH:
            return std::make_unique<TanH>();
        }
        throw runtime_error("An invalid activation function type was given. " +
          "Accepted types include: " +
          "ActivationFunctionFactory::FAST_SIGMOID, " +
          "ActivationFunctionFactory::LEAKY_RELU, " +
          "ActivationFunctionFactory::LINEAR, " +
          "ActivationFunctionFactory::RELU, " +
          "ActivationFunctionFactory::TANH.");
      }
  }

}
