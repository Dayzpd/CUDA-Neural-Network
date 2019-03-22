
#include "ActivationFactory.h"
#include "FastSigmoid.h"
#include "LeakyReLU.h"
#include "Linear.h"
#include "ReLU.h"
#include "TanH.h"

#include <memory>
#include <stdexcept>

namespace neural_network
{
  /// <summary><c>create</c> serves as a factory function that returns an
  /// activation function of the specified type.</summary>
  /// <param name="activation_type">The <c>ActivationFactory</c>
  /// class supplies constants that must to be used (e.g.
  /// <c>ActivationFactory::FAST_SIGMOID</c>,
  /// <c>ActivationFactory::LEAKY_RELU</c>,
  /// <c>ActivationFactory::LINEAR</c>,
  /// <c>ActivationFactory::RELU</c>,
  /// <c>ActivationFactory::SIGMOID</c>,
  /// <c>ActivationFactory::TANH</c>).</param name>
  /// <returns>Returns a pointer to an activation function of the specified
  /// type.</returns>
  std::unique_ptr<ActivationFunction> ActivationFactory::create(
    Type activation_type
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
      "ActivationFactory::FAST_SIGMOID, " +
      "ActivationFactory::LEAKY_RELU, " +
      "ActivationFactory::LINEAR, " +
      "ActivationFactory::RELU, " +
      "ActivationFactory::TANH.");
  }

}
