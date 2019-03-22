
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

  /// <summary><c>get_instance</c> returns the singleton instance to give the
  /// Network class the ability to create new activation strategies.</summary>
  /// <returns>Returns the singleton instance of <c>ActivationFactory</c>.
  /// </returns>
  ActivationFactory& ActivationFactory::get_instance()
  {
    static ActivationFactory factory_instance;
    return factory_instance;
  }

  /// <summary><c>create</c> serves as a factory function that returns an
  /// activation function of the specified type.</summary>
  /// <param name="activation_type">The <c>ActivationFactory</c> class supplies
  /// constants that must to be used (e.g.
  /// <c>ActivationFactory::FAST_SIGMOID</c>,
  /// <c>ActivationFactory::LEAKY_RELU</c>,
  /// <c>ActivationFactory::LINEAR</c>,
  /// <c>ActivationFactory::RELU</c>,
  /// <c>ActivationFactory::SIGMOID</c>,
  /// <c>ActivationFactory::TANH</c>).</param name>
  /// <returns>Returns a reference to an activation function of the specified
  /// type.</returns>
  ActivationFunction& ActivationFactory::create(Type activation_type)
  {
    switch (activation_type)
    {
      case FAST_SIGMOID:
        static FastSigmoid fast_sigmoid_instance;
        return fast_sigmoid;
      case LEAKY_RELU:
        static LeakyReLU leaky_relu_instance;
        return leaky_relu_instance;
      case LINEAR:
        static Linear linear_instance;
        return linear_instance;
      case RELU:
        static ReLU relu_instance;
        return relu_instance;
      case TANH:
        static TanH tanh_instance;
        return tanh_instance;
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
