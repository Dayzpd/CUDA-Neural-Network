
namespace neural_network
{
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
  static std::unique_ptr<ActivationFunction> ActivationFunction::create(
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
      "ActivationFunctionFactory::FAST_SIGMOID, " +
      "ActivationFunctionFactory::LEAKY_RELU, " +
      "ActivationFunctionFactory::LINEAR, " +
      "ActivationFunctionFactory::RELU, " +
      "ActivationFunctionFactory::TANH.");
  }

}
