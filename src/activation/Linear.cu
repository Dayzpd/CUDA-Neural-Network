
#include "Linear.h"

namespace neural_network {

  /// <summary>Linear activation function.</summary>
  /// <param name="x">Represents the aggregated value obtained during
  /// forward propagation (bias + sum(inputs * weights)) for a given neuron
  /// </param name>
  /// <returns>Returns <c>x</c>.</returns>
  double Linear::calculate(double& x)
  {
    return x;
  }

  /// <summary>Derivative of Linear activation function.
  /// </summary>
  /// <param name="x">Represents the aggregated value obtained during
  /// forward propagation (bias + sum(inputs * weights)) for a given neuron
  /// </param name>
  /// <returns>Always returns 1 because the derivative of any function f(x)
  /// results in f'(x) = 1.</returns>
  double Linear::calculate_deriv(double& x)
  {
    return 1.0;
  }

}
