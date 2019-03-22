
#include "TanH.h"

#include <math.h>

namespace neural_network {

  /// <summary>Hyperbolic Tangent activation function.</summary>
  /// <param name="x">Represents the aggregated value obtained during
  /// forward propagation (bias + sum(inputs * weights)) for a given neuron.
  /// </param name>
  /// <returns>Returns the maximum of the input value and 0.</returns>
  double TanH::calculate(double& x)
  {
    return tanh(x);
  }

  /// <summary>Derivative of Hyperbolic Tangent activation function.
  /// </summary>
  /// <param name="x">Represents the aggregated value obtained during
  /// forward propagation (bias + sum(inputs * weights)) for a given neuron.
  /// </param name>
  /// <returns>Returns the derivative of tanh.</returns>
  double TanH::calculate_deriv(double& x)
  {
    return 1 - pow(tanh(x), 2);
  }

}
