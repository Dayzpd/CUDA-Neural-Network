
#include "ActivationFunction.h"
#include "ReLU.h"

namespace neural_network {

  ReLU::ReLU();

  ReLU::~ReLU();

  /// <summary>Rectified Linear Unit activation function.</summary>
  /// <param name="x">Represents the aggregated value obtained during
  /// forward propagation (bias + sum(inputs * weights)) for a given neuron.
  /// </param name>
  /// <returns>Returns the maximum of the input value and 0.</returns>
  double ReLU::calculate(double& x)
  {
    return std::max(0, x);
  }

  /// <summary>Derivative of Rectified Linear Unit activation function.
  /// </summary>
  /// <param name="x">Represents the aggregated value obtained during
  /// forward propagation (bias + sum(inputs * weights)) for a given neuron.
  /// </param name>
  /// <returns>Returns the 1 if the input value <c>x</c> is >= 0, else 0 is
  /// returned.</returns>
  int ReLU::calculate_deriv(double& x)
  {
    if (x >= 0)
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }

}
