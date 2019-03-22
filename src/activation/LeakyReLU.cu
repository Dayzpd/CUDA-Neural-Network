
#include "LeakyReLU.h"

namespace neural_network {

    /// <summary>Leaky Rectified Linear Unit activation function.</summary>
    /// <param name="x">Represents the aggregated value obtained during
    /// forward propagation (bias + sum(inputs * weights)) for a given neuron
    /// </param name>
    /// <returns>Returns <c>x</c> if the input value <c>x</c> is >= 0, else
    /// <c>0.01 * x</c> is returned.</returns>
    double LeakyReLU::calculate(double& x)
    {
      if (x >= 0)
      {
        return x;
      }
      else
      {
        return 0.01 * x;
      }
    }

    /// <summary>Derivative of Leaky Rectified Linear Unit activation
    /// function.</summary>
    /// <param name="x">Represents the aggregated value obtained during
    /// forward propagation (bias + sum(inputs * weights)) for a given neuron
    /// </param name>
    /// <returns>Returns 1 if the input value <c>x</c> is >= 0, else 0.01 is
    /// returned.</returns>
    double LeakyReLU::calculate_deriv(double& x)
    {
      if (x >= 0)
      {
        return 1.0;
      }
      else
      {
        return 0.01;
      }
    }

}
