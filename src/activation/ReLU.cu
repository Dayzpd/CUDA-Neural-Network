
#include "ActivationFunction.h"
#include "ReLU.h"

namespace neural_network {

  class ReLU : public ActivationFunction
  {
    public:

      ReLU();

      ~ReLU();

      /// <summary>Rectified Linear Unit activation function.</summary>
      /// <param name="x">Represents the aggregated value obtained during
      /// forward propagation (bias + sum(inputs * weights)) for a given neuron
      /// </param name>
      /// <returns>Returns the maximum of the input value and 0.</returns>
      double calculate(double& x)
      {
        return std::max(0, x);
      }
  }

}
