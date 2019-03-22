
#include "ActivationFunction.h"

namespace neural_network
{

  double ActivationFunction::calculate(double& x)
  {
    static_cast<const Derived*>(this)->calculate(x);
  }

  double ActivationFunction::calculate_deriv(double& x)
  {
    static_cast<const Derived*>(this)->calculate_deriv(x);
  }

}
