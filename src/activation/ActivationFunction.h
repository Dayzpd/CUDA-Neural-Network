
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

namespace neural_network
{

  template <typename Derived>
  class ActivationFunction
  {
    public:
      double calculate(double& x);

      double calculate_deriv(double& x);
  }

}

#endif
