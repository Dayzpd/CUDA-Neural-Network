
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

namespace neural_network {

  class ActivationFunction
  {
    public:
      ActivationFunction();

      virtual ~ActivationFunction();

      virtual double calculate(double& x) = 0;

      virtual double calculate_deriv(double& x);

  }

}

#endif
