
#ifndef ACTIVATOR_H
#define ACTIVATOR_H

namespace neural_network {

  class ActivationFunction
  {
    public:
      ActivationFunction();

      virtual ~ActivationFunction();

      virtual double calculate(double& x) = 0;
  }

}

#endif
