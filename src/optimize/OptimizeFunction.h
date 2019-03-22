
#ifndef OPTIMIZE_FUNCTION_H
#define OPTIMIZE_FUNCTION_H

namespace neural_network
{

  class OptimizeFunction
  {
    public:
      virtual double calculate(double& x) = 0;
  }

}

#endif
