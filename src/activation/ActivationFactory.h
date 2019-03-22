
#ifndef ACTIVATION_FACTORY_H
#define ACTIVATION_FACTORY_H

namespace neural_network
{

  class ActivationFactory
  {
    private:
      ActivationFactory() {}

    public:
      enum Type {
        FAST_SIGMOID = "FAST_SIGMOID",
        LEAKY_RELU = "LEAKY_RELU",
        LINEAR = "LINEAR",
        RELU = "RELU",
        TANH = "TANH"
      };

      static ActivationFactory& get_instance();

      static ActivationFunction& create(Type activation_type);
  }

}

#endif
