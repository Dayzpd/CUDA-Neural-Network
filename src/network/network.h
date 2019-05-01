
#ifndef NETWORK_H
#define NETWORK_H

#include "../loss/LossFunction.h"
#include "../layer/Layer.h"
#include "../neurons/Neurons.h"

#include <memory>
#include <string>
#include <vector>

namespace cuda_net
{

  class Network
  {
    private:
      std::vector<std::unique_ptr<Layer>> layers;

      Neurons prob;
      Neurons prob_delta;

      LossFunction* loss_func;

    public:
      Network(std::string loss_type);

      ~Network();

      // Activation Layer
      void add_layer(std::string layer_type);

      // Connection Layer
      void add_layer(std::string layer_type, size_t x, size_t y);

      void train();

      Neurons forward_propagate(Neurons& feature);

      void back_propagate(Neurons& prob, Neurons& actual, float learning_rate);
  };

}

#endif
