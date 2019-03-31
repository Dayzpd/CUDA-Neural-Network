
#ifndef NETWORK_H
#define NETWORK_H

#include "../loss/LossFunction.h"
#include "../layer/Layer.h"
#include "../optimize/OptimizeFunction.h"

#include <string>
#include <vector>

namespace neural_network
{

  class Network
  {
    private:
      LayerFactory& layer_factory;

      std::vector<Layer*> layers;

    public:
      Network();

      ~Network();

      // Activation Layer
      void add_layer(std::string layer_type);

      // Connection Layer
      void add_layer(std::string layer_type, size_t x, size_t y);

      void train(Features& features, int batch_size, float learning_rate);

      void forward_propagate();

      void back_propagate();

      double calculate_loss();
  }

}

#endif
