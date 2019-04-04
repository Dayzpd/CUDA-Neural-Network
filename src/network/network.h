
#ifndef NETWORK_H
#define NETWORK_H

#include "../loss/LossFunction.h"
#include "../layer/Layer.h"
#include "../optimize/OptimizeFunction.h"

#include <string>
#include <tuple>
#include <vector>

namespace neural_network
{

  class Network
  {
    private:
      LayerFactory& layer_factory;
      LossFunction& loss_func;
      OptimizeFunction& optimize_func;

      std::vector<Layer*> layers;

      std::vector<tuple<Neurons, Neurons>> features;

      int num_classes;

    public:
      Network(int num_classes, float learning_rate, std::string loss_type,
        std::string optimize_type);

      ~Network();

      void add_feature(Neurons feature, Neurons class);

      // Activation Layer
      void add_layer(std::string layer_type);

      // Connection Layer
      void add_layer(std::string layer_type, size_t x, size_t y);

      int get_num_classes();

      void train(Features& features, int batch_size, float learning_rate);

      void forward_propagate();

      void back_propagate();

      double calculate_loss();
  }

}

#endif
